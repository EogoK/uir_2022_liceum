// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stefan Paquay (Eindhoven University of Technology)
------------------------------------------------------------------------- */

#include "pair_sph_rhosum_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define KOKKOS_CUDA_MAX_THREADS 256
#define KOKKOS_CUDA_MIN_BLOCKS 8

#define MAXLINE 1024
#define DELTA 4


/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSPHRhosumKokkos<DeviceType>::PairSPHRhosumKokkos(LAMMPS *lmp) : PairSPHRhoSum(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSPHRhosumKokkos<DeviceType>::~PairSPHRhosumKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
    

  }
}

/* ---------------------------------------------------------------------- */


template<class DeviceType>
void PairSPHRhosumKokkos<DeviceType>::coeff(int narg, char** arg) {
     if (narg != 3)
    error->all(FLERR,"Incorrect number of args for sph/rhosum coefficients");


  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR,arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR,arg[1], 1, atom->ntypes, jlo, jhi, error);

  double cut_one = utils::numeric(FLERR,arg[2],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
       k_params.h_view(i,j).cut = cut_one;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
}

template<class DeviceType>
void PairSPHRhosumKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  //add
  mass = atomKK->k_mass.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  //
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;

  inum = list->inum;
  //const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_ilist = k_list->d_ilist; //non-declared
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  
  EV_FLOAT ev;
  EV_FLOAT ev_all;
  
  //
  if(nstep != 0){
    if ((update->ntimestep % nstep) == 0){
          // initialize density with self-contribution,
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHRhosumComputeShortNeigh<FULL,1> >(0, inum), *this);
      ev_all += ev;
      // add density at each atom via kernel function overlap
       Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHRhosumComputeShortNeigh1<FULL,1> >(0, inum), *this);

    }
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHRhosumKokkos<DeviceType>::operator()(TagPairSPHRhosumComputeShortNeigh<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const{
  const int i = d_ilist[ii];
  const int itype = type[i];
  const X_FLOAT imass = mass[itype];

  const X_FLOAT h = k_params.h_view(i, i).cut;
  X_FLOAT wf;
  if (domain->dimension == 3) {
  /*
  // Lucy kernel, 3d
  wf = 2.0889086280811262819e0 / (h * h * h);
  */

  // quadric kernel, 3d
  wf = 2.1541870227086614782 / (h * h * h);
  } else {
  /*
  // Lucy kernel, 2d
  wf = 1.5915494309189533576e0 / (h * h);
  */

  // quadric kernel, 2d
  wf = 1.5915494309189533576e0 / (h * h);
  }
  rho[i] = imass * wf;
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHRhosumKokkos<DeviceType>::operator()(TagPairSPHRhosumComputeShortNeigh<NEIGHFLAG,EVFLAG>, const int& ii) const{

  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairSPHRhosumComputeShortNeigh<NEIGHFLAG,EVFLAG>(), ii, ev);

}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHRhosumKokkos<DeviceType>::operator()(TagPairSPHRhosumComputeShortNeigh1<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const{
  const int i = d_ilist[ii];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const int itype = type[i];
  //const int* jlist = d_neighbors.h_view(i); //pleas find!
  const int jnum = d_numneigh[i];

  X_FLOAT wf;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    j &= NEIGHMASK;

    const int jtype = type[j];
    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    if (rsq < k_params.h_view(itype, jtype).cutsq) {
      const X_FLOAT h = k_params.h_view(itype, jtype).cut;
      const X_FLOAT ih = 1.0 / h;
      const X_FLOAT ihsq = ih * ih;

      if (domain->dimension == 3) {
        /*
        // Lucy kernel, 3d
        r = sqrt(rsq);
        wf = (h - r) * ihsq;
        wf =  2.0889086280811262819e0 * (h + 3. * r) * wf * wf * wf * ih;
        */

        // quadric kernel, 3d
        wf = 1.0 - rsq * ihsq;
        wf = wf * wf;
        wf = wf * wf;
        wf = 2.1541870227086614782e0 * wf * ihsq * ih;
      } else {
        // Lucy kernel, 2d
        //r = sqrt(rsq);
        //wf = (h - r) * ihsq;
        //wf = 1.5915494309189533576e0 * (h + 3. * r) * wf * wf * wf;

        // quadric kernel, 2d
        wf = 1.0 - rsq * ihsq;
        wf = wf * wf;
        wf = wf * wf;
        wf = 1.5915494309189533576e0 * wf * ihsq;
      }

      rho[i] += mass[jtype] * wf;
    }
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHRhosumKokkos<DeviceType>::operator()(TagPairSPHRhosumComputeShortNeigh1<NEIGHFLAG,EVFLAG>, const int& ii) const{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairSPHRhosumComputeShortNeigh1<NEIGHFLAG,EVFLAG>(), ii, ev);
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhosumKokkos<DeviceType>::allocate()
{
  PairSPHRhoSum::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_sph**,Kokkos::LayoutRight,DeviceType>("PairSPHRhoSum::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhosumKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg > 2) error->all(FLERR,"Illegal pair_style command");

  PairSPHRhoSum::settings(1,arg);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhosumKokkos<DeviceType>::init_style()
{
  PairSPHRhoSum::init_style();

  // error if rRESPA with inner levels

  // if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
  //   int respa = 0;
  //   if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
  //   if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
  //   if (respa) 
  //     error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  // }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  if (neighflag == FULL) request->enable_full();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
// Rewrite this.
template<class DeviceType>
double PairSPHRhosumKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairSPHRhoSum::init_one(i,j);

  // k_params.h_view(i,j).d0     = d0[i][j];
  // k_params.h_view(i,j).alpha  = alpha[i][j];
  // k_params.h_view(i,j).r0     = r0[i][j];
  // k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cutsq  = cutone*cutone;
  k_params.h_view(i,j).cut  = cutone;
  k_params.h_view(j,i)        = k_params.h_view(i,j);

  // if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
  //   m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
  //   m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
  // }

  // k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}



namespace LAMMPS_NS {
template class PairSPHRhosumKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSPHRhosumKokkos<LMPHostType>;
#endif
}

