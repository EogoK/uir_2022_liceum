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

#include "pair_sph_taitwater_kokkos_morris.h"

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
PairSPHTaitwaterMorrisKokkos<DeviceType>::PairSPHTaitwaterMorrisKokkos(LAMMPS *lmp) : PairSPHTaitwater(lmp)
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
PairSPHTaitwaterMorrisKokkos<DeviceType>::~PairSPHTaitwaterMorrisKokkos()
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
void PairSPHTaitwaterMorrisKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  //add
  mass = atomKK->k_mass.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  v = atomKK->k_vest.view<DeviceType>();
  //
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  const int inum = list->inum;
  //const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_ilist = k_list->d_ilist; //non-declared
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  EV_FLOAT ev;
  //d_params = k_params.template view<DeviceType>();

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairKokkosTaitwater<HALF,1> >(0,inum),*this);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  // if (vflag_fdotr) pair_virial_fdotr_compute(this);

  // if (eflag_atom) {
  //   k_eatom.template modify<DeviceType>();
  //   k_eatom.template sync<LMPHostType>();
  // }

  // if (vflag_atom) {
  //   k_vatom.template modify<DeviceType>();
  //   k_vatom.template sync<LMPHostType>();
  // } //??????????????????????
}

// template<class DeviceType>
// template<bool STACKPARAMS, class Specialisation>
// KOKKOS_INLINE_FUNCTION
// F_FLOAT PairSPHTaitwaterMorrisKokkos<DeviceType>::
// compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
//   (void) i;
//   (void) j;
//   const F_FLOAT rr = sqrt(rsq);
//   const F_FLOAT r0 = STACKPARAMS ? m_params[itype][jtype].r0 : params(itype,jtype).r0;
//   const F_FLOAT d0 = STACKPARAMS ? m_params[itype][jtype].d0 : params(itype,jtype).d0;
//   const F_FLOAT aa = STACKPARAMS ? m_params[itype][jtype].alpha : params(itype,jtype).alpha;
//   const F_FLOAT dr = rr - r0;

//   // U  =  d0 * [ exp( -2*a*(x-r0)) - 2*exp(-a*(x-r0)) ]
//   // f  = -2*a*d0*[ -exp( -2*a*(x-r0) ) + exp( -a*(x-r0) ) ] * grad(r)
//   //    = +2*a*d0*[  exp( -2*a*(x-r0) ) - exp( -a*(x-r0) ) ] * grad(r)
//   const F_FLOAT dexp    = exp( -aa*dr );
//   const F_FLOAT forcelj = 2*aa*d0*dexp*(dexp-1.0);

//   return forcelj / rr;
// }

// template<class DeviceType>
// template<bool STACKPARAMS, class Specialisation>
// KOKKOS_INLINE_FUNCTION
// F_FLOAT PairSPHTaitwaterMorrisKokkos<DeviceType>::
// compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
//   (void) i;
//   (void) j;
//   const F_FLOAT rr = sqrt(rsq);
//   const F_FLOAT r0 = STACKPARAMS ? m_params[itype][jtype].r0 : params(itype,jtype).r0;
//   const F_FLOAT d0 = STACKPARAMS ? m_params[itype][jtype].d0 : params(itype,jtype).d0;
//   const F_FLOAT aa = STACKPARAMS ? m_params[itype][jtype].alpha : params(itype,jtype).alpha;
//   const F_FLOAT dr = rr - r0;

//   // U  =  d0 * [ exp( -2*a*(x-r0)) - 2*exp(-a*(x-r0)) ]
//   // f  = -2*a*d0*[ -exp( -2*a*(x-r0) ) + exp( -a*(x-r0) ) ] * grad(r)
//   //    = +2*a*d0*[  exp( -2*a*(x-r0) ) - exp( -a*(x-r0) ) ] * grad(r)
//   const F_FLOAT dexp    = exp( -aa*dr );

//   return d0 * dexp * ( dexp - 2.0 );
// }

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::allocate()
{
  PairSPHTaitwater::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_sph**,Kokkos::LayoutRight,DeviceType>("PairSPHTaitwater::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg > 2) error->all(FLERR,"Illegal pair_style command");

  PairSPHTaitwater::settings(1,arg);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::init_style()
{
  PairSPHTaitwater::init_style();

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

/*----------------------------------------------------------------------
   coef init ploho
-------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
// Rewrite this.
template<class DeviceType>
double PairSPHTaitwaterMorrisKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairSPHTaitwater::init_one(i,j);

  k_params.h_view(i,j).cut     = cut[i][j];
  k_params.h_view(i,j).viscosity  = viscosity[i][j];//maybe create copy of k_params to d_params and get valus
  k_params.h_view(j,i)        = k_params.h_view(i,j);

  // if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
  //   m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
  // }

  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::coeff(int narg, char **arg){
  if (narg != 6)
    error->all(FLERR,
        "Incorrect args for pair_style sph/taitwater coefficients");

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR,arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR,arg[1], 1, atom->ntypes, jlo, jhi, error);

  double rho0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double soundspeed_one = utils::numeric(FLERR,arg[3],false,lmp);
  double viscosity_one = utils::numeric(FLERR,arg[4],false,lmp);
  double cut_one = utils::numeric(FLERR,arg[5],false,lmp);
  double B_one = soundspeed_one * soundspeed_one * rho0_one / 7.0;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k_params.h_view(i, 0).rho0 = rho0_one;
    k_params.h_view(i, 0).soundspeed  = soundspeed_one;
    k_params.h_view(i, 0).B = B_one;
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      k_params.h_view(i, j).viscosity = viscosity_one;
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      k_params.h_view(i, j).cut = cut_one;
      //cut[j][i] = cut[i][j];
      //viscosity[j][i] = viscosity[i][j];
      //setflag[j][i] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");


} 

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHTaitwaterMorrisKokkos<DeviceType>::operator()(TagPairKokkosTaitwater<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const{
  const int i = d_ilist[ii];
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);
  const X_FLOAT vxtmp = v(i, 0);
  const X_FLOAT vytmp = v(i, 1);
  const X_FLOAT vztmp = v(i, 2);
  const int itype = type[i];
  const int jnum = d_numneigh[i];

  const X_FLOAT imass = mass[itype];

  const X_FLOAT tmp = rho[i] / k_params.h_view(itype, 0).rho0;
  X_FLOAT fi = tmp * tmp * tmp;
  fi = k_params.h_view(itype, 0).B * (fi * fi * tmp - 1.0) / (rho[i] * rho[i]);

    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors(i, jj);
      j &= NEIGHMASK;

      const F_FLOAT delx = xtmp - x(j, 0);
      const F_FLOAT dely = ytmp - x(j, 1);
      const F_FLOAT delz = ztmp - x(j, 2);
      const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;
      const int jtype = type[j];
      const X_FLOAT jmass = mass[jtype];

      if (rsq < k_params.h_view(itype, jtype).cutsq) {

        F_FLOAT h = k_params.h_view(itype, jtype).cut;
        F_FLOAT ih = 1.0 / h;
        F_FLOAT ihsq = ih * ih;

        F_FLOAT wfd = h - sqrt(rsq);
        if (domain->dimension == 3) {
          // Lucy Kernel, 3d
          // Note that wfd, the derivative of the weight function with respect to r,
          // is lacking a factor of r.
          // The missing factor of r is recovered by
          // (1) using delV . delX instead of delV . (delX/r) and
          // (2) using f[i][0] += delx * fpair instead of f[i][0] += (delx/r) * fpair
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih; 
        } else {
          // Lucy Kernel, 2d
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
        }

        // compute pressure  of atom j with Tait EOS
        F_FLOAT tmp = rho[j] / k_params.h_view(jtype, 0).rho0;
        F_FLOAT fj = tmp * tmp * tmp;
        fj = k_params.h_view(jtype, 0).B * (fj * fj * tmp - 1.0) / (rho[j] * rho[j]);
        
        F_FLOAT velx = vxtmp - v(j, 0);
        F_FLOAT vely = vytmp - v(j, 1);
        F_FLOAT velz = vztmp - v(j, 2);

        // dot product of velocity delta and distance vector
        F_FLOAT delVdotDelR = delx * velx + dely * vely + delz * velz;;

        // artificial viscosity (Monaghan 1992)
        F_FLOAT fvisc = 2 * viscosity[itype][jtype] / (rho[i] * rho[j]);

        fvisc *= imass * jmass * wfd;

        // total pair force & thermal energy increment
        F_FLOAT fpair = -imass * jmass * (fi + fj) * wfd;
        F_FLOAT deltaE = -0.5 *(fpair * delVdotDelR + fvisc * (velx*velx + vely*vely + velz*velz));

       // printf("testvar= %f, %f \n", delx, dely);

        f(i, 0) += delx * fpair + velx * fvisc;
        f(i, 1) += dely * fpair + vely * fvisc;
        f(i, 2) += delz * fpair + velz * fvisc;
        // and change in density
        drho[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        desph[i] += deltaE;

        if (newton_pair || j < nlocal) {
          f(i, 0) -= delx * fpair + velx * fvisc;
          f(i, 1) -= dely * fpair + vely * fvisc;
          f(i, 2) -= delz * fpair + velz * fvisc;
          desph[j] += deltaE;
          drho[j] += imass * delVdotDelR * wfd;
        }

        // if (evflag)
        //   PairComputeFunctor<Meso, HALF, STACKPARAMS, void>ev_tally(ev, i, j, 0.0, fpair, delx, dely, delz);//maybe not work
      }
    }
}


template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHTaitwaterMorrisKokkos<DeviceType>::operator()(TagPairKokkosTaitwater<NEIGHFLAG,EVFLAG>, const int& ii) const{
    EV_FLOAT ev;
    this->template operator()<NEIGHFLAG,EVFLAG>(TagPairKokkosTaitwater<NEIGHFLAG,EVFLAG>(), ii, ev);

}



namespace LAMMPS_NS {
template class PairSPHTaitwaterMorrisKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSPHTaitwaterMorrisKokkos<LMPHostType>;
#endif
}

