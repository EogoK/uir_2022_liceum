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

#include "fix_sph_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixSPHKokkos<DeviceType>::FixSPHKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixSPH(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = X_MASK | V_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::init()
{
  FixSPH::init();

  atomKK->k_mass.modify<LMPHostType>();
  atomKK->k_mass.sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->modified(execution_space,datamask_modify);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  esph = atomKK->k_esph.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();
  vest = atomKK->k_vest.view<DeviceType>();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  if (rmass.data()) {
    FixSPHKokkosInitialIntegrateFunctor<DeviceType,1> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  } else {
    FixSPHKokkosInitialIntegrateFunctor<DeviceType,0> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::initial_integrate_item(int i) const
{
  if (mask[i] & groupbit) {
    const double dtfm = dtf / mass[type[i]];

    esph(i) += dtf * desph(i); // half-step update of particle internal energy
    rho(i) += dtf * drho(i); // ... and density

    // extrapolate velocity for use with velocity-dependent potentials, e.g. SPH
    vest(i,0) = v(i,0) + 2.0 * dtfm * f(i,0);
    vest(i,1) = v(i,1) + 2.0 * dtfm * f(i,1);
    vest(i,2) = v(i,2) + 2.0 * dtfm * f(i,2);

    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::initial_integrate_rmass_item(int i) const
{
  if (mask[i] & groupbit) {
    const double dtfm = dtf / rmass[i];

    esph(i) += dtf * desph(i); // half-step update of particle internal energy
    rho(i) += dtf * drho(i); // ... and density

    // extrapolate velocity for use with velocity-dependent potentials, e.g. SPH
    vest(i,0) = v(i,0) + 2.0 * dtfm * f(i,0);
    vest(i,1) = v(i,1) + 2.0 * dtfm * f(i,1);
    vest(i,2) = v(i,2) + 2.0 * dtfm * f(i,2);

    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::final_integrate()
{
  atomKK->sync(execution_space,V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
  atomKK->modified(execution_space,V_MASK);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  esph = atomKK->k_esph.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  if (rmass.data()) {
    FixSPHKokkosFinalIntegrateFunctor<DeviceType,1> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  } else {
    FixSPHKokkosFinalIntegrateFunctor<DeviceType,0> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  }

  // debug
  //atomKK->sync(Host,datamask_read);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::final_integrate_item(int i) const
{
  if (mask[i] & groupbit) {
    const double dtfm = dtf / mass[type[i]];

    esph(i) += dtf * desph(i);
    rho(i) += dtf * drho(i); 

    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::final_integrate_rmass_item(int i) const
{
  if (mask[i] & groupbit) {
    const double dtfm = dtf / rmass[i];

    esph(i) += dtf * desph(i);
    rho(i) += dtf * drho(i); 

    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::cleanup_copy()
{
  id = style = nullptr;
  vatom = nullptr;
}

namespace LAMMPS_NS {
template class FixSPHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSPHKokkos<LMPHostType>;
#endif
}

