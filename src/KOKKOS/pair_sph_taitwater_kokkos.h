/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sph/kk,PairSPHKokkos<LMPDeviceType>);
PairStyle(sph/kk/device,PairSPHKokkos<LMPDeviceType>);
PairStyle(sph/kk/host,PairSPHKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_MORSE_KOKKOS_H
#define LMP_PAIR_MORSE_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_sph_taitwater.h"
#include "neigh_list_kokkos.h"

template<int NEIGHFLAG, int EVFLAG>
struct TagPairKokkosTaitwater{};

namespace LAMMPS_NS {

template<class DeviceType>
class PairSPHTaitwaterKokkos : public PairSPH {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  PairSPHKokkos(class LAMMPS *);
  ~PairSPHKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void coeff(int, char **) override;

  
  struct params_sph{
    KOKKOS_INLINE_FUNCTION
    params_sph() {cutsq=0,rho0=0;soundspeed=0;B=0;viscosity=0;}
    KOKKOS_INLINE_FUNCTION
    params_sph(int /*i*/) {cutsq=0,rho0=0;soundspeed=0;B=0;viscosity=0;}
    F_FLOAT cutsq,rho0,soundspeed,B,viscosity;
  };

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;


  template<class DeviceType>
  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairKokkosTaitwater<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  Kokkos::DualView<params_sph**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_sph**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_sph m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_x_array c_x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;

  double special_lj[4];

  typename ArrayTypes<DeviceType>::tdual_ffloat_2d k_cutsq;
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq;

  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_ilist;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_numneigh;
  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  void allocate() override;
  friend struct PairComputeFunctor<PairSPHKokkos,FULL,true>;
  friend struct PairComputeFunctor<PairSPHKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairSPHKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairSPHKokkos,FULL,false>;
  friend struct PairComputeFunctor<PairSPHKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairSPHKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairSPHKokkos,FULL,void>(PairSPHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairSPHKokkos,HALF,void>(PairSPHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairSPHKokkos,HALFTHREAD,void>(PairSPHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairSPHKokkos,void>(PairSPHKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairSPHKokkos>(PairSPHKokkos*);
};

}

#endif
#endif