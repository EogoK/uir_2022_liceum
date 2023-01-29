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
PairStyle(sph/heatconduction/kk,PairSPHHeatConductionKokkos<LMPDeviceType>);
PairStyle(sph/heatconduction/kk/device,PairSPHHeatConductionKokkos<LMPDeviceType>);
PairStyle(sph/heatconduction/kk/host,PairSPHHeatConductionKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_SPH_HEATCONDUCTION_KOKKOS_H
#define LMP_PAIR_SPH_HEATCONDUCTION_KOKKOS_H

#include "pair_sph_headconduction.h"
#include "pair_kokkos.h"

template<int NEIGHFLAG, int EVFLAG>
struct TagPairKokkosHeatConduction{};

namespace LAMMPS_NS {
template<class DeviceType>
class PairSPHHeatConductionKokkos: public PairSPHHeatConduction{
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  PairSPHHeatConductionKokkos(class LAMMPS *);
  ~PairSPHHeatConductionKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void coeff(int, char **) override;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairKokkosHeatConduction<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;


  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairKokkosHeatConduction<NEIGHFLAG,EVFLAG>, const int&) const;


  struct params_sph{
    KOKKOS_INLINE_FUNCTION
    params_sph() {cut=0,cutsq=0,alpha=0;}
    KOKKOS_INLINE_FUNCTION
    params_sph(int /*i*/) {cut=0,cutsq=0,alpha=0;}
    F_FLOAT cut,alpha,cutsq;
  };


 protected:

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

  typename ArrayTypes<DeviceType>::t_float_1d mass;
  typename ArrayTypes<DeviceType>::t_float_1d desph;
  typename ArrayTypes<DeviceType>::t_float_1d esph;
  //typename ArrayTypes<DeviceType>::t_float_1d drho;
  
  typename ArrayTypes<DeviceType>::t_efloat_1d rho;
  //typename ArrayTypes<DeviceType>::t_float_2d v;


  double special_lj[4];

  typename ArrayTypes<DeviceType>::tdual_ffloat_2d k_cutsq;
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq;

  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_ilist;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_numneigh;
  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  void allocate();
  // friend struct PairComputeFunctor<PairSPHHeatConductionKokkos,FULL,true>;
  // friend struct PairComputeFunctor<PairSPHHeatConductionKokkos,HALF,true>;
  // friend struct PairComputeFunctor<PairSPHHeatConductionKokkos,HALFTHREAD,true>;
  // friend struct PairComputeFunctor<PairSPHHeatConductionKokkos,FULL,false>;
  // friend struct PairComputeFunctor<PairSPHHeatConductionKokkos,HALF,false>;
  // friend struct PairComputeFunctor<PairSPHHeatConductionKokkos,HALFTHREAD,false>;
  // friend EV_FLOAT pair_compute_neighlist<PairSPHHeatConductionKokkos,FULL,void>(PairSPHHeatConductionKokkos*,NeighListKokkos<DeviceType>*);
  // friend EV_FLOAT pair_compute_neighlist<PairSPHHeatConductionKokkos,HALF,void>(PairSPHHeatConductionKokkos*,NeighListKokkos<DeviceType>*);
  // friend EV_FLOAT pair_compute_neighlist<PairSPHHeatConductionKokkos,HALFTHREAD,void>(PairSPHHeatConductionKokkos*,NeighListKokkos<DeviceType>*);
  // friend EV_FLOAT pair_compute<PairSPHHeatConductionKokkos,void>(PairSPHHeatConductionKokkos*,NeighListKokkos<DeviceType>*);
  // friend void pair_virial_fdotr_compute<PairSPHHeatConductionKokkos>(PairSPHHeatConductionKokkos*);
};

}

#endif
#endif