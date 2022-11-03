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
PairStyle(sph/kk,PairSPHRhosumKokkos<LMPDeviceType>);
PairStyle(sph/kk/device,PairSPHRhosumKokkos<LMPDeviceType>);
PairStyle(sph/kk/host,PairSPHRhosumKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_MORSE_KOKKOS_H
#define LMP_PAIR_MORSE_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_sph_rhosum.h"
#include "neigh_list_kokkos.h"

template<int NEIGHFLAG, int EVFLAG>
struct TagPairSPHRhosumComputeShortNeigh{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairSPHRhosumComputeShortNeigh1{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairSPHRhosumComputeShortNeigh2{};


namespace LAMMPS_NS {

template<class DeviceType>
class PairSPHRhosumKokkos : public PairSPH {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  PairSPHRhosumKokkos(class LAMMPS *);
  ~PairSPHRhosumKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
 
  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHRhosumComputeShortNeigh<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;
  
  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHRhosumComputeShortNeigh1<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHRhosumComputeShortNeigh2<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;
    

 protected:
  Kokkos::DualView<params_sph**,Kokkos::LayoutRight,DeviceType> k_params;
  t_param_1d d_params;
  typename Kokkos::DualView<params_sph**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_sph m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_x_array c_x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type;


  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_ilist;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;

  double special_lj[4];
  int nstep, first;

  typename ArrayTypes<DeviceType>::tdual_ffloat_2d k_cutsq;
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq;


  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  int inum;
  void allocate() override;
  friend struct PairComputeFunctor<PairSPHRhosumKokkos,FULL,true>;
  friend struct PairComputeFunctor<PairSPHRhosumKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairSPHRhosumKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairSPHRhosumKokkos,FULL,false>;
  friend struct PairComputeFunctor<PairSPHRhosumKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairSPHRhosumKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairSPHRhosumKokkos,FULL,void>(PairSPHRhosumKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairSPHRhosumKokkos,HALF,void>(PairSPHRhosumKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairSPHRhosumKokkos,HALFTHREAD,void>(PairSPHRhosumKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairSPHRhosumKokkos,void>(PairSPHRhosumKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairSPHRhosumKokkos>(PairSPHRhosumKokkos*);
};

}

#endif
#endif