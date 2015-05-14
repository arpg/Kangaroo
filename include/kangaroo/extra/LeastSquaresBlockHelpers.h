#pragma once

#include <Eigen/Eigen>

//! Update JTJ and JTy with sparse outer product of NR sparse rows of J.
//! J = (.., J0, .., J1, .., J2, .. )
template<typename T, unsigned NR, unsigned NJ0, unsigned NJ1, unsigned NJ2 >
inline void AddSparseOuterProduct(
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& JTJ,
        Eigen::Matrix<T,Eigen::Dynamic,1>& JTy,
        const Eigen::Matrix<T,NR,NJ0>& J0, int J0pos,
        const Eigen::Matrix<T,NR,NJ1>& J1, int J1pos,
        const Eigen::Matrix<T,NR,NJ2>& J2, int J2pos,
        const Eigen::Matrix<T,NR,1>& y
) {
    // On diagonal blocks
    if(NJ0) JTJ.template block<NJ0,NJ0>(J0pos,J0pos) += J0.transpose() * J0;
    if(NJ1) JTJ.template block<NJ1,NJ1>(J1pos,J1pos) += J1.transpose() * J1;
    if(NJ2) JTJ.template block<NJ2,NJ2>(J2pos,J2pos) += J2.transpose() * J2;

    // Lower diagonal blocks
    if(NJ1 && NJ0) JTJ.template block<NJ1,NJ0>(J1pos,J0pos) += J1.transpose() * J0;
    if(NJ2 && NJ0) JTJ.template block<NJ2,NJ0>(J2pos,J0pos) += J2.transpose() * J0;
    if(NJ2 && NJ1) JTJ.template block<NJ2,NJ1>(J2pos,J1pos) += J2.transpose() * J1;

    // Upper diagonal blocks TODO: use only one diagonal in future
    if(NJ0 && NJ1) JTJ.template block<NJ0,NJ1>(J0pos,J1pos) += J0.transpose() * J1;
    if(NJ0 && NJ2) JTJ.template block<NJ0,NJ2>(J0pos,J2pos) += J0.transpose() * J2;
    if(NJ1 && NJ2) JTJ.template block<NJ1,NJ2>(J1pos,J2pos) += J1.transpose() * J2;

    // Errors
    if(NJ0) JTy.template segment<NJ0>(J0pos) += J0.transpose() * y;
    if(NJ1) JTy.template segment<NJ1>(J1pos) += J1.transpose() * y;
    if(NJ2) JTy.template segment<NJ2>(J2pos) += J2.transpose() * y;
}

//! Update JTJ and JTy with sparse outer product of NR sparse rows of J.
//! J = (.., J0, .., J1, .. )
template<typename T, unsigned NR, unsigned NJ0, unsigned NJ1>
inline void AddSparseOuterProduct(
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& JTJ,
        Eigen::Matrix<T,Eigen::Dynamic,1>& JTy,
        const Eigen::Matrix<T,NR,NJ0>& J0, int J0pos,
        const Eigen::Matrix<T,NR,NJ1>& J1, int J1pos,
        const Eigen::Matrix<T,NR,1>& y
) {
    // On diagonal blocks
    if(NJ0) JTJ.template block<NJ0,NJ0>(J0pos,J0pos) += J0.transpose() * J0;
    if(NJ1) JTJ.template block<NJ1,NJ1>(J1pos,J1pos) += J1.transpose() * J1;

    // Lower diagonal blocks
    if(NJ1 && NJ0) JTJ.template block<NJ1,NJ0>(J1pos,J0pos) += J1.transpose() * J0;

    // Upper diagonal blocks TODO: use only one diagonal in future
    if(NJ0 && NJ1) JTJ.template block<NJ0,NJ1>(J0pos,J1pos) += J0.transpose() * J1;

    // Errors
    if(NJ0) JTy.template segment<NJ0>(J0pos) += J0.transpose() * y;
    if(NJ1) JTy.template segment<NJ1>(J1pos) += J1.transpose() * y;
}
