#ifndef _CVAR_HELPERS_
#define _CVAR_HELPERS_

#include <CVars/CVarVectorIO.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

////////////////////////////////////////////////////////////////////////////
// Overloading Eigen for CVars
namespace CVarUtils
{
    template<typename T, int R, int C>
    inline std::ostream& operator<<( std::ostream& Stream, const Eigen::Matrix<T,R,C>& Mat )
    {
        unsigned int nRows = Mat.rows();
        unsigned int nCols = Mat.cols();

        Stream << "[ ";

        for( unsigned int ii = 0; ii < nRows-1; ii++ ) {
            for( unsigned int jj = 0; jj < nCols-1; jj++ ) {
                Stream << Mat(ii, jj);
                Stream << ", ";
            }
            Stream << Mat(ii, nCols-1);
            Stream << "; ";
        }
        for( unsigned int jj = 0; jj < nCols-1; jj++ ) {
            Stream << Mat(nRows-1, jj);
            Stream << ", ";
        }
        Stream << Mat(nRows-1, nCols-1);
        Stream << " ]";

        return Stream;
    }

    ////////////////////////////////////////////////////////////////////////////
    template<typename T, int R, int C>
    inline std::istream& operator>>( std::istream& Stream, Eigen::Matrix<T,R,C>& Mat )
    {

        unsigned int nRows = Mat.rows();
        unsigned int nCols = Mat.cols();
        char str[256];

        Stream.getline(str, 255, '[');
        if( Stream.gcount() > 1 ) {
            return Stream;
        }
        for( unsigned int ii = 0; ii < nRows-1; ii++ ) {
            for( unsigned int jj = 0; jj < nCols-1; jj++ ) {
                Stream.getline(str, 255, ',');
                Mat(ii, jj) = std::strtod(str, NULL);
            }
            Stream.getline(str, 255, ';');
            Mat(ii, nCols-1) = std::strtod(str, NULL);
        }
        for( unsigned int jj = 0; jj < nCols-1; jj++ ) {
            Stream.getline(str, 255, ',');
            Mat(nRows-1, jj) = std::strtod(str, NULL);
        }
        Stream.getline(str, 255, ']');
        Mat(nRows-1, nCols-1) = std::strtod(str, NULL);
        return Stream;
    }

    ////////////////////////////////////////////////////////////////////////////
    inline std::ostream& operator<<( std::ostream& Stream, const Sophus::SO3d& R )
    {
        Stream << R.unit_quaternion().coeffs();
        return Stream;
    }

    ////////////////////////////////////////////////////////////////////////////
    inline std::ostream& operator<<( std::ostream& Stream, const Sophus::SE3d& T )
    {
        Stream << "[" << T.so3() << "," << T.translation() << "]";
        return Stream;
    }

    ////////////////////////////////////////////////////////////////////////////
    inline std::istream& operator>>( std::istream& Stream, Sophus::SO3d& R )
    {
        Eigen::Matrix<double,4,1> coeffs;
        Stream >> coeffs;
        R.setQuaternion(Eigen::Quaterniond(coeffs));
        return Stream;
    }

    ////////////////////////////////////////////////////////////////////////////
    inline std::istream& operator>>( std::istream& Stream, Sophus::SE3d& T )
    {
        char str[256];

        Stream.getline(str, 255, '[');
        if( Stream.gcount() > 1 ) {
            return Stream;
        }
        Stream >> T.so3();
        Stream.getline(str, 255, ',');
        Stream >> T.translation();
        Stream.getline(str, 255, ']');
        return Stream;
    }

}

#endif
