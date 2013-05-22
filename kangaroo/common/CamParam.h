#pragma once

#include <Eigen/Eigen>

template<typename Cam>
struct CamParamFixed {
    const static int PARAMS = 0;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const Cam& cam, const Eigen::Vector2d&)
    {
        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0), 0,
                0, cam.K()(1,1);
        return ret;
    }

    static inline Eigen::Matrix<double,2,PARAMS> dmap_by_dk(const Cam&, const Eigen::Vector2d&)
    {
        return Eigen::Matrix<double,2,PARAMS>();
    }

    static inline void UpdateCam(Cam&, const Eigen::Matrix<double,PARAMS,1>& )
    {
    }
};

template<typename Cam>
struct CamParamLinearFuEqFv {
    const static int PARAMS = 3;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const Cam& cam, const Eigen::Vector2d&)
    {
        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0), 0,
                0, cam.K()(1,1);
        return ret;
    }

    static inline Eigen::Matrix<double,2,PARAMS> dmap_by_dk(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double u = x(0);
        const double v = x(1);
        const double f  = cam.K()(0,0);
        const double u0 = cam.K()(0,2);
        const double v0 = cam.K()(1,2);

        Eigen::Matrix<double,2,PARAMS> d;
        d <<    f * u, u0, 0,
                f * v, 0, v0;
        return d;
    }

    static inline void UpdateCam(Cam& cam, const Eigen::Matrix<double,PARAMS,1>& x )
    {
        Eigen::Matrix3d K = cam.K();
        K(0,0) *= exp(x(0));
        K(1,1) *= exp(x(0));
        K(0,2) *= exp(x(1));
        K(1,2) *= exp(x(2));
        cam.SetK(K);
    }
};

template<typename Cam>
struct CamParamLinear {
    const static int PARAMS = 4;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const Cam& cam, const Eigen::Vector2d&)
    {
        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0), 0,
                0, cam.K()(1,1);
        return ret;
    }

    static inline Eigen::Matrix<double,2,PARAMS> dmap_by_dk(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double u = x(0);
        const double v = x(1);
        const double fu = cam.K()(0,0);
        const double fv = cam.K()(1,1);
        const double u0 = cam.K()(0,2);
        const double v0 = cam.K()(1,2);

        Eigen::Matrix<double,2,PARAMS> d;
        d <<    fu * u, 0, u0, 0,
                0, fv * v, 0, v0;
        return d;
    }

    static inline void UpdateCam(Cam& cam, const Eigen::Matrix<double,PARAMS,1>& x )
    {
        Eigen::Matrix3d K = cam.K();
        K(0,0) *= exp(x(0));
        K(1,1) *= exp(x(1));
        K(0,2) *= exp(x(2));
        K(1,2) *= exp(x(3));
        cam.SetK(K);
    }
};

template<typename Cam>
struct CamParamMatlab {
    const static int PARAMS = 4 + 2;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double k1 = cam._k1;
        const double k2 = cam._k2;
        const double u = x(0);
        const double uu = u*u;
        const double uuu = uu*u;
        const double v = x(1);
        const double vv = v*v;
        const double vvv = vv*v;

        const double dpoly_by_u = 2*k1*u + k2*(4*uuu + 4*u*vv); // + k3*(6*u*vvv*v + 12*uuu*vv + 6*uuu*uu);
        const double dpoly_by_v = 2*k1*v + k2*(4*vvv + 4*v*uu); // + k3*(6*v*uuu*u + 12*vvv*uu + 6*vvv*vv);

        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0) * (dpoly_by_u*u + 1), cam.K()(0,0) * dpoly_by_v * u,
                cam.K()(1,1) * dpoly_by_u*v, cam.K()(1,1) * (dpoly_by_v*v + 1);
        return ret;
    }

    static inline Eigen::Matrix<double,2,PARAMS> dmap_by_dk(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double u = x(0);
        const double v = x(1);

        const double k1 = cam._k1;
        const double k2 = cam._k2;

        const double rd2 = u*u + v*v;
        const double rd4 = rd2*rd2;
//        const double rd6 = rd4*rd6;

        const double fu = cam.K()(0,0);
        const double fv = cam.K()(1,1);
        const double u0 = cam.K()(0,2);
        const double v0 = cam.K()(1,2);

        const double poly = 1 + k1*rd2 + k2*rd4; // + k3*rd6;

        Eigen::Matrix<double,2,PARAMS> d;
        d <<    fu * poly * u, 0, u0, 0, fu * k1*rd2 * u, fu * k2*rd4 * u, // fu * k3*rd6 * u,
                0, fv * poly * v, 0, v0, fv * k1*rd2 * v, fv * k2*rd4 * v; // fv * k3*rd6 * v;
        return d;
    }

    static inline void UpdateCam(Cam& cam, const Eigen::Matrix<double,PARAMS,1>& x )
    {
        Eigen::Matrix3d K = cam.K();
        K(0,0) *= exp(x(0));
        K(1,1) *= exp(x(1));
        K(0,2) *= exp(x(2));
        K(1,2) *= exp(x(3));
        cam.SetK(K);

        cam._k1 *= exp(x(4));
        cam._k2 *= exp(x(5));
//        cam._k3 *= exp(x(6));
    }
};
