#pragma once

#include <ceres/ceres.h>
//#include <ceres/rotation.h>

////////////////////////////////////////////////////////////
// Unit Quaternion utilities for scalar arrays
// Differs from ceres implementation because array order is different
// QuatXYZW
////////////////////////////////////////////////////////////

template <typename T>
void QuatXYZWConjugate(const T in[4], T out[4] )
{
    out[0] = -in[0];
    out[1] = -in[1];
    out[2] = -in[2];
    out[3] = in[3];
}

template <typename T>
void QuatXYZWNormalize(T q[4])
{
    const T scale = T(1) / ceres::sqrt(
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    );

    q[0] *= scale;
    q[1] *= scale;
    q[2] *= scale;
    q[3] *= scale;
}

template <typename T>
void QuatXYZWSetIdentity(T q[4])
{
    q[0] = 0;
    q[1] = 0;
    q[2] = 0;
    q[3] = 1;
}

template <typename T> inline
void UnitQuatXYZWRotatePoint(const T q[4], const T pt[3], T result[3]) {
  const T t2 =  q[1] * q[2];
  const T t3 =  q[3] * q[1];
  const T t4 =  q[3] * q[2];
  const T t5 = -q[0] * q[0];
  const T t6 =  q[0] * q[1];
  const T t7 =  q[0] * q[2];
  const T t8 = -q[1] * q[1];
  const T t9 =  q[1] * q[2];
  const T t1 = -q[2] * q[2];
  result[0] = T(2) * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0];  // NOLINT
  result[1] = T(2) * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1];  // NOLINT
  result[2] = T(2) * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2];  // NOLINT
}

template <typename T> inline
void UnitQuatXYZWRotatePointNeg(const T q[4], const T pt[3], T result[3]) {
  const T t2 =  q[1] * q[2];
  const T t3 =  q[3] * q[1];
  const T t4 =  q[3] * q[2];
  const T t5 = -q[0] * q[0];
  const T t6 =  q[0] * q[1];
  const T t7 =  q[0] * q[2];
  const T t8 = -q[1] * q[1];
  const T t9 =  q[1] * q[2];
  const T t1 = -q[2] * q[2];
  result[0] = T(-1) * (T(2) * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0]);  // NOLINT
  result[1] = T(-1) * (T(2) * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1]);  // NOLINT
  result[2] = T(-1) * (T(2) * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2]);  // NOLINT
}

template <typename T> inline
void UnitQuatXYZWInverseRotatePoint(const T q[4], const T pt[3], T result[3]) {
  const T t2 =  q[1] * q[2];
  const T t3 = -q[3] * q[1];
  const T t4 = -q[3] * q[2];
  const T t5 = -q[0] * q[0];
  const T t6 =  q[0] * q[1];
  const T t7 =  q[0] * q[2];
  const T t8 = -q[1] * q[1];
  const T t9 =  q[1] * q[2];
  const T t1 = -q[2] * q[2];
  result[0] = T(2) * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0];  // NOLINT
  result[1] = T(2) * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1];  // NOLINT
  result[2] = T(2) * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2];  // NOLINT
}

template<typename T> inline
void QuatXYZWProduct(const T z[4], const T w[4], T zw[4]) {
  zw[0] = z[3] * w[0] + z[0] * w[3] + z[1] * w[2] - z[2] * w[1];
  zw[1] = z[3] * w[1] - z[0] * w[2] + z[1] * w[3] + z[2] * w[0];
  zw[2] = z[3] * w[2] + z[0] * w[1] - z[1] * w[0] + z[2] * w[3];
  zw[3] = z[3] * w[3] - z[0] * w[0] - z[1] * w[1] - z[2] * w[2];
}

template<typename T> inline
void QuatXYZWInverseProduct(const T ab[4], const T bc[4], T ac[4]) {
  ac[0] = ab[3] * bc[0] - ab[0] * bc[3] - ab[1] * bc[2] + ab[2] * bc[1];
  ac[1] = ab[3] * bc[1] + ab[0] * bc[2] - ab[1] * bc[3] - ab[2] * bc[0];
  ac[2] = ab[3] * bc[2] - ab[0] * bc[1] + ab[1] * bc[0] - ab[2] * bc[3];
  ac[3] = ab[3] * bc[3] + ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2];
}

template <typename T> inline
void XYZUnitQuatXYZWCompose(
        const T R_cb[4], const T t_cb[3],
        const T R_ba[4], const T t_ba[3],
        T R_ca[7], T t_ca[3]
) {
    // T_ca.t = T_cb.t + T_cb.R * T_ba.t
    UnitQuatXYZWRotatePoint(R_cb, t_ba, t_ca);
    t_ca[0] += t_cb[0];
    t_ca[1] += t_cb[1];
    t_ca[2] += t_cb[2];

    // T_ca.R = T_cb.R * T+ba.R
    QuatXYZWProduct(R_cb, R_ba, R_ca);
    QuatXYZWNormalize(R_ca);
}

template <typename T> inline
void XYZUnitQuatXYZWInverse(
    const T R_ab[4], const T t_ab[3],
    T R_ba[4], T t_ba[3]
) {
    QuatXYZWConjugate(R_ab, R_ba);
    UnitQuatXYZWRotatePointNeg(R_ba, t_ab, t_ba);
}

template <typename T> inline
void XYZUnitQuatXYZWInverseCompose(
    const T R_bc[4], const T t_bc[3],
    const T R_ba[4], const T t_ba[3],
    T R_ca[7], T t_ca[3]
) {
    T R_cb[4];
    T t_cb[3];
    XYZUnitQuatXYZWInverse(R_bc,t_bc, R_cb, t_cb);
    XYZUnitQuatXYZWCompose(R_cb, t_cb, R_ba, t_ba, R_ca, t_ca);
}

template <typename T> inline
void XYZQuatXYZWSetIdentity(
    T R_ab[4], T t_ab[3]
)
{
    t_ab[0] = 0;
    t_ab[1] = 0;
    t_ab[2] = 0;
    QuatXYZWSetIdentity(R_ab);
}

template<typename T>
inline void QuatXYZWToAngleAxis(const T* quaternion, T* angle_axis) {
  const T& q1 = quaternion[0];
  const T& q2 = quaternion[1];
  const T& q3 = quaternion[2];
  const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

  // For quaternions representing non-zero rotation, the conversion
  // is numerically stable.
  if (sin_squared_theta > T(0.0)) {
    const T sin_theta = sqrt(sin_squared_theta);
    const T& cos_theta = quaternion[3];

    // If cos_theta is negative, theta is greater than pi/2, which
    // means that angle for the angle_axis vector which is 2 * theta
    // would be greater than pi.
    //
    // While this will result in the correct rotation, it does not
    // result in a normalized angle-axis vector.
    //
    // In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
    // which is equivalent saying
    //
    //   theta - pi = atan(sin(theta - pi), cos(theta - pi))
    //              = atan(-sin(theta), -cos(theta))
    //
    const T two_theta =
        T(2.0) * ((cos_theta < 0.0)
                  ? atan2(-sin_theta, -cos_theta)
                  : atan2(sin_theta, cos_theta));
    const T k = two_theta / sin_theta;
    angle_axis[0] = q1 * k;
    angle_axis[1] = q2 * k;
    angle_axis[2] = q3 * k;
  } else {
    // For zero rotation, sqrt() will produce NaN in the derivative since
    // the argument is zero.  By approximating with a Taylor series,
    // and truncating at one term, the value and first derivatives will be
    // computed correctly when Jets are used.
    const T k(2.0);
    angle_axis[0] = q1 * k;
    angle_axis[1] = q2 * k;
    angle_axis[2] = q3 * k;
  }
}

// Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x
// with * being the quaternion multiplication operator. Here we assume
// that the first element of the quaternion vector is the real (cos
// theta) part.
class QuatXYZWParameterization : public ceres::LocalParameterization {
 public:
  virtual ~QuatXYZWParameterization() {}
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const;
  virtual bool ComputeJacobian(const double* x,
                               double* jacobian) const;
  virtual int GlobalSize() const { return 4; }
  virtual int LocalSize() const { return 3; }
};

bool QuatXYZWParameterization::Plus(const double* x,
                                      const double* delta,
                                      double* x_plus_delta) const {
  const double norm_delta =
      sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  if (norm_delta > 0.0) {
    const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
    double q_delta[4];
    q_delta[0] = sin_delta_by_delta * delta[0];
    q_delta[1] = sin_delta_by_delta * delta[1];
    q_delta[2] = sin_delta_by_delta * delta[2];
    q_delta[3] = cos(norm_delta);
    QuatXYZWProduct(q_delta, x, x_plus_delta);
    QuatXYZWNormalize(x_plus_delta);
  } else {
    for (int i = 0; i < 4; ++i) {
      x_plus_delta[i] = x[i];
    }
  }
  return true;
}

bool QuatXYZWParameterization::ComputeJacobian(const double* x,
                                                 double* jacobian) const {
    jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT
    jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
    jacobian[6] =  x[1]; jacobian[7]  = -x[0]; jacobian[8]  =  x[3];  // NOLINT
    jacobian[9] = -x[0]; jacobian[10] = -x[1]; jacobian[11] = -x[2];  // NOLINT
    return true;
}
