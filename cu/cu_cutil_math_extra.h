//////////////////////////////////////////////////////
// Additions to cutil_math.h
//////////////////////////////////////////////////////

inline __host__ __device__ float3 operator*(float b, uchar3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float3 operator*(uchar3 a, float b)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float1 operator*(float b, uchar1 a)
{
    return make_float1(b * a.x);
}

inline __host__ __device__ float1 operator*(uchar1 a, float b)
{
    return make_float1(b * a.x);
}
