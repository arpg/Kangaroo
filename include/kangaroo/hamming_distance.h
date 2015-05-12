#pragma once

#include <cuda_runtime.h>

namespace roo
{

//////////////////////////////////////////////////////
// Hamming Distance for unsigned int types
//////////////////////////////////////////////////////

inline __device__
unsigned HammingDistance(unsigned int p, unsigned int q)
{
    return __popc(p^q);
}

inline __device__
unsigned HammingDistance(const uint2 p, const uint2 q)
{
    return __popc(p.x^q.x) + __popc(p.y^q.y);
}

inline __device__
unsigned HammingDistance(const uint3 p, const uint3 q)
{
    return __popc(p.x^q.x) + __popc(p.y^q.y) + __popc(p.z^q.z);
}

inline __device__
unsigned HammingDistance(const uint4 p, const uint4 q)
{
    return __popc(p.x^q.x) + __popc(p.y^q.y) + __popc(p.z^q.z) + __popc(p.w^q.w);
}

//////////////////////////////////////////////////////
// Hamming Distance for unsigned long types
//////////////////////////////////////////////////////

inline __device__
unsigned HammingDistance(unsigned long p, unsigned long q)
{
    return __popc(p^q);
}

inline __device__
unsigned HammingDistance(const ulong2 p, const ulong2 q)
{
    return __popc(p.x^q.x) + __popc(p.y^q.y);
}

inline __device__
unsigned HammingDistance(const ulong3 p, const ulong3 q)
{
    return __popc(p.x^q.x) + __popc(p.y^q.y) + __popc(p.z^q.z);
}

inline __device__
unsigned HammingDistance(const ulong4 p, const ulong4 q)
{
    return __popc(p.x^q.x) + __popc(p.y^q.y) + __popc(p.z^q.z) + __popc(p.w^q.w);
}

}
