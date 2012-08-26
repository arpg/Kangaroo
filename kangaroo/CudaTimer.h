#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace Gpu
{

////////////////////////////////////////
// Definition
////////////////////////////////////////

// Adapted from https://bitbucket.org/ashwin/cudatimer
class CudaTimer
{
private:
    cudaEvent_t _begEvent;
    cudaEvent_t _endEvent;

public:
    CudaTimer();
    ~CudaTimer();
    void Start();
    void Stop();

    float Elapsed_ms();
};

////////////////////////////////////////
// Implementation
////////////////////////////////////////

#define SafeTimerCall( err ) __safeTimerCall( err, __FILE__, __LINE__ )

inline void __safeTimerCall( cudaError err, const char *file, const int line )
{
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do
    {
        if ( cudaSuccess != err )
        {
                        fprintf( stderr, "CudaTimer failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );

#pragma warning( pop )
    return;
}

inline CudaTimer::CudaTimer()
{
    // Create
    SafeTimerCall( cudaEventCreate( &_begEvent ) );
    SafeTimerCall( cudaEventCreate( &_endEvent ) );

    return;
}

inline CudaTimer::~CudaTimer()
{
    // Destroy
    SafeTimerCall( cudaEventDestroy( _begEvent ) );
    SafeTimerCall( cudaEventDestroy( _endEvent ) );

    return;
}

inline void CudaTimer::Start()
{
    // Record start time
    SafeTimerCall( cudaEventRecord( _begEvent, 0 ) );

    return;
}

inline void CudaTimer::Stop()
{
    // Record end time
    SafeTimerCall( cudaEventRecord( _endEvent, 0 ) );

    return;
}

inline float CudaTimer::Elapsed_ms()
{
    // Wait until end event is finished
    SafeTimerCall( cudaEventSynchronize( _endEvent ) );

    // Time difference
    float timeVal;
    SafeTimerCall( cudaEventElapsedTime( &timeVal, _begEvent, _endEvent ) );

    return timeVal;
}

}
