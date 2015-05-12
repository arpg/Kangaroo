#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

namespace roo
{

////////////////////////////////////////
// Definition
////////////////////////////////////////

// Adapted from https://bitbucket.org/ashwin/cudatimer
class CudaTimer
{
public:
    CudaTimer();
    ~CudaTimer();
    void Start();
    void Stop();

    void SyncTime();

    float Elapsed_ms();
    float Average_ms();
    float Min_ms();
    float Max_ms();

    void PrintSummary();

    void Reset();


protected:
    cudaEvent_t _begEvent;
    cudaEvent_t _endEvent;

    bool newTime;
    long trials;
    float timeLast;
    float timeAvg;
    float timeMin;
    float timeMax;
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
    : newTime(false), trials(0), timeLast(0), timeAvg(0), timeMin(1E10), timeMax(-1E10)
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
    newTime = true;
    return;
}

inline void CudaTimer::SyncTime()
{
    if(newTime) {
        SafeTimerCall( cudaEventSynchronize( _endEvent ) );
        SafeTimerCall( cudaEventElapsedTime( &timeLast, _begEvent, _endEvent ) );
        trials++;
        timeAvg = ((trials-1)*timeAvg + timeLast) / trials;
        timeMin = std::min(timeMin, timeLast);
        timeMax = std::max(timeMax, timeLast);
        newTime = false;
    }
}

inline float CudaTimer::Elapsed_ms()
{
    SyncTime();
    return timeLast;
}

inline float CudaTimer::Average_ms()
{
    SyncTime();
    return timeAvg;
}

inline float CudaTimer::Min_ms()
{
    SyncTime();
    return timeMin;
}

inline float CudaTimer::Max_ms()
{
    SyncTime();
    return timeMax;
}

inline void CudaTimer::PrintSummary()
{
    SyncTime();
    std::cout << std::fixed << std::setprecision(4) << timeLast << "ms (" << timeAvg << " avg, " << timeMin << " min, " << timeMax << " max)" << std::endl;
}

inline void CudaTimer::Reset()
{
    trials = 0;
    timeAvg = 0;
    timeMin = 1E10;
    timeMax = -1E10;
}

}
