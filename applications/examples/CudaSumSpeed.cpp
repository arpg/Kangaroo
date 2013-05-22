#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>

#include <kangaroo/kangaroo.h>

using namespace std;

int main( int /*argc*/, char* argv[] )
{
    cudaError_t err = cudaSetDevice(0);
    if(err != cudaSuccess) {
        cout << "Unable to set device: " << err << endl;
        exit(-1);
    }

    const int l = 0;
    const int w = 640 >> l;
    const int h = 480 >> l;
    const int num_trials = 100;

    Gpu::Image<unsigned char,Gpu::TargetDevice, Gpu::Manage> dWorkspace(w*sizeof(Gpu::LeastSquaresSystem<float,6>),h);

    Gpu::CudaTimer timer;

    timer.Start();
    cout << "Started" << endl;
    for(int trials = 0; trials < num_trials; trials++)
    {
        SumSpeedTest(dWorkspace, w,h,16,16);
    }
    cout << "Finished" << endl;
    timer.Stop();

    cout << timer.Elapsed_ms() / num_trials << "ms" << endl;
}
