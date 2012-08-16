#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>

using namespace std;

int main( int /*argc*/, char* argv[] )
{
    cudaError_t err = cudaSetDevice(0);
    if(err != cudaSuccess) {
        cout << "Unable to set device: " << err << endl;
    }

    size_t avail;
    size_t total;

    while(1) {
        err = cudaMemGetInfo( &avail, &total );
        if(err != cudaSuccess) {
            cout << "cudaMemGetInfo failed: " << err << endl;

            err = cudaDeviceReset();
            if(err != cudaSuccess) {
                cout << "Unable to reset device: " << err << endl;
            }
        }else{
            size_t used = total - avail;
            const unsigned bytes_per_mb = 1024*1000;
            cout << "Total: " << total/bytes_per_mb << ", Available: " << avail/bytes_per_mb << ", Used: " << used/bytes_per_mb << endl;
        }
        usleep(1000*1000);
    }
}
