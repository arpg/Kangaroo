#include <iostream>

#include "common/BitonicSortingNetwork.h"

using namespace std;

int OutputBitonicNetwork()
{
    const int size = 9*9;

    // Which sorted indices do we need?
    set<int> desired;

    // 5x5 median - just central value
//    desired.insert(size/2);

    // 5x5 median - upper half (so we can offset start)
    for(int i=size/2; i<size; ++i ) {
        desired.insert(i);
    }

    BitonicNetwork network(size);
    network.Compute();
    cout << "Complete sort network (" << network.Size() << " swaps)" << endl;
    network.Print();

    network.Prune(desired);
    cout << endl << "Pruned sort network (" << network.Size() << " swaps)" << endl;
    network.Print();
}

#include <fiducials/tracker.h>

void TestEPS()
{
    Tracker tracker(640,480);
    tracker.target.LoadEPS("/Users/slovegrove/Desktop/stereo.eps");

}

int main( int /*argc*/, char* argv[] )
{
//    OutputBitonicNetwork();
    TestEPS();
}
