#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

#include <math.h>

typedef std::pair<int,int> biswap;

struct BitonicSwaps
{
    BitonicSwaps(int size)
        :size(size)
    {

    }

    void AddSwap(int a, int b)
    {
        // Only compare/swap if in bounds
        if(a < size && b < size) {
            swaps.push_back(biswap(a,b));
        }
    }

    void Print()
    {
        for(int i=0; i < swaps.size(); ++i )
        {
            biswap swap = swaps[i];
            std::cout << "t2(" << swap.first << "," << swap.second << "); ";
        }
        std::cout << std::endl;
    }

    int Size()
    {
        return swaps.size();
    }

    int size;
    std::vector<biswap> swaps;
};

// Based on http://en.wikipedia.org/wiki/Bitonic_sorter
// Some talk of using it with cuda here:
// http://www.scribd.com/doc/87393205/25/Parallel-Bitonic-Sort-On-CUDA#outer_page_33
// But we're inspired more by this implementation in glsl
// http://graphics.cs.williams.edu/papers/MedianShaderX6/median5.pix
struct BitonicNetwork
{
    BitonicNetwork(int size)
        :size(size)
    {
        stages = ceil(log(size) / log(2));
        N = 1 << stages;
        std::cout << "Numbers: " << size << std::endl;
        std::cout << "Stages:" << stages << std::endl;
        std::cout << "Network size: " << N << std::endl;
    }

    void Compute()
    {
        rounds.clear();
        for(int s=0; s<stages; ++s) {
            ComputeStage(s);
        }
        std::cout << "Swap rounds: " << rounds.size() << std::endl;
    }

    void Prune(std::set<int>& workingset)
    {
        for(int r=rounds.size()-1; r>=0; r--)
        {
            Prune(r, workingset);
        }
    }

    void Print()
    {
        for(int r=0; r < rounds.size(); ++r)
        {
            rounds[r].Print();
        }
    }

    int Size()
    {
        int size = 0;
        for(int r=0; r < rounds.size(); ++r)
        {
            size += rounds[r].Size();
        }
        return size;
    }

protected:
    void Prune(int round, std::set<int>& workingset)
    {
        std::vector<biswap>& swaps = rounds[round].swaps;

        for(int sw = swaps.size()-1; sw >= 0; sw--)
        {
            const biswap& swap = swaps[sw];

            if(workingset.find(swap.first) == workingset.end() && workingset.find(swap.second) == workingset.end())
            {
                swaps.erase(swaps.begin() + sw);
            }else{
                workingset.insert(swap.first);
                workingset.insert(swap.second);
            }
        }
    }

    // http://en.wikipedia.org/wiki/File:BitonicSort1.svg
    void AddRoundSwaps(BitonicSwaps& swaps, int s, int r)
    {
        const int blocks = N>>(s+1-r);

        for(int b=0; b< blocks; b++ ) {
            const int start = b<<(s+1-r);
            const int num = 1<<(s-r);
            const int step = 1<<(s-r);
            const bool dn = (b >> r) % 2;
            for(int i=0; i< num; ++i) {
                if(!dn) {
                    swaps.AddSwap(start + i ,start + i + step);
                }else{
                    swaps.AddSwap(start + i + step, start + i);
                }
            }
        }
    }

    // http://en.wikipedia.org/wiki/File:BitonicSort.svg
    void AddRoundSwapsKeepOrder(BitonicSwaps& swaps, int s, int r)
    {
        const int blocks = N>>(s+1-r);

        for(int b=0; b< blocks; b++ ) {
            const int start = b<<(s+1-r);
            const int num = 1<<(s-r);
            const int step = 1<<(s-r);
            for(int i=0; i< num; ++i) {
                const int nextstart = (b+1)<<(s+1-r);
                const int a = start + i;
                const int b = (r==0 && s > 0) ? nextstart - (i+1) : start + i + step;
                swaps.AddSwap(a,b);
            }
        }
    }

    void ComputeStage(int s)
    {
        for(int r=0; r <= s; ++ r)
        {
            BitonicSwaps swaps(size);
            AddRoundSwapsKeepOrder(swaps,s,r);
            rounds.push_back(swaps);
        }
    }

    int size;
    int N;
    int stages;
    std::vector<BitonicSwaps> rounds;
};

//int OutputBitonicNetwork()
//{
//    const int size = 9*9;

//    // Which sorted indices do we need?
//    set<int> desired;

//    // 5x5 median - just central value
////    desired.insert(size/2);

//    // 5x5 median - upper half (so we can offset start)
//    for(int i=size/2; i<size; ++i ) {
//        desired.insert(i);
//    }

//    BitonicNetwork network(size);
//    network.Compute();
//    cout << "Complete sort network (" << network.Size() << " swaps)" << endl;
//    network.Print();

//    network.Prune(desired);
//    cout << endl << "Pruned sort network (" << network.Size() << " swaps)" << endl;
//    network.Print();
//}
