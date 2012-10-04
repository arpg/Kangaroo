#pragma once

#include <Eigen/Eigen>
#include <Mvlpp/Mvl.h>
#include <vector>

inline void LoadPosesFromFile(
    std::vector<Sophus::SE3>& vecT_wh, const std::string& filename, int startframe,
    const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; lines < 10000;lines++)
        {
            for(int i=0; i<6; ++i ) {
                ifs >> row(i);
            }
            if(lines >= startframe) {
                Sophus::SE3 T_wr( T_hf * mvl::Cart2T(row) * T_fh );
                vecT_wh.push_back(T_wr);
            }
        }

        ifs.close();
    }
}
