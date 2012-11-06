#include <iostream>

#include "common/PoseGraph.h"


using namespace std;


int main( int /*argc*/, char* argv[] )
{
    PoseGraph posegraph;
    const int coord_vicon = posegraph.AddSecondaryCoordinateFrame();
    const int kf_sdf = posegraph.AddKeyframe();

    const Sophus::SE3 T_vk(Sophus::SO3(1,2,3),Eigen::Vector3d(4,5,6));
    const Sophus::SE3 T_sr(Sophus::SO3(3,2,1),Eigen::Vector3d(6,5,4));

    for( int i=0; i<1000; ++i) {
        Sophus::SE3 T_rv(Sophus::SO3(i*0.01,i*-0.02, i*0.03), Eigen::Vector3d(i*0.01,2*i*0.02,0) );
        Sophus::SE3 T_sk = T_sr * T_rv * T_vk;

        const int kfid = posegraph.AddKeyframe(new Keyframe(T_sk));
//        posegraph.SetKeyframeFreedom(kfid,false,false);
        posegraph.AddIndirectUnaryEdge(kfid,coord_vicon,T_rv);
        posegraph.AddBinaryEdge(kf_sdf,kfid,T_sk);
    }

    posegraph.Solve();

    cout << T_sr.inverse().matrix3x4() << endl << endl;
    cout << T_vk.inverse().matrix3x4() << endl << endl;

    cout << posegraph.GetKeyframe(kf_sdf).GetT_wk().matrix3x4() << endl << endl;
    cout << posegraph.GetSecondaryCoordinateFrame(coord_vicon).GetT_wk().matrix3x4() << endl << endl;
}
