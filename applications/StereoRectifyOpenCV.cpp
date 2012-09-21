#include "common/RpgCameraOpen.h"
#include "common/SaveMvlCamModel.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const std::string dDir = "./rectified/";

    CameraDevice video;
    OpenRpgCamera(video,argc,argv,2);

    // Capture first image
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);
    const int w = images[0].width();
    const int h = images[0].height();
    Size imageSize(w,h);

    // Read parameters
    FileStorage fs_ext("extrinsics.yml", FileStorage::READ);
    FileStorage fs_int("intrinsics.yml", FileStorage::READ);

    Mat cameraMatrix[2], distCoeffs[2];
    Mat R, T;

    fs_int["M1"] >> cameraMatrix[0];
    fs_int["M2"] >> cameraMatrix[1];
    fs_int["D1"] >> distCoeffs[0];
    fs_int["D2"] >> distCoeffs[1];

    fs_ext["R"] >> R;
    fs_ext["T"] >> T;

    const double alpha = 0; //1;
    const int correct_disp = 0; //CALIB_ZERO_DISPARITY;

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  correct_disp, alpha, imageSize, &validRoi[0], &validRoi[1]);

    Mat rmap[2][2];
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    cout << "P1: " << P1 << endl;
    cout << "P2: " << P2 << endl;

    Eigen::Matrix4d T_lr = Eigen::Matrix4d::Identity();
    T_lr(0,3) = -P2.at<double>(0,3) / P2.at<double>(0,0);
    SaveCamModelLeftRightVisionConvention(
                dDir, w,h,
                P1.at<double>(0,0), P1.at<double>(1,1), P1.at<double>(0,2), P1.at<double>(1,2), 0, 0, 0, 0, 0, 0,
                P2.at<double>(0,0), P2.at<double>(1,1), P2.at<double>(0,2), P2.at<double>(1,2), 0, 0, 0, 0, 0, 0,
                T_lr
    );

    Mat rimg[2];

    for(int frame=0; ; ++frame)
    {
        if(frame > 0 ) {
            if( !video.Capture(images) )
                break;
        }

        // rectify
        for(int k=0; k < 2; ++k) {
            remap(images[k].Image, rimg[k], rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
        }

        char Index[10];
        sprintf( Index, "%05d", frame );

        imwrite(dDir + "left_" + Index + "_rect.pgm", rimg[0]);
        imwrite(dDir + "right_" + Index + "_rect.pgm", rimg[1]);

        cout << Index << '\r';
        cout.flush();
    }

}
