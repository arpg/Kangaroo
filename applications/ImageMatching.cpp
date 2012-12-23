#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <npp.h>

#include <fiducials/drawing.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/BaseDisplay.h"
#include "common/ImageSelect.h"
#include "common/CameraModelPyramid.h"

#include <kangaroo/kangaroo.h>
#include <SceneGraph/SceneGraph.h>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;
using namespace mvl;
using namespace SceneGraph;

float distance(const cv::Mat& im1, const cv::Mat& im2)
{
    const int N = im1.rows * im1.cols;
    float m1 = 0;
    float m2 = 0;

    for(int r=0; r < im1.rows; ++r) {
        for(int c=0; c< im1.cols; ++c) {
            m1 += im1.at<unsigned char>(r,c);
            m2 += im2.at<unsigned char>(r,c);
        }
    }
    m1 /= N;
    m2 /= N;

    float sum = 0;
    for(int r=0; r < im1.rows; ++r) {
        for(int c=0; c< im1.cols; ++c) {
            const float diff = (im1.at<unsigned char>(r,c)-m1) - (im2.at<unsigned char>(r,c)-m2);
            sum += abs(diff);
        }
    }
    return sum / (255.0 * N);
}

int main( int argc, char* argv[] )
{
    CameraDevice video = OpenRpgCamera(argc,argv);

    std::vector<rpg::ImageWrapper> images;

    const int max_level = 6;
    std::vector<cv::Mat> pyramid;
    std::vector<cv::Mat> thumbnails;

    while(video.Capture(images)) {
        cv::buildPyramid(images[0].Image, pyramid, max_level);
        thumbnails.push_back( pyramid[max_level].clone() );
    }

    const size_t N = thumbnails.size();
    cout << "Num Images " << N << endl;

    float dist_min = std::numeric_limits<float>::max();
    float dist_max = std::numeric_limits<float>::min();

    cv::Mat M(N,N,CV_32FC1);
    for(size_t u=0; u < N; ++u) {
        for(size_t v=u; v < N; ++v) {
            const float mdist = distance(thumbnails[u], thumbnails[v]);
            dist_min = std::min(dist_min,mdist);
            dist_max = std::max(dist_max,mdist);
//            M(u,v) = mdist;
//            M(v,u) = mdist;
            M.at<float>(u,v) = mdist;
            M.at<float>(v,u) = mdist;
        }
    }

//    const unsigned int w = images[0].width();
//    const unsigned int h = images[0].height();

//    View& container = SetupPangoGL(640,480);
//    SetupContainer(container, 1, (float)w/h);

//    Var<bool> step("ui.step", false, false);
//    Var<bool> run("ui.run", false, true);

//    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
//    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    cv::namedWindow("cvwin");
    cv::imshow("cvwin", M);
    cv::waitKey(0);

//    for(unsigned long frame=0; frame < thumbnails.size(); )
//    {
//        const bool go = true; //(frame==0) || run || Pushed(step);

//        if(go) {
//            cv::imshow("cvwin", thumbnails[frame++]);
//            cv::waitKey(1);
//        }

////        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
////        pangolin::FinishGlutFrame();
//    }
}
