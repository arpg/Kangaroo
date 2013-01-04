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
//            const float diff = im1.at<unsigned char>(r,c) - im2.at<unsigned char>(r,c);
            const float diff = (im1.at<unsigned char>(r,c)-m1) - (im2.at<unsigned char>(r,c)-m2);
            sum += abs(diff);
//            sum += diff*diff;
        }
    }
    return sum / (255.0 * N);
//    return sum / (255.0 * 5.0 * N);
}

int main( int argc, char* argv[] )
{
    CameraDevice video = OpenRpgCamera(argc,argv);

    std::vector<rpg::ImageWrapper> images;

    const int max_images = 4000;
    const int max_level = 5;
    std::vector<cv::Mat> pyramid;
    std::vector<cv::Mat> thumbnails;

    while(video.Capture(images) && thumbnails.size() < max_images ) {
        cv::buildPyramid(images[0].Image, pyramid, max_level);
        thumbnails.push_back( pyramid[max_level].clone() );
    }

    const size_t N = thumbnails.size();
    cout << "Num Images " << N << endl;
    cout << thumbnails[0].cols << "x" << thumbnails[0].rows << endl;

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

    View& container = SetupPangoGL(640,480,0);
    SceneGraph::ImageView costMatrix(true,false);
    container.AddDisplay(costMatrix);
    costMatrix.SetImage(M.data, M.cols, M.rows, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);
    costMatrix.SetBounds(0.0, 1.0, 0.0, 0.6, 1.0);
    Handler2dImageSelect imgSelect(N,N);
    costMatrix.SetHandler(&imgSelect);

    SceneGraph::ImageView im1(true,false);
    container.AddDisplay(im1);
    im1.SetBounds(0.0, 0.5, 0.6, 1.0, 1.0);
    SceneGraph::ImageView im2(true,false);
    container.AddDisplay(im2);
    im2.SetBounds(0.5, 1.0, 0.6, 1.0, 1.0);

    while(!pangolin::ShouldQuit())
    {
        if(imgSelect.IsSelected()) {
            imgSelect.Deselect();
            const Eigen::Vector2d p = imgSelect.GetSelectedPoint(true);
            im1.SetImage(thumbnails[p(0)].data, thumbnails[p(0)].cols, thumbnails[p(0)].rows,  GL_LUMINANCE8, GL_LUMINANCE, GL_UNSIGNED_BYTE);
            im2.SetImage(thumbnails[p(1)].data, thumbnails[p(1)].cols, thumbnails[p(1)].rows,  GL_LUMINANCE8, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
    }
}
