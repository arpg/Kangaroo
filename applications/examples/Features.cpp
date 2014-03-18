#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>
#include <kangaroo/extra/DisplayUtils.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/ImageSelect.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    // Open video device
    const std::string vid_uri = argc >= 2 ? argv[1] : "";    
    pangolin::VideoInput video(vid_uri);
    if(video.PixFormat().format != "GRAY8")
        throw pangolin::VideoException("Wrong format. Gray8 required.");

    // Image dimensions and host copy
    const unsigned int w = video.Width();
    const unsigned int h = video.Height();
    roo::Image<unsigned char, roo::TargetHost, roo::Manage> host(w,h);

    // Initialise window
    View& container = SetupPangoGLWithCuda(180+2*w, h,180);

    // Allocate Camera Images on device for processing
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> img(w,h);
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> imgf(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgs(w,h);

    ActivateDrawImage<unsigned char> adg(img, GL_LUMINANCE8, true, true);
    ActivateDrawImage<unsigned char> ads(imgf, GL_LUMINANCE8, true, true);

    Handler2dImageSelect handler2d(w,h);
    SetupContainer(container, 2, (float)w/h);
    container[0].SetDrawFunction(std::ref(adg)).SetHandler(&handler2d);
    container[1].SetDrawFunction(std::ref(ads)).SetHandler(&handler2d);

    Var<bool> run("ui.run", true, true);
    Var<bool> step("ui.step", false, false);
    Var<float> harris_thresh("ui.harris threshold", 1E4, 1, 1E8, true);
    Var<float> harris_lambda("ui.harris lambda", 0.04, 0, 1);
    Var<int> harris_nms_rad("ui.harris nonmax rad", 1, 0, 5);
    Var<float> fast_thresh("ui.fast threshold", 10, 0, 255);
    Var<int> minseglen("ui.min seg len", 9, 9, 16);

    Var<bool> do_fast("ui.do fast", false, true);

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        bool go = (frame==0) || run || Pushed(step);

        if(go) {
            if(video.GrabNext(host.ptr)) {
                img.CopyFrom(host);
            }
        }

        go |= GuiVarHasChanged();
        if(go) {
            if(do_fast) {
                roo::SegmentTest(imgf, img, fast_thresh, minseglen);
            }else{
                roo::HarrisScore(imgs, img, harris_lambda);
                roo::NonMaximalSuppression(imgf, imgs, harris_nms_rad, harris_thresh);
            }
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishFrame();
    }
}
