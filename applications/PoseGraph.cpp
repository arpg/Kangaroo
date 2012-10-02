#include <pangolin/pangolin.h>
#include <ceres/ceres.h>

using namespace std;
using namespace pangolin;
using namespace Eigen;
using namespace ceres;

class SimpleCostFunction
  : public SizedCostFunction<1 /* number of residuals */,
                             1 /* size of first parameter */> {
 public:
  virtual ~SimpleCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double x = parameters[0][0];

    // f(x) = 10 - x.
    residuals[0] = 10 - x;

    // f'(x) = -1. Since there's only 1 parameter and that parameter
    // has 1 dimension, there is only 1 element to fill in the
    // jacobians.
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};

int main( int /*argc*/, char* argv[] )
{
    const int w = 640;
    const int h = 480;

    pangolin::CreateGlutWindowAndBind(__FILE__,w,h);

    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam;
    s_cam.SetProjectionMatrix(ProjectionMatrix(640,480,420,420,320,240,0.1,1000));
    s_cam.SetModelViewMatrix(ModelViewLookAt(0,5,5,0,0,0,0,0,1));

    View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(s_cam,AxisZ));

    ceres::Problem problem;
    double x = 5.0;
    problem.AddResidualBlock(new SimpleCostFunction, NULL, &x);

    // Run the solver!
    Solver::Options options;
    options.max_num_iterations = 10;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : 5.0 -> " << x << "\n";

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        d_cam.ActivateScissorAndClear(s_cam);

        glColor3f(0.8,0.8,0.8);
        glDraw_z0(1.0,5);
        glDisable(GL_DEPTH_TEST);
        glDrawAxis(2);
        glEnable(GL_DEPTH_TEST);

        pangolin::FinishGlutFrame();
    }
}
