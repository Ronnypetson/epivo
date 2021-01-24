#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include "sophus/geometry.hpp"

using namespace std;
using namespace Eigen;

struct Scene {
    // T is the point transformation, not camera pose
    MatrixXd T; // 4x4
    MatrixXd X; // Nx3
    MatrixXd p; // Nx3
    MatrixXd p_; // Nx3
};

class RepJacobian {
    // NxD (D=6) Jacobian of one reprojection wrt ONE zeta of point transformation
    int zeta_id;
    int source_id;
    int target_id;
  public:
    RepJacobian (int z, int s, int t){
        assert(s <= t);
        assert(s <= z);
        assert(z <= t);
        zeta_id = z;
        source_id = s;
        target_id = t;
    }
    void compute (const MatrixXd &p,
                  const MatrixXd &p_,
                  const vector<MatrixXd> &T0s,
                  MatrixXd &J_r_eps);
    void compute (const MatrixXd &p,
                  const MatrixXd &p_,
                  const MatrixXd &Tl0,
                  const MatrixXd &Tr0,
                  MatrixXd &J_r_eps);
};

struct Sequence {
    // Point transformations should be multiplied on the left
    // Camera pose transformations on the right
    vector<Scene> scenes;
};
