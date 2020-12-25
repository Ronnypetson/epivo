#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

int Dr_Deps(const MatrixXd &R0,
            const VectorXd &t0,
            const VectorXd &ep0,
            const MatrixXd &p,
            const MatrixXd &p_,
            MatrixXd &J_r_eps){
    // Computes jacobian J_r_eps (N,9) of residuals w.r.t eps in the algebra of SE(3) at eps = 0
    // p (N,3) and p_ (N,3) are point matches
    // R0 (3,3), t0 (3), and ep0 (3) are the current values
    // eps order: tr (3), rot (3), ep (3)
    assert(t0.size() == 3);
    assert(ep0.size() == 3);
    assert(R0.rows() == 3);
    assert(R0.cols() == 3);
    assert(p.cols() == 3);
    assert(p_.cols() == 3);
    int n = p.rows();
    int eps_dim = 9;
    assert(J_r_eps.cols() == eps_dim);
    assert(n > 0);
    assert(p_.rows() == n);
    assert(J_r_eps.rows() == n);

    MatrixXd J_expw_eps(9, 9); // 0_9x3 J_expw_w 0_9x3
    MatrixXd _J_expw_eps(9, 3);
    _J_expw_eps << 0, 0, 0,
                   0, 0, 1,
                   0,-1, 0,
                   0, 0,-1,
                   0, 0, 0,
                   1, 0, 0,
                   0, 1, 0,
                  -1, 0, 0,
                   0, 0, 0;
    
    J_expw_eps.col(3) = _J_expw_eps.col(0);
    J_expw_eps.col(4) = _J_expw_eps.col(1);
    J_expw_eps.col(5) = _J_expw_eps.col(2);

    vector<MatrixXd> _J_expw_eps_33;
    for(int i = 0; i < 3; i++){
        Map<MatrixXd> M(_J_expw_eps.col(i).data(), 3, 3);
        _J_expw_eps_33.push_back(M);
    }

    for(int i = 0; i < n; i++){
        MatrixXd _p(2, 3);
        _p(0, 0) = 1.0;
        _p(1, 1) = 1.0;
        _p(0, 2) = -p_(i, 0);
        _p(1, 2) = -p_(i, 1);
        MatrixXd A(2, 1), B(2, 1);
        A = _p * R0 * ep0;
        B = _p * R0 * p.row(i).transpose();
        
        // t0 .. t2 r0 .. r2 e0 .. e2
        vector<MatrixXd> J_C_eps;
        for(int j = 0; j < eps_dim; j++){
            J_C_eps.push_back(MatrixXd(2, 3));
        }
        for(int j = 0; j < 3; j++){
            J_C_eps[j] = _p * R0 * _J_expw_eps_33[j];
            cout << J_C_eps[j] << endl << endl;
        }
    }
}

int main(){
    MatrixXd R0(3, 3);
    MatrixXd p(10, 3);
    MatrixXd p_(10, 3);
    MatrixXd J(10, 9);
    VectorXd t0(3);
    VectorXd ep0(3);

    R0 << 1, 0, 0,
          0, 1, 0,
          0, 0, 1;
    ep0 << 0, 0, 1;
    t0 << 0, 0, 10;
    p_.row(0) << 5, 5, 1;
    Dr_Deps(R0, t0, ep0, p, p_, J);
}
