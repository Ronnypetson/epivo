#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

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
    const int n = p.rows();
    const int eps_dim = 9;
    assert(J_r_eps.cols() == eps_dim);
    assert(n > 0);
    assert(p_.rows() == n);
    assert(J_r_eps.rows() == n);

    MatrixXd J_expw_eps = MatrixXd::Zero(9, 9); // 0_9x3 J_expw_w 0_9x3
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

    MatrixXd J_expe_eps = MatrixXd::Zero(12, 9);
    J_expe_eps.block<9, 3>(0, 3) = _J_expw_eps;
    J_expe_eps.block<3, 3>(9, 0) = MatrixXd::Identity(3, 3);

    vector<MatrixXd> _J_expw_eps_33;
    for(int i = 0; i < 3; i++){
        // Only the middle 3 columns
        Map<MatrixXd> M(_J_expw_eps.col(i).data(), 3, 3);
        _J_expw_eps_33.push_back(M);
    }

    vector<MatrixXd> J_expe_eps_34;
    for(int i = 0; i < eps_dim; i++){
        Map<MatrixXd> M(J_expe_eps.col(i).data(), 3, 4);
        J_expe_eps_34.push_back(M);
    }

    MatrixXd H_A = MatrixXd::Zero(4, 3);
    MatrixXd H_b = MatrixXd::Zero(4, 1);
    H_A.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
    H_b(3, 0) = 1.0;

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
            J_C_eps.push_back(MatrixXd::Zero(2, 3));
        }

        for(int j = 3; j < 6; j++){
            J_C_eps[j] = _p * R0 * _J_expw_eps_33[j - 3];
        }

        MatrixXd J_A_eps(2, eps_dim), J_B_eps(2, eps_dim);
        for(int j = 0; j < 6; j++){
            J_A_eps.col(j) = J_C_eps[j] * ep0;
            J_B_eps.col(j) = J_C_eps[j] * p.row(i).transpose();
        }

        double ATA = (A.transpose() * A)(0, 0);
        double BTB = (B.transpose() * B)(0, 0);
        MatrixXd J_d_eps = MatrixXd::Zero(1, eps_dim);
        // If indeterminate, jacobian is set to zero

        if(ATA != 0 && BTB != 0){
            double sqATA = sqrt(ATA);
            double sqBTB = sqrt(BTB);
            double _sqATA = 1.0 / sqATA;
            double _sqBTB = 1.0 / sqBTB;
            MatrixXd AT = A.transpose();
            MatrixXd BT = B.transpose();
            J_d_eps = (_sqATA * sqBTB * AT * J_A_eps - _sqBTB * sqATA * BT * J_B_eps) / BTB;
        }

        double d0 = A.norm() / B.norm();
        MatrixXd pd0 = p.row(i).transpose() * d0;
        MatrixXd Hpd0 = H_A * pd0 + H_b;

        MatrixXd J_THpd_eps = MatrixXd::Zero(4, eps_dim);
        for(int j = 0; j < 6; j++){
            J_THpd_eps.col(j) = H_A * R0 * J_expe_eps_34[j] * Hpd0;
        }

        J_THpd_eps += H_A * R0 * p.row(i).transpose() * J_d_eps;
        MatrixXd J_hTHpd_eps = H_A.transpose() * J_THpd_eps;

        MatrixXd _X0 = R0 * pd0 + t0;
        double _xz = _X0(2, 0);
        MatrixXd J_Pi_X = MatrixXd::Zero(3, 3);

        if(_xz != 0){
            double _xz2 = _xz * _xz;
            J_Pi_X << 1.0 / _xz, 0, -_X0(0, 0) / _xz2,
                      0, 1.0 / _xz, -_X0(1, 0) / _xz2,
                      0,         0,                 0;
        } else {
            cout << "Found point with z = 0." << endl;
        }

        MatrixXd J_PihTHpd_eps = J_Pi_X * J_hTHpd_eps;
        MatrixXd p_0(3, 1);
        p_0 << _X0(0, 0) / _xz,
               _X0(1, 0) / _xz,
                             1;
        p_0 = p_0 - p_.row(i).transpose();
        
        MatrixXd J_norm_x = sqrt((p_0.transpose() * p_0)(0, 0)) * p_0.transpose();
        J_r_eps.row(i) = J_norm_x * J_PihTHpd_eps;
    }
    cout << J_r_eps << endl;
}

int main(){
    MatrixXd R0(3, 3);
    MatrixXd p = MatrixXd::Random(10, 3);
    MatrixXd p_ = MatrixXd::Random(10, 3);
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
