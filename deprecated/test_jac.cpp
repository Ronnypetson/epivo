#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include "sophus/geometry.hpp"

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
        MatrixXd _p = MatrixXd::Zero(2, 3);
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

        MatrixXd J_A_eps = MatrixXd::Zero(2, eps_dim);
        MatrixXd J_B_eps = MatrixXd::Zero(2, eps_dim);
        for(int j = 0; j < 6; j++){
            J_A_eps.col(j) = J_C_eps[j] * ep0;
            J_B_eps.col(j) = J_C_eps[j] * p.row(i).transpose();
        }

        double ATA = (A.transpose() * A)(0, 0);
        double BTB = (B.transpose() * B)(0, 0);
        MatrixXd J_d_eps = MatrixXd::Zero(1, eps_dim);
        // If indeterminate, jacobian is set to zero

        if(ATA == 0 || BTB == 0){
            continue;
        }

        double sqATA = sqrt(ATA);
        double sqBTB = sqrt(BTB);
        double _sqATA = 1.0 / sqATA;
        double _sqBTB = 1.0 / sqBTB;
        MatrixXd AT = A.transpose();
        MatrixXd BT = B.transpose();
        J_d_eps = (_sqATA * sqBTB * AT * J_A_eps - _sqBTB * sqATA * BT * J_B_eps) / BTB;

        double d0 = A.norm() / B.norm();
        MatrixXd pd0 = p.row(i).transpose() * d0;
        MatrixXd Hpd0 = H_A * pd0 + H_b;

        MatrixXd J_THpd_eps = MatrixXd::Zero(4, eps_dim);
        for(int j = 0; j < 6; j++){
            J_THpd_eps.col(j) = H_A * R0 * J_expe_eps_34[j] * Hpd0;
            //J_THpd_eps.col(j) = H_A * (R0 * J_expe_eps_34[j] * Hpd0 + t0);
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
        
        //MatrixXd J_norm_x = sqrt((p_0.transpose() * p_0)(0, 0)) * p_0.transpose();
        MatrixXd J_norm2_x = 2 * p_0.transpose();
        J_r_eps.row(i) = J_norm2_x * J_PihTHpd_eps;

        // J_r_ep
        MatrixXd J_A_ep = _p * R0;
        MatrixXd J_d_ep = _sqBTB * _sqATA * AT * J_A_ep;
        MatrixXd J_THpd_ep = R0 * p.row(i).transpose() * J_d_ep;
        MatrixXd J_PihTHpd_ep = J_Pi_X * J_THpd_ep;
        J_r_eps.block<1, 3>(i, 6) = J_norm2_x * J_PihTHpd_ep;
        //J_r_eps.block<1, 3>(i, 6) = J_norm2_x * J_PihTHpd_ep - 1E-5 * 2 * ep0.transpose();
    }
}

int res(const MatrixXd &R0,
        const VectorXd &t0,
        const VectorXd &ep0,
        const MatrixXd &p,
        const MatrixXd &p_,
        MatrixXd &r){
    // r (N,1) is the resulting scalar residuals
    assert(t0.size() == 3);
    assert(ep0.size() == 3);
    assert(R0.rows() == 3);
    assert(R0.cols() == 3);
    assert(p.cols() == 3);
    assert(p_.cols() == 3);
    const int n = p.rows();
    assert(r.cols() == 1);
    assert(n > 0);
    assert(p_.rows() == n);
    assert(r.rows() == n);

    for(int i = 0; i < n; i++){
        MatrixXd _p = MatrixXd::Zero(2, 3);
        _p(0, 0) = 1.0;
        _p(1, 1) = 1.0;
        _p(0, 2) = -p_(i, 0);
        _p(1, 2) = -p_(i, 1);
        MatrixXd A(2, 1), B(2, 1);

        A = _p * R0 * ep0; // point R0, inverse ep0
        B = _p * R0 * p.row(i).transpose();
        double d = 0;

        if(B.norm() > 0){
            d = A.norm() / B.norm();
        }

        //cout << A.norm() << " " << B.norm() << " " << d << endl;

        MatrixXd X =  p.row(i).transpose() * d;
        MatrixXd X_ = R0 * X + t0;
        MatrixXd p__ = X_ / X_(2, 0);
        //r(i, 0) = (p_.row(i).transpose() - p__).norm();
        MatrixXd diff = (p_.row(i).transpose() - p__);
        r(i, 0) = (diff.transpose() * diff)(0, 0);
    }
}

int main(){
    srand(time(0));
    const int N = 15;
    MatrixXd R0(3, 3), R(3, 3);
    VectorXd t0(3), t(3);
    VectorXd ep0(3), ep(3);

    const double pi = Sophus::Constants<double>::pi();
    double r = 2.0 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rx = Sophus::SO3d::rotX(r * pi / 4);
    r = 2.0 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Ry = Sophus::SO3d::rotY(r * pi / 4);
    r = 2.0 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rz = Sophus::SO3d::rotZ(r * pi / 4);
    R = Rx.matrix() * Ry.matrix() * Rz.matrix();
    t = 10.0 * MatrixXd::Random(3, 1);
    ep = R.inverse() * t;

    double noise = 5E-1 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rx_noise = Sophus::SO3d::rotX(noise);
    noise = 5E-1 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Ry_noise = Sophus::SO3d::rotY(noise);
    noise = 5E-1 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rz_noise = Sophus::SO3d::rotZ(noise);
    R0 = R * (Rx_noise.matrix() * Ry_noise.matrix() * Rz_noise.matrix());
    t0 = t + 1E-2 * VectorXd::Random(3, 1);
    ep0 = R0.inverse() * t0;

    MatrixXd X = 100.0 * MatrixXd::Random(N, 3);
    MatrixXd p = MatrixXd::Zero(N, 3);
    MatrixXd p_ = MatrixXd::Zero(N, 3);

    for(int i = 0; i < N; i++){
        if(X(i, 2) < 0.0){
            X(i, 2) *= -1.0;
            X(i, 2) += 10.0;
        }
        p.row(i) = X.row(i) / X(i, 2);
        MatrixXd x_ = (R * X.row(i).transpose() + t).transpose();
        p_.row(i) = x_ / x_(2);
    }

    MatrixXd r0 = MatrixXd::Zero(N, 1);
    MatrixXd J = MatrixXd::Zero(N, 9);
    MatrixXd delta = MatrixXd::Zero(9, 1);
    MatrixXd delta_T = MatrixXd::Zero(4, 4);
    MatrixXd delta_ep = MatrixXd::Zero(3, 1);
    MatrixXd H = MatrixXd::Zero(9, 9);
    MatrixXd b = MatrixXd::Zero(9, 1);
    double lambda = 0.01;
    double prev_E = 1E10;
    double epsilon = 1E-8;

    for(int i = 0; i < 60; i++){
        res(R0, t0, ep0, p, p_, r0);
        Dr_Deps(R0, t0, ep0, p, p_, J);

        b = J.transpose() * r0;
        H = J.transpose() * J;
        H = H + lambda * H.diagonal().asDiagonal().toDenseMatrix();

        //delta = - H.block<6, 6>(0, 0).inverse() * b.block<6, 1>(0, 0);
        //delta_T = Sophus::SE3<double>::exp(delta).matrix();
        delta = - H.inverse() * b;
        delta_T = Sophus::SE3<double>::exp(delta.block<6, 1>(0, 0)).matrix();
        delta_ep = delta.block<3, 1>(6, 0);

        if(delta.norm() < epsilon){
            break;
        }

        cout << i
             << " " << H.norm()
             << " " << r0.norm()
             << " " << delta.norm()
             << " " << lambda << endl;

        MatrixXd t0_ = t0 + R0 * delta_T.block<3, 1>(0, 3);
        MatrixXd R0_ = R0 * delta_T.block<3, 3>(0, 0);
        //MatrixXd ep0_ = R0_.inverse() * t0_; // DELET
        MatrixXd ep0_ = ep0 + delta_ep;

        //res(R0_, t0_, ep0, p, p_, r0);
        res(R0_, t0_, ep0_, p, p_, r0);
        //res(R0_, t0_, ep0_, p, p_, r0); // DELET

        double curr_E = r0.norm();
        if(curr_E < prev_E){
            prev_E = curr_E;
            R0 = R0_;
            t0 = t0_;
            ep0 = ep0_;
            //ep0 = ep0_; // DELET
            lambda /= 2.0;
        } else {
            lambda *= 5.0;
        }
    }

    cout << endl;
    cout << (R - R0).norm() << endl;
    cout << endl;
    cout << t.transpose() << endl;
    cout << t0.transpose() << endl;
    cout << t(0, 0) / t0(0, 0) << " "
         << t(1, 0) / t0(1, 0) << " "
         << t(2, 0) / t0(2, 0) << endl;
    cout << endl;
    cout << ep.transpose() << endl;
    cout << ep0.transpose() << endl;
    cout << ep(0, 0) / ep0(0, 0) << " "
         << ep(1, 0) / ep0(1, 0) << " "
         << ep(2, 0) / ep0(2, 0) << endl;
}
