#include <utility>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include "sophus/geometry.hpp"
#include "test_jac_Rt_gen.hpp"
#include "sequence.hpp"

using namespace std;
using namespace Eigen;

const double huber_delta = 1.0;

int Dr_Deps(const MatrixXd &Tl0,
            const MatrixXd &Tr0,
            const MatrixXd &p,
            const MatrixXd &p_,
            MatrixXd &J_r_eps){
    // Computes jacobian J_r_eps (N,9) of residuals w.r.t eps in the algebra of SE(3) at eps = 0
    // p (N,3) and p_ (N,3) are point matches
    // R0 (3,3), t0 (3), and ep0 (3) are the current values
    // eps order: tr (3), rot (3), ep (3)
    assert(Tl0.rows() == 4);
    assert(Tl0.cols() == 4);
    assert(Tr0.rows() == 4);
    assert(Tr0.cols() == 4);
    assert(p.cols() == 3);
    assert(p_.cols() == 3);
    const int n = p.rows();
    const int eps_dim = 6;
    assert(J_r_eps.cols() == eps_dim);
    assert(n > 0);
    assert(p_.rows() == n);
    assert(J_r_eps.rows() == n);

    MatrixXd J_expw_eps = MatrixXd::Zero(9, 6); // 0_9x3 J_expw_w 0_9x3
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

    MatrixXd J_expe_eps = MatrixXd::Zero(12, 6);
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

    MatrixXd J_expe_eps_3xD = MatrixXd::Zero(3, eps_dim);
    for(int i = 0; i < eps_dim; i++){
        J_expe_eps_3xD.col(i) = J_expe_eps_34[i].col(3);
    }

    vector<MatrixXd> J_expe_eps_44;
    for(int i = 0; i < eps_dim; i++){
        MatrixXd M = MatrixXd::Zero(4, 4);
        M.block<3, 4>(0, 0) = J_expe_eps_34[i];
        J_expe_eps_44.push_back(M);
    }

    MatrixXd H_A = MatrixXd::Zero(4, 3);
    MatrixXd H_b = MatrixXd::Zero(4, 1);
    H_A.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
    H_b(3, 0) = 1.0;

    MatrixXd T0 = Tl0 * Tr0;
    VectorXd t0 = T0.block<3, 1>(0, 3);
    MatrixXd R0 = T0.block<3, 3>(0, 0);
    MatrixXd Rl0 = Tl0.block<3, 3>(0, 0);
    MatrixXd Rr0 = Tr0.block<3, 3>(0, 0);

    MatrixXd S = MatrixXd::Zero(3, 4);
    S.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
    MatrixXd s(4, 1);
    s << 0, 0, 0, 1;

    for(int i = 0; i < n; i++){
        MatrixXd _p = MatrixXd::Zero(2, 3);
        _p(0, 0) = 1.0;
        _p(1, 1) = 1.0;
        _p(0, 2) = -p_(i, 0);
        _p(1, 2) = -p_(i, 1);
        MatrixXd A(2, 1), B(2, 1);
        A = _p * t0;
        B = _p * R0 * p.row(i).transpose();
        
        // t0 .. t2 r0 .. r2 e0 .. e2
        vector<MatrixXd> J_C_eps;
        for(int j = 0; j < eps_dim; j++){
            J_C_eps.push_back(MatrixXd::Zero(2, 3));
        }

        for(int j = 3; j < 6; j++){
            J_C_eps[j] = _p * Rl0 * _J_expw_eps_33[j - 3] * Rr0;
        }

        MatrixXd J_A_eps = MatrixXd::Zero(2, eps_dim);
        MatrixXd J_B_eps = MatrixXd::Zero(2, eps_dim);
        for(int j = 0; j < 6; j++){
            //J_A_eps.col(j) = J_C_eps[j] * ep0;
            J_B_eps.col(j) = J_C_eps[j] * p.row(i).transpose();
        }

        MatrixXd J_t_eps = MatrixXd::Zero(3, eps_dim);
        //J_t_eps = R0 * J_expe_eps_3xD;
        //J_A_eps += _p * R0 * J_t_eps;
        for(int j = 0; j < eps_dim; j++){
            J_t_eps.col(j) = S * Tl0 * J_expe_eps_44[j] * Tr0 * s;
        }

        J_A_eps += _p * J_t_eps;

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
            //J_THpd_eps.col(j) = H_A * R0 * J_expe_eps_34[j] * Hpd0;
            J_THpd_eps.col(j) = Tl0 * J_expe_eps_44[j] * Tr0 * Hpd0;
        }

        //J_THpd_eps += H_A * R0 * p.row(i).transpose() * J_d_eps; // check again
        J_THpd_eps += T0 * H_A * p.row(i).transpose() * J_d_eps;
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
        //J_r_eps.row(i) = J_norm2_x * J_PihTHpd_eps;
        MatrixXd J_norm1_x = p_0.transpose() / p_0.norm();

        if((p_0.transpose() * p_0)(0, 0) <= huber_delta){
            J_r_eps.row(i) = J_norm2_x * J_PihTHpd_eps / 2.0;
        } else {
            J_r_eps.row(i) = huber_delta * J_norm1_x * J_PihTHpd_eps;
        }
    }
}

int res(const MatrixXd &R0,
        const VectorXd &t0,
        const MatrixXd &p,
        const MatrixXd &p_,
        MatrixXd &r){
    // r (N,1) is the resulting scalar residuals
    assert(t0.size() == 3);
    //assert(ep0.size() == 3);
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

        A = _p * t0; // point R0, inverse ep0
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
        //r(i, 0) = (diff.transpose() * diff)(0, 0);
        r(i, 0) = (diff.transpose() * diff)(0, 0) / 2.0;
        if(r(i, 0) > huber_delta){
            r(i, 0) = huber_delta * (sqrt(r(i, 0)) - huber_delta / 2.0);
        }
    }
}

void RepJacobian::compute(const MatrixXd &p,
                          const MatrixXd &p_,
                          const vector<MatrixXd> &T0s,
                          MatrixXd &J_r_eps){
    // T0s are ordered in camera frame order
    MatrixXd Tl0 = MatrixXd::Identity(4, 4);
    MatrixXd Tr0 = MatrixXd::Identity(4, 4);

    for(int i = this->source_id; i < this->zeta_id; i++){
        Tr0 = T0s[i] * Tr0;
    }

    for(int i = this->zeta_id; i <= this->target_id; i++){
        Tl0 = T0s[i] * Tl0;
    }

    Dr_Deps(Tl0, Tr0, p, p_, J_r_eps);
}

int main(){
    srand(time(0));
    const int N = 15;
    const int n_zeta = 3;
    const int n_rep = n_zeta * (n_zeta + 1) / 2;
    vector<pair<int, int> > reps; // First zeta, last zeta. NOT first and last frames
    double epsilon = 1E-8;
    const int eps_dim = 6;

    for(int i = 0; i < n_zeta; i++){
        for(int j = i; j < n_zeta; j++){
            reps.push_back(make_pair(i, j));
        }
    }

    vector<MatrixXd> Ts;
    gen_sequence(n_zeta, Ts);
    vector<MatrixXd> T0s;
    noise_sequence(Ts, T0s);

    for(int i = 0; i < n_rep; i++){
        MatrixXd X, p, p_;
        int z0, z1;
        z0 = reps[i].first;
        z1 = reps[i].second;
        
        //cout << z0 << " " << z1 << endl << endl;

        MatrixXd T = MatrixXd::Identity(4, 4); // composed T
        MatrixXd T0 = MatrixXd::Identity(4, 4); // composed T0
        MatrixXd R0(3, 3), t0(3, 1);

        for(int j = z0; j <= z1; j++){
            T = Ts[j] * T;
            T0 = T0s[j] * T0;
        }
        gen_points(N, T, X, p, p_);

        int zeta_dims = (z1 - z0 + 1) * eps_dim;
        MatrixXd r0 = MatrixXd::Zero(N, 1);
        MatrixXd J = MatrixXd::Zero(N, zeta_dims);
        MatrixXd H = MatrixXd::Zero(zeta_dims, zeta_dims);
        MatrixXd b = MatrixXd::Zero(zeta_dims, 1);
        MatrixXd delta = MatrixXd::Zero(zeta_dims, 1);
        //MatrixXd delta_T = MatrixXd::Zero(4, 4);

        // Levenberg-Marquadt
        double lambda = 0.01;
        double prev_E = 1E10;
        for(int j = 0; j < 60; j++){
            R0 = T0.block<3, 3>(0, 0);
            t0 = T0.block<3, 1>(0, 3);
            res(R0, t0, p, p_, r0);

            for(int k = z0; k <= z1; k++){
                // Compute J_res_zeta
                MatrixXd Jz = MatrixXd::Zero(N, eps_dim);
                RepJacobian Jr(k, z0, z1);
                Jr.compute(p, p_, T0s, Jz);
                J.block<N, eps_dim>(0, (k - z0) * eps_dim) = Jz;
            }

            b = J.transpose() * r0;
            H = J.transpose() * J;
            H = H + lambda * H.diagonal().asDiagonal().toDenseMatrix();

            delta = -H.inverse() * b;

            vector<MatrixXd> T0s_;
            MatrixXd T0_ = MatrixXd::Identity(4, 4); // composite new T0
            for(int k = z0; k <= z1; k++){
                MatrixXd delta_k = delta.block<eps_dim, 1>((k - z0) * eps_dim, 0);
                MatrixXd delta_T = Sophus::SE3<double>::exp(delta_k).matrix();
                MatrixXd new_T = T0s[k] * delta_T;
                T0s_.push_back(new_T);
                T0_ = new_T * T0_;
            }

            if(delta.norm() < epsilon){
                break;
            }

            MatrixXd t0_ = T0_.block<3, 1>(0, 3);
            MatrixXd R0_ = T0_.block<3, 3>(0, 0);

            res(R0_, t0_, p, p_, r0);

            double curr_E = r0.norm();
            if(curr_E < prev_E){
                prev_E = curr_E;

                for(int k = z0; k <= z1; k++){
                    T0s[k] = T0s_[k - z0];
                }

                T0 = T0_;

                lambda /= 2.0;
            } else {
                lambda *= 5.0;
            }
        }

        cout << " " << H.norm()
             << " " << r0.norm()
             << " " << delta.norm()
             << " " << lambda << endl;

        cout << endl;
    }

    MatrixXd R, t, R0, t0;
    for(int i = 0; i < n_zeta; i++){
        R = Ts[i].block<3, 3>(0, 0);
        t = Ts[i].block<3, 1>(0, 3);

        R0 = T0s[i].block<3, 3>(0, 0);
        t0 = T0s[i].block<3, 1>(0, 3);

        cout << endl;
        cout << (R - R0).norm() << endl;
        cout << endl;
        cout << t.transpose() << endl;
        cout << t0.transpose() << endl;
        cout << t(0, 0) / t0(0, 0) << " "
             << t(1, 0) / t0(1, 0) << " "
             << t(2, 0) / t0(2, 0) << endl;
    }

    // MatrixXd R0(3, 3), R(3, 3); //, pR(3, 3);
    // VectorXd t0(3), t(3); //, pt(3);
    // MatrixXd T0(4, 4), T(4, 4), Tn(4, 4);    

    // gen_T(T);
    // R = T.block<3, 3>(0, 0);
    // t = T.block<3, 1>(0, 3);
    // T_noise(Tn);
    // T0 = T * Tn;
    // R0 = T0.block<3, 3>(0, 0);
    // t0 = T0.block<3, 1>(0, 3);
    // t0 = t0 / t0.norm();
    
    // MatrixXd Tl0 = MatrixXd::Identity(4, 4);
    // MatrixXd Tr0 = MatrixXd::Identity(4, 4);
    // Tl0.block<3, 3>(0, 0) = R0;
    // Tl0.block<3, 1>(0, 3) = t0;

    // MatrixXd r0 = MatrixXd::Zero(N, 1);
    // MatrixXd J = MatrixXd::Zero(N, 6);
    // MatrixXd delta = MatrixXd::Zero(6, 1);
    // MatrixXd delta_T = MatrixXd::Zero(4, 4);
    // //MatrixXd delta_ep = MatrixXd::Zero(3, 1);
    // MatrixXd H = MatrixXd::Zero(6, 6);
    // MatrixXd b = MatrixXd::Zero(6, 1);

    // for(int i = 0; i < 60; i++){
    //     res(R0, t0, p, p_, r0);
    //     Dr_Deps(Tl0, Tr0, p, p_, J);

    //     b = J.transpose() * r0;
    //     H = J.transpose() * J;
    //     H = H + lambda * H.diagonal().asDiagonal().toDenseMatrix();

    //     delta = - H.inverse() * b;
    //     delta_T = Sophus::SE3<double>::exp(delta).matrix();

    //     if(delta.norm() < epsilon){
    //         break;
    //     }

    //     cout << i
    //          << " " << H.norm()
    //          << " " << r0.norm()
    //          << " " << delta.norm()
    //          << " " << lambda << endl;

    //     MatrixXd t0_ = t0 + R0 * delta_T.block<3, 1>(0, 3);
    //     MatrixXd R0_ = R0 * delta_T.block<3, 3>(0, 0);

    //     res(R0_, t0_, p, p_, r0);

    //     double curr_E = r0.norm();
    //     if(curr_E < prev_E){
    //         prev_E = curr_E;
    //         R0 = R0_;
    //         t0 = t0_;
    //         Tl0.block<3, 3>(0, 0) = R0_;
    //         Tl0.block<3, 1>(0, 3) = t0_;
    //         lambda /= 2.0;
    //     } else {
    //         lambda *= 5.0;
    //     }
    // }

    // cout << endl;
    // cout << (R - R0).norm() << endl;
    // cout << endl;
    // cout << t.transpose() << endl;
    // cout << t0.transpose() << endl;
    // cout << t(0, 0) / t0(0, 0) << " "
    //      << t(1, 0) / t0(1, 0) << " "
    //      << t(2, 0) / t0(2, 0) << endl;
}
