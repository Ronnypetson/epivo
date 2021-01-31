#include <utility>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <map>
#include <fstream>
#include "sophus/geometry.hpp"
#include "test_jac_Rt_gen.hpp"
#include "sequence.hpp"

using namespace std;
using namespace Eigen;


const double huber_delta = 1.0;
const int rep_max = 128;
//map<pair<int, int>, MatrixXd> T_mem;
MatrixXd T0_mem[rep_max][rep_max];


int Dr_Deps(const MatrixXd &Tl0,
            const MatrixXd &Tr0,
            const MatrixXd &p,
            const MatrixXd &p_,
            const bool reverse,
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

    MatrixXd T0 = Tl0 * Tr0; //
    VectorXd t0 = T0.block<3, 1>(0, 3);
    MatrixXd R0 = T0.block<3, 3>(0, 0);
    MatrixXd Rl0 = Tl0.block<3, 3>(0, 0);
    MatrixXd Rr0 = Tr0.block<3, 3>(0, 0);

    MatrixXd S = MatrixXd::Zero(3, 4);
    S.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
    MatrixXd s(4, 1);
    s << 0, 0, 0, 1;

    double rev_sign = reverse ? -1.0 : 1.0;

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
            //J_C_eps[j] = _p * Rl0 * _J_expw_eps_33[j - 3] * Rr0;
            J_C_eps[j] = rev_sign * _p * Rl0 * _J_expw_eps_33[j - 3] * Rr0;
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
            //J_t_eps.col(j) = S * Tl0 * J_expe_eps_44[j] * Tr0 * s;
            J_t_eps.col(j) = rev_sign * S * Tl0 * J_expe_eps_44[j] * Tr0 * s;
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
            //J_THpd_eps.col(j) = Tl0 * J_expe_eps_44[j] * Tr0 * Hpd0;
            J_THpd_eps.col(j) = rev_sign * Tl0 * J_expe_eps_44[j] * Tr0 * Hpd0;
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
    bool reverse = (this->source_id > this->target_id);

    if(this->source_id <= this->target_id){
        if(this->source_id < this->zeta_id){
            Tr0 = T0_mem[this->source_id][this->zeta_id - 1];
        }
        Tl0 = T0_mem[this->zeta_id][this->target_id];
    } else {
        if(this->source_id > this->zeta_id){
            Tr0 = T0_mem[this->zeta_id + 1][this->source_id].inverse();
        }
        Tl0 = T0_mem[this->target_id][this->zeta_id].inverse(); // sign
    }

    Dr_Deps(Tl0, Tr0, p, p_, reverse, J_r_eps);
}


int Levenberg_Marquardt(const int n_zeta,
                        const double epsilon,
                        const vector<pair<int, int> > &reps,
                        const double lambda0,
                        vector<MatrixXd> &T0s,
                        vector<MatrixXd> &pr,
                        vector<MatrixXd> &p_r
                        ){
    // srand(time(0));
    const int N = pr[0].rows(); // Number of points of each reprojection
    // const int n_zeta = 32;
    // vector<pair<int, int> > reps; // First zeta, last zeta. NOT first and last frames
    // double epsilon = 1E-8;
    const int eps_dim = 6;

    // for(int i = 0; i < n_zeta; i++){
    //     for(int j = i; j < n_zeta; j++){
    //         reps.push_back(make_pair(i, j));
    //     }
    // }

    // for(int i = 0; i < n_zeta; i++){
    //     //reps.push_back(make_pair(i, i));
    //     reps.push_back(make_pair(i, min(i + 1, n_zeta - 1)));
    //     reps.push_back(make_pair(min(i + 1, n_zeta - 1), i));
    //     reps.push_back(make_pair(i, min(i + 2, n_zeta - 1)));
    //     reps.push_back(make_pair(min(i + 2, n_zeta - 1), i));
    // }
    //reps.push_back(make_pair(1, 1));
    //reps.push_back(make_pair(0, 1));

    const int n_rep = reps.size();

    //vector<MatrixXd> Ts, T0s;
    // gen_sequence(n_zeta, Ts);
    // vector<MatrixXd> T0s;
    // noise_sequence(Ts, T0s);

    // Compute reprojection data
    vector<MatrixXd> T0r, T0r_; // Tr,
    //vector<MatrixXd> Xr, pr, p_r;
    MatrixXd R0(3, 3), t0(3, 1);
    // for(int i = 0; i < n_rep; i++){
        // int z0, z1;
        // z0 = reps[i].first;
        // z1 = reps[i].second;
        // MatrixXd T = MatrixXd::Identity(4, 4); // composed T
        // MatrixXd T0 = MatrixXd::Identity(4, 4); // composed T0
        // if(z0 <= z1){
        //     for(int j = z0; j <= z1; j++){
        //         T = Ts[j] * T;
        //     }
        // } else {
        //     for(int j = z0; j >= z1; j--){
        //         T = Ts[j].inverse() * T;
        //     }
        // }
        // T0r.push_back(T0);
        // //Tr.push_back(T); // obsolete
        // T0r_.push_back(T0);
        // MatrixXd X, p, p_;
        // gen_points(N, T, X, p, p_);
        // Xr.push_back(X);
        // pr.push_back(p);
        // p_r.push_back(p_);
    // }

    for(int i = 0; i < n_rep; i++){
        T0r.push_back(MatrixXd::Identity(4, 4));
        T0r_.push_back(MatrixXd::Identity(4, 4));
    }

    //gen_scene_sequence(N, n_zeta, reps, Ts, T0s, Xr, pr, p_r);

    const int zeta_dims = n_zeta * eps_dim;
    const int rep_N = n_rep * N;
    MatrixXd r0 = MatrixXd::Zero(rep_N, 1);
    MatrixXd J = MatrixXd::Zero(rep_N, zeta_dims);
    MatrixXd H = MatrixXd::Zero(zeta_dims, zeta_dims);
    MatrixXd b = MatrixXd::Zero(zeta_dims, 1);
    MatrixXd delta = MatrixXd::Zero(zeta_dims, 1);

    // Levenberg-Marquardt
    //double lambda = 0.01;
    double lambda = lambda0;
    double prev_E = 1E10;
    for(int i = 0; i < 60; i++){
        r0 = MatrixXd::Zero(rep_N, 1);
        J = MatrixXd::Zero(rep_N, zeta_dims);

        // Load aggregated sub-poses
        for(int j = 0; j < n_zeta; j++){
            MatrixXd sT = T0s[j];
            T0_mem[j][j] = T0s[j];
            for(int k = j + 1; k < n_zeta; k++){
                sT = T0s[k] * sT;
                T0_mem[j][k] = sT; // .replicate(1, 1)
            }
        }

        // Update T0r
        for(int j = 0; j < n_rep; j++){
            int z0, z1;
            z0 = reps[j].first;
            z1 = reps[j].second;

            if(z0 <= z1){
                T0r[j] = T0_mem[z0][z1];
            } else {
                T0r[j] = T0_mem[z1][z0].inverse();
            }
        }

        // Concatenate r0s
        for(int j = 0; j < n_rep; j++){
            //cout << T0r[j] << endl;
            //double g;
            //cin >> g;

            MatrixXd r0_rep = MatrixXd::Zero(N, 1);
            R0 = T0r[j].block<3, 3>(0, 0);
            t0 = T0r[j].block<3, 1>(0, 3);

            res(R0, t0, pr[j], p_r[j], r0_rep);
            //r0.block<N, 1>(j * N, 0) = r0_rep;
            r0.block(j * N, 0, N, 1) = r0_rep;
        }

        // Concatenate J through zetas and reprojections
        for(int j = 0; j < n_rep; j++){
            const int z0 = reps[j].first;
            const int z1 = reps[j].second;
            if(z0 <= z1){
                MatrixXd Jrep = MatrixXd::Zero(N, (z1 - z0 + 1) * eps_dim);
                for(int k = z0; k <= z1; k++){
                    // Compute J_res_zeta
                    MatrixXd Jz = MatrixXd::Zero(N, eps_dim);
                    RepJacobian Jr(k, z0, z1);
                    Jr.compute(pr[j], p_r[j], T0s, Jz);
                    //Jrep.block<N, eps_dim>(0, (k - z0) * eps_dim) = Jz;
                    Jrep.block(0, (k - z0) * eps_dim, N, eps_dim) = Jz;
                }
                //for(int k = z0; k <= z1; k++){
                //    J.block<N, eps_dim>(j * N, k * eps_dim)
                //            = Jrep.block<N, eps_dim>(0, (k - z0) * eps_dim);
                //}
                J.block(j * N, z0 * eps_dim, N, (z1 - z0 + 1) * eps_dim) = Jrep;
            } else {
                MatrixXd Jrep = MatrixXd::Zero(N, (z0 - z1 + 1) * eps_dim);
                for(int k = z0; k >= z1; k--){
                    // Compute J_res_zeta
                    MatrixXd Jz = MatrixXd::Zero(N, eps_dim);
                    RepJacobian Jr(k, z0, z1);
                    Jr.compute(pr[j], p_r[j], T0s, Jz);
                    //Jrep.block<N, eps_dim>(0, (k - z1) * eps_dim) = Jz;
                    Jrep.block(0, (k - z1) * eps_dim, N, eps_dim) = Jz;
                }
                //for(int k = z0; k >= z1; k--){
                //    J.block<N, eps_dim>(j * N, k * eps_dim)
                //            = Jrep.block<N, eps_dim>(0, (k - z1) * eps_dim);
                //}
                J.block(j * N, z1 * eps_dim, N, (z0 - z1 + 1) * eps_dim) = Jrep;
            }
        }

        b = J.transpose() * r0;
        H = J.transpose() * J;
        H = H + lambda * H.diagonal().asDiagonal().toDenseMatrix();

        delta = -H.inverse() * b;

        vector<MatrixXd> T0s_;
        for(int j = 0; j < T0s.size(); j++){
            MatrixXd delta_k = delta.block<eps_dim, 1>(j * eps_dim, 0);
            MatrixXd delta_T = Sophus::SE3<double>::exp(delta_k).matrix();
            MatrixXd new_T = T0s[j] * delta_T;
            T0s_.push_back(new_T);
        }

        if(delta.norm() < epsilon){
            break;
        }

        // Compute candidate T0r_
        for(int j = 0; j < n_rep; j++){
            int z0, z1;
            z0 = reps[j].first;
            z1 = reps[j].second;
            MatrixXd T0 = MatrixXd::Identity(4, 4); // composed T0
            
            if(z0 <= z1){
                for(int k = z0; k <= z1; k++){
                    T0 = T0s_[k] * T0;
                }
                
            } else {
                for(int k = z0; k >= z1; k--){
                    T0 = T0s_[k].inverse() * T0;
                }
            }
            T0r_[j] = T0;
        }

        // Concatenate candidate r0s
        for(int j = 0; j < n_rep; j++){
            MatrixXd r0_rep = MatrixXd::Zero(N, 1);
            
            R0 = T0r_[j].block<3, 3>(0, 0);
            t0 = T0r_[j].block<3, 1>(0, 3);

            res(R0, t0, pr[j], p_r[j], r0_rep);
            //r0.block<N, 1>(j * N, 0) = r0_rep;
            r0.block(j * N, 0, N, 1) = r0_rep;
        }

        //cout << "candidate r0" << endl << endl;

        double curr_E = r0.norm();
        if(curr_E < prev_E){
            prev_E = curr_E;

            for(int j = 0; j < T0s.size(); j++){
                T0s[j] = T0s_[j];
            }

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


int main(){
    srand(time(0));
    const int N = 15; // Number of points of each reprojection
    const int n_zeta = 10;
    const double epsilon = 1E-8;
    //const int eps_dim = 6;
    vector<pair<int, int> > reps; // First zeta, last zeta. NOT first and last frames

    // for(int i = 0; i < n_zeta; i++){
    //     reps.push_back(make_pair(i, i));
    //     //reps.push_back(make_pair(i, min(i + 1, n_zeta - 1)));
    //     reps.push_back(make_pair(min(i + 1, n_zeta - 1), i));
    //     //reps.push_back(make_pair(i, min(i + 2, n_zeta - 1)));
    //     //reps.push_back(make_pair(min(i + 2, n_zeta - 1), i));
    //     //if(i < n_zeta - 1){
    //     //    reps.push_back(make_pair(i + 1, i));
    //     //}
    // }
    reps.push_back(make_pair(n_zeta - 1, 0));
    //reps.push_back(make_pair(0, n_zeta - 1));

    vector<MatrixXd> Ts, T0s;
    vector<MatrixXd> Xr, pr, p_r;
    gen_scene_sequence(N, n_zeta, reps, Ts, T0s, Xr, pr, p_r);

    double lambda0 = 0.01;
    Levenberg_Marquardt(n_zeta,
                        epsilon,
                        reps,
                        lambda0,
                        T0s,
                        pr,
                        p_r);
    
    // Save output
    ofstream est, gt;
    est.open("est.pose");
    gt.open("gt.pose");

    MatrixXd R(3, 3), t(3, 1);
    MatrixXd R0(3, 3), t0(3, 1);
    double scale;
    MatrixXd T = MatrixXd::Identity(4, 4);
    MatrixXd T0 = MatrixXd::Identity(4, 4);
    for(int i = 0; i < Ts.size(); i++){
        T = Ts[i] * T;
        T0 = T0s[i] * T0;
    }

    t = T.block<3, 1>(0, 3);
    t0 = T0.block<3, 1>(0, 3);
    scale = ( t(0, 0) / t0(0, 0)
            + t(1, 0) / t0(1, 0)
            + t(2, 0) / t0(2, 0)) / 3.0;
    
    MatrixXd gT0 = MatrixXd::Identity(4, 4);
    MatrixXd gT = MatrixXd::Identity(4, 4);
    est << gT0 << "\n\n";
    gt << gT << "\n\n";
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
        
        T0s[i].block<3, 1>(0, 3) *= scale;

        gT0 = T0s[i] * gT0;
        gT = Ts[i] * gT;

        est << gT0 << "\n\n";
        gt << gT << "\n\n";
    }

    est.close();
    gt.close();
}
