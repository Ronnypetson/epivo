#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include "sophus/geometry.hpp"

using namespace std;
using namespace Eigen;

int gen_T(MatrixXd &T){
    MatrixXd R(3, 3); //, pR(3, 3);
    VectorXd t(3); //, pt(3);

    const double pi = Sophus::Constants<double>::pi();
    double r = 2.0 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rx = Sophus::SO3d::rotX(r * pi / 12);
    r = 2.0 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Ry = Sophus::SO3d::rotY(r * pi / 12);
    r = 2.0 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rz = Sophus::SO3d::rotZ(r * pi / 12);
    R = Rx.matrix() * Ry.matrix() * Rz.matrix();
    t = 1.0 * MatrixXd::Random(3, 1);
    if(t(2, 0) < 0.0){
        t(2, 0) *= -1.0;
    }
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    T.block<1, 4>(3, 0) << 0, 0, 0, 1;
}

int gen_sequence(const int n, vector<MatrixXd> &Ts){
    for(int i = 0; i < n; i++){
        MatrixXd T(4, 4);
        gen_T(T);
        Ts.push_back(T);
    }
}

int T_noise(MatrixXd &T){
    double noise = 1E-1 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rx_noise = Sophus::SO3d::rotX(noise);
    noise = 1E-1 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Ry_noise = Sophus::SO3d::rotY(noise);
    noise = 1E-1 * (0.5 - (double) rand() / (RAND_MAX));
    Sophus::SO3d Rz_noise = Sophus::SO3d::rotZ(noise);
    
    T.block<3, 3>(0, 0) = Rx_noise.matrix() * Ry_noise.matrix() * Rz_noise.matrix();
    T.block<3, 1>(0, 3) = 1E-1 * VectorXd::Random(3, 1);
    T.block<1, 4>(3, 0) << 0, 0, 0, 1;
}

int noise_sequence(const vector<MatrixXd> &Ts, vector<MatrixXd> &T0s){
    for(int i = 0; i < Ts.size(); i++){
        MatrixXd Tn(4, 4);
        T_noise(Tn);
        Tn = Ts[i] * Tn;
        MatrixXd t = Tn.block<3, 1>(0, 3);
        t = t / t.norm(); // Comment to test validity of the optimization
        Tn.block<3, 1>(0, 3) = t;
        T0s.push_back(Tn);
    }
}

int gen_points(const int N,
               const MatrixXd T,
               MatrixXd &X,
               MatrixXd &p,
               MatrixXd &p_){
    //X = 100.0 * MatrixXd::Random(N, 3);
    X = MatrixXd::Zero(N, 3);
    p = MatrixXd::Zero(N, 3);
    p_ = MatrixXd::Zero(N, 3);

    MatrixXd R = T.block<3, 3>(0, 0);
    MatrixXd t = T.block<3, 1>(0, 3);
    double mag_t = t.norm();

    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1E-3);

    for(int i = 0; i < N; i++){
        MatrixXd x_ = MatrixXd::Zero(1, 3);
        do {
            X.row(i) = 10 * mag_t * MatrixXd::Random(1, 3);
            if(X(i, 2) < 0.0){
                X(i, 2) *= -1.0;
                X(i, 2) += 10.0;
            }
            x_ = (R * X.row(i).transpose() + t).transpose();
        } while(x_(0, 2) <= 10.0);

        p.row(i) = X.row(i) / X(i, 2);
        p_.row(i) = x_ / x_(0, 2);

        // Add Gaussian noise to measurements
        for(int j = 0; j < 2; j++){
            p(i, j) += distribution(generator);
        }

        for(int j = 0; j < 2; j++){
            p_(i, j) += distribution(generator);
        }
    }
}
