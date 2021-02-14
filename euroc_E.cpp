#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/core/eigen.hpp>

//#include <opencv2/features2d.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include "test_jac_Rt_gen.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    std::getline(indata, line);
    while (std::getline(indata, line)) {
        //if(rows == 0){
        //    rows++;
        //    continue;
        //}
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}


void load_fns(const string &path, vector<string> &fns){
    ifstream indata;
    indata.open(path);
    string line;
    uint rows = 0;
    while (getline(indata, line)) {
        if(rows == 0){
            rows++;
            continue;
        }
        stringstream lineStream(line);
        string cell;
        uint col = 0;
        while (getline(lineStream, cell, ',')) {
            if(col == 0){
                fns.push_back(cell);
            }
            col++;
        }
        //rows++;
    }
}


void quat_to_R(const MatrixXd q, MatrixXd &R){
    double qw = q(0, 0);
    double qx = q(0, 1);
    double qy = q(0, 2);
    double qz = q(0, 3);

    const double n = 1.0 / sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    qx *= n;
    qy *= n;
    qz *= n;
    qw *= n;

    R << 1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw,
         2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw,
         2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy;
}


int main(){
    int h, w;
    h = 480;
    w = 752;

    Mat cam = (Mat_<float>(3, 3) << 458.654,  0.0,      367.215,
                                    0.0,      457.296,  248.375,
                                    0.0,      0.0,      1.0);
    Mat dist = (Mat_<float>(1, 5) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0);
    Mat rect = (Mat_<float>(3, 3) << 0.999966347530033,    -0.001422739138722922, 0.008079580483432283,
                                     0.001365741834644127,  0.9999741760894847,   0.007055629199258132,
                                    -0.008089410156878961, -0.007044357138835809, 0.9999424675829176);
    Mat proj = (Mat_<float>(3, 4) << 435.2046959714599, 0.0,               367.4517211914062, 0.0,
                                     0.0,               435.2046959714599, 252.2008514404297, 0.0,
                                     0.0,               0.0,               1.0,               0.0);
    
    Mat map1, map2;
    initUndistortRectifyMap(cam,
                            dist,
                            rect,
                            proj,
                            Size(w, h),
                            map1.type(),
                            map1,
                            map2);

    MatrixXd cam_(3, 3), new_cam_(3, 3), new_cam_inv(3, 3);
    cam_ << 458.654,  0.0,      367.215,
            0.0,      457.296,  248.375,
            0.0,      0.0,      1.0; // Change to Euroc MAV camera
    cam_ = cam_.inverse();

    MatrixXd T_DC(4, 4), T_DC_(4, 4);
    T_DC << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
            0.0, 0.0, 0.0, 1.0;
    T_DC_ = T_DC.inverse();

    new_cam_ << 435.2046959714599, 0.0,               367.4517211914062,
                0.0,               435.2046959714599, 252.2008514404297,
                0.0,               0.0,               1.0;

    new_cam_inv = new_cam_.inverse();

    string base_dir = "/home/ronnypetson/Downloads/";
    string seq = "V2_01_easy/";
    //string seq = "MH_01_easy/";
    MatrixXd poses = load_csv<MatrixXd>(base_dir + seq + "mav0/state_groundtruth_estimate0/data.csv");
    vector<MatrixXd> X;
    vector<int> limits;
    string src_fn, tgt_fn;
    string base_img = base_dir + seq + "mav0/cam0/data/";
    string stamps_fn = base_dir + seq + "mav0/cam0/data.csv";
    vector<string> fns;
    load_fns(stamps_fn, fns);

    int last = min(2200, (int)(fns.size() - 1));

    Mat new_cam;
    Rect *roi = new Rect();
    new_cam = (Mat_<float>(3, 3) << 435.2046959714599, 0.0,               367.4517211914062,
                                    0.0,               435.2046959714599, 252.2008514404297,
                                    0.0,               0.0,               1.0);

    double x, y, _h, _w;
    vector<MatrixXd> all_T;
    Mat src_, tgt_;
    MatrixXd cT = MatrixXd::Identity(4, 4);
    for(int i = 28; i < last; i++){
        cout << i << " ";
        src_fn = base_img + fns[i];
        tgt_fn = base_img + fns[i + 1];

        Mat src = imread(src_fn, IMREAD_GRAYSCALE);
        Mat tgt = imread(tgt_fn, IMREAD_GRAYSCALE);

        // undistort
        remap(src, src_, map1, map2, INTER_LINEAR);
        src_.copyTo(src);
        
        // undistort
        remap(tgt, tgt_, map1, map2, INTER_LINEAR);
        tgt_.copyTo(tgt);
        
        vector<KeyPoint> kp0, kp_; // kp1,
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
        vector<Mat> descriptor;

        detector->detect(src, kp0, Mat());
        //detector->detect(tgt, kp1, Mat());

        vector<Point2f> pt0, pt1_;
        KeyPoint::convert(kp0, pt0);

        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(src,
                             tgt,
                             pt0,
                             pt1_,
                             status,
                             err);
        
        vector<Point2f> _cpt0, _cpt1_;
        for(int j = 0; j < status.size(); j++){
            if((int)status[j] == 1){
                _cpt0.push_back(pt0[j]);
                _cpt1_.push_back(pt1_[j]);
            }
        }

        vector<uchar> mask_ess;
        Mat ess = findEssentialMat(_cpt0,
                                   _cpt1_,
                                   new_cam,
                                   RANSAC,
                                   0.99,
                                   0.3,
                                   mask_ess);
        
        vector<Point2f> cpt0, cpt1_;
        for(int j = 0; j < mask_ess.size(); j++){
            if((int)mask_ess[j] == 1){
                cpt0.push_back(_cpt0[j]);
                cpt1_.push_back(_cpt1_[j]);
            }
        }

        MatrixXd R(3, 3), t(3, 1);
        MatrixXd T = MatrixXd::Identity(4, 4);
        MatrixXd pT = MatrixXd::Identity(4, 4);

        MatrixXd _pT = MatrixXd::Identity(4, 4);
        MatrixXd _T = MatrixXd::Identity(4, 4);
        MatrixXd qT, qpT;

        int cnt_found = 0;
        for(int j = 9.25 * (i - 28); j < poses.rows(); j++){
            if(abs(poses(j, 0) - stod(fns[i])) < 5760512 - 760576){
                qpT = poses.row(j).block<1, 4>(0, 4);
                quat_to_R(qpT, R);
                _pT.block<3, 3>(0, 0) = R;
                _pT.block<3, 1>(0, 3) = poses.row(j).block<1, 3>(0, 1);
                cnt_found++;
                break;
            }
        }
        for(int j = 9.25 * (i - 28); j < poses.rows(); j++){
            if(abs(poses(j, 0) - stod(fns[i + 1])) < 5760512 - 760576){
                qT = poses.row(j).block<1, 4>(0, 4);
                quat_to_R(qT, R);
                _T.block<3, 3>(0, 0) = R;
                _T.block<3, 1>(0, 3) = poses.row(j).block<1, 3>(0, 1);
                cnt_found++;
                break;
            }
        }
        assert(cnt_found == 2);

        // Initialize with essential matrix
        Mat rot, tr;
        recoverPose(ess, cpt0, cpt1_, new_cam, rot, tr); // mask_ess
        MatrixXd dT = MatrixXd::Identity(4, 4);

        //T = _T * T_DC;
        //pT = _pT * T_DC;

        MatrixXd erot, etr;
        cv2eigen(rot, erot);
        cv2eigen(tr, etr);

        //erot = MatrixXd::Identity(3, 3);
        //etr << 0.0, 0.0, 1.0;

        if(erot.trace() < 3.0 * 0.9){
            erot = MatrixXd::Identity(3, 3);
            etr << 0.0, 0.0, 1.0;
        }

        if(etr.norm() < 1E-5){
            etr << 0.0, 0.0, 1.0;
        }

        // Run Levenberg-Marquardt
        vector<pair<int, int> > reps;
        reps.push_back(make_pair(0, 0));

        vector<MatrixXd> T0s;
        dT.block<3, 3>(0, 0) = erot;
        dT.block<3, 1>(0, 3) = etr;
        T0s.push_back(dT); // dT.inverse()

        vector<MatrixXd> pr, p_r;
        int N = min(48, (int)cpt0.size());
        MatrixXd pr_(N, 3), p_r_(N, 3);
        for(int j = 0; j < N; j++){
            pr_.row(j) << cpt0[j].x, cpt0[j].y, 1.0;
            p_r_.row(j) << cpt1_[j].x, cpt1_[j].y, 1.0;

            pr_.row(j) = new_cam_inv * pr_.row(j).transpose();
            p_r_.row(j) = new_cam_inv * p_r_.row(j).transpose();
        }
        pr.push_back(pr_);
        p_r.push_back(p_r_);

        // Levenberg_Marquardt(1,
        //                     1e-8,
        //                     reps,
        //                     1e-2,
        //                     T0s,
        //                     pr,
        //                     p_r);
        
        dT = ((_pT * T_DC).inverse() * (_T * T_DC)).inverse();
        double scale = dT.block<3, 1>(0, 3).norm();
        T0s[0].block<3, 1>(0, 3) /= T0s[0].block<3, 1>(0, 3).norm();
        dT.block<3, 1>(0, 3) = T0s[0].block<3, 1>(0, 3) * scale;
        dT.block<3, 3>(0, 0) = T0s[0].block<3, 3>(0, 0);
        
        dT = dT.inverse();

        //cout << (_pT * T_DC).inverse() * (_T * T_DC) << endl << endl;
        //cout << dT << endl << endl;
        //double g;
        //cin >> g;

        pT = cT; // * T_DC;
        cT = cT * dT;
        T = cT; // * T_DC;

        all_T.push_back(pT); // * T_DC

        //MatrixXd dT = pT.inverse() * T;

        dT = dT.inverse();
        R = dT.block<3, 3>(0, 0);
        t = dT.block<3, 1>(0, 3);

        MatrixXd p_(2, 3), p(3, 1), x_;
        MatrixXd A, B, cp(3, 1), cp_(3, 1);
        double d;
        limits.push_back(X.size());
        for(int j = 0; j < cpt0.size(); j++){
            cp_ << cpt1_[j].x, cpt1_[j].y, 1.0;
            cp << cpt0[j].x, cpt0[j].y, 1.0;
            cp_ = new_cam_inv * cp_;
            cp = new_cam_inv * cp;

            p_ << 1.0, 0.0, -cp_(0, 0), 0.0, 1.0, -cp_(1, 0);
            A = p_ * t;
            B = p_ * R * cp;
            if(B.norm() > 1E-2){
                d = A.norm() / B.norm();
                x_ = pT.block<3, 3>(0, 0) * (d * cp)
                    + pT.block<3, 1>(0, 3); // debug
                X.push_back(x_);
            }
        }
    }

    ofstream pt_cloud, lims, poses_f;
    pt_cloud.open("pts.cld");

    for(int i = 0; i < X.size(); i++){
        pt_cloud << X[i].transpose() << "\n\n";
    }

    pt_cloud.close();

    lims.open("lims");

    for(int i = 0; i < limits.size(); i++){
        lims << limits[i] << " ";
    }

    lims.close();

    poses_f.open("euroc.T");
    for(int i = 0; i < all_T.size(); i++){
        poses_f << all_T[i] << "\n\n";
    }

    poses_f.close();

    //Mat ess = findEssentialMat(cpt0, cpt1_, cam);
    //Mat rot, tr;
    //recoverPose(ess, cpt0, cpt1_, cam, rot, tr);

    //cout << rot << endl << endl;
    //cout << tr << endl << endl;

    //drawKeypoints(src, keypointsD, src);
    //drawKeypoints(tgt, keypointsD2, tgt);

    //imshow("keypoints", src);
    //waitKey();
    //imshow("keypoints", tgt);
    //waitKey();
}
