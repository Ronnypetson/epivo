#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/core/eigen.hpp>
#include <thread>
#include <map>
#include <chrono>

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
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}


int extract_kp(const string base_img,
               const int num_frames,
               vector<vector<Point2f> > &key_points,
               vector<string> &img_fns){
    assert(key_points.size() == 0);
    assert(img_fns.size() == 0);

    Mat src;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    for(int i = 0; i < num_frames; i++){
        string src_fn;
        src_fn = base_img;
        stringstream ss0;
        ss0 << setw(6) << setfill('0') << i;
        src_fn += ss0.str() + ".png";

        src = imread(src_fn, IMREAD_GRAYSCALE);

        vector<KeyPoint> kp0;
        detector->detect(src, kp0, Mat());
        vector<Point2f> pt0;
        cv::KeyPoint::convert(kp0, pt0);

        key_points.push_back(pt0);
        img_fns.push_back(src_fn);
    }
}


struct reproj{
    vector<Point2f> p0, p1;
    MatrixXd R, t;
};


int robust_ass(const vector<pair<int, int> > window,
               const int stride,
               const int num_frames,
               const vector<vector<Point2f> > &source_kp,
               const vector<string> &img_fns,
               const Mat cam,
               map<pair<int, int>, reproj> &reprojs){
    assert(stride > 0);

    vector<uchar> status;
    vector<float> err;

    Mat src, tgt;
    int i0, i1;
    string src_fn, tgt_fn;

    for(int i = 0; i < num_frames; i += stride){
        for(int j = 0; j < window.size(); j++){
            i0 = i + window[j].first;
            i1 = i + window[j].second;

            if(reprojs.find(make_pair(i0, i1)) != reprojs.end()){
                continue;
            }

            if(max(i0, i1) >= num_frames){
                break;
            }

            while(source_kp.size() < max(i0, i1) + 1){
                this_thread::sleep_for(chrono::milliseconds(10));
            }

            src_fn = img_fns[i0];
            tgt_fn = img_fns[i1];

            src = imread(src_fn, IMREAD_GRAYSCALE);
            tgt = imread(tgt_fn, IMREAD_GRAYSCALE);

            vector<Point2f> pt0, pt1;
            pt0 = source_kp[i0];

            calcOpticalFlowPyrLK(src, tgt, pt0, pt1, status, err);

            vector<Point2f> _cpt0, _cpt1;
            for(int k = 0; k < status.size(); k++){
                if((int)status[k] == 1){
                    _cpt0.push_back(pt0[k]);
                    _cpt1.push_back(pt1[k]);
                }
            }

            vector<uchar> mask_ess;
            Mat ess = findEssentialMat(_cpt0, _cpt1, cam, LMEDS, 0.99, 0.01, mask_ess);

            vector<Point2f> cpt0, cpt1;
            for(int k = 0; k < mask_ess.size(); k++){
                if((int)mask_ess[k] == 1){
                    cpt0.push_back(_cpt0[k]);
                    cpt1.push_back(_cpt1[k]);
                }
            }

            // Initialize
            Mat rot, tr;
            vector<uchar> rec_mask;
            recoverPose(ess, cpt0, cpt1, cam, rot, tr, rec_mask);

            MatrixXd erot(3, 3), etr(3, 1);
            cv2eigen(rot, erot);
            cv2eigen(tr, etr);

            if(erot.trace() < 3.0 * 0.9){
                erot = MatrixXd::Identity(3, 3);
                etr << 0.1, 0.1, -0.9;
            }

            if(etr.norm() < 1E-5){
                etr << 0.1, 0.1, -0.9;
            }

            vector<Point2f> fpt0, fpt1;
            for(int k = 0; k < rec_mask.size(); k++){
                if((int)rec_mask[k] == 255){
                    fpt0.push_back(cpt0[k]);
                    fpt1.push_back(cpt1[k]);
                }
            }

            reproj rep;
            rep.p0 = fpt0;
            rep.p1 = fpt1;
            rep.R = erot;
            rep.t = etr;
            reprojs.insert(make_pair(make_pair(i0, i1), rep));
        }
    }
}


int main(){
    Mat cam = (Mat_<float>(3,3) << 718.8560, 0.0, 607.1928,
                                    0.0, 718.8560, 185.2157,
                                    0.0, 0.0,      1.0);
    MatrixXd cam_(3, 3);
    cam_ << 718.8560, 0.0,      607.1928,
            0.0,      718.8560, 185.2157,
            0.0,      0.0,      1.0;
    cam_ = cam_.inverse();
    MatrixXd poses = load_csv<MatrixXd>("/home/ronnypetson/dataset/poses/00.txt");
    vector<MatrixXd> X;
    vector<int> limits;
    string src_fn, tgt_fn;
    const string base_img = "/home/ronnypetson/dataset/sequences/00/image_0/";

    vector<MatrixXd> all_T, all_GT;
    MatrixXd cT = MatrixXd::Identity(4, 4);

    const int num_frames = 100;
    vector<vector<Point2f> > key_points;
    vector<string> img_fns;

    thread kp_extractor(extract_kp,
                        base_img,
                        num_frames,
                        ref(key_points),
                        ref(img_fns));
    
    vector<pair<int, int> > window;
    window.push_back(make_pair(0, 1));
    window.push_back(make_pair(1, 2));
    window.push_back(make_pair(0, 2));

    map<pair<int, int>, reproj> reprojs;

    thread match_kp(robust_ass, window, 1, num_frames,
                    ref(key_points), ref(img_fns), cam, ref(reprojs));

    kp_extractor.join();
    match_kp.join();

    cout << key_points.size() << endl << endl;
    cout << key_points[0].size() << endl << endl;
    cout << reprojs.size() << endl << endl;
    exit(0);

    for(int i = 0; i < num_frames; i++){
        cout << i << " ";
        src_fn = base_img;
        tgt_fn = base_img;
        stringstream ss0, ss1;
        ss0 << setw(6) << setfill('0') << i;
        ss1 << setw(6) << setfill('0') << i + 1;
        src_fn += ss0.str() + ".png";
        tgt_fn += ss1.str() + ".png";

        Mat src = imread(src_fn, IMREAD_GRAYSCALE);
        Mat tgt = imread(tgt_fn, IMREAD_GRAYSCALE);
        
        vector<KeyPoint> kp0, kp_; // kp1,
        Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
        vector<Mat> descriptor;

        detector->detect(src, kp0, Mat());

        vector<Point2f> pt0, pt1_;
        cv::KeyPoint::convert(kp0, pt0);

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
                                   cam,
                                   RANSAC,
                                   0.999,
                                   0.3,
                                   mask_ess);
        
        vector<Point2f> cpt0, cpt1_;
        for(int j = 0; j < mask_ess.size(); j++){
            if((int)mask_ess[j] == 1){
                cpt0.push_back(_cpt0[j]);
                cpt1_.push_back(_cpt1_[j]);
            }
        }

        //cout << status.size() << endl;
        //cout << cpt0.size() << " " << cpt1_.size() << endl;

        // Initialize
        Mat rot, tr;
        recoverPose(ess, cpt0, cpt1_, cam, rot, tr); // mask_ess

        MatrixXd erot(3, 3), etr(3, 1);
        cv2eigen(rot, erot);
        cv2eigen(tr, etr);
        //erot = MatrixXd::Identity(3, 3);
        //etr << 0.0, 0.0, -1.0;

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
        MatrixXd T0_0 = MatrixXd::Identity(4, 4);
        T0_0.block<3, 3>(0, 0) = erot;
        T0_0.block<3, 1>(0, 3) = etr;
        T0s.push_back(T0_0); // .inverse()
        vector<MatrixXd> bT0s(T0s);

        vector<MatrixXd> pr, p_r;

        // vector<MatrixXd> fcpt0, fcpt1;
        // for(int j = 0; j < cpt0.size(); j++){
        //     MatrixXd cp0(3, 1), cp1(3, 1);
        //     cp0 << cpt0[j].x, cpt0[j].y, 1.0;
        //     cp1 << cpt1_[j].x, cpt1_[j].y, 1.0;
        //     if((cp0 - cp1).norm() >= 5.0){
        //         cp0 = cam_ * cp0;
        //         cp1 = cam_ * cp1;
        //         fcpt0.push_back(cp0);
        //         fcpt1.push_back(cp1);
        //     }
        // }
        // int N = min(200, (int)fcpt0.size());
        // MatrixXd pr_(N, 3), p_r_(N, 3);
        // for(int j = 0; j < N; j++){
        //     pr_.row(j) = fcpt0[j].transpose();
        //     p_r_.row(j) = fcpt1[j].transpose();
        // }
        // assert((int)fcpt0.size() >= 15);

        int N = min(48, (int)cpt0.size());
        assert(N >= 48);
        MatrixXd pr_(N, 3), p_r_(N, 3);
        for(int j = 0; j < N; j++){
            pr_.row(j) << cpt0[j].x, cpt0[j].y, 1.0;
            p_r_.row(j) << cpt1_[j].x, cpt1_[j].y, 1.0;

            pr_.row(j) = cam_ * pr_.row(j).transpose();
            p_r_.row(j) = cam_ * p_r_.row(j).transpose();
        }
        
        pr.push_back(pr_);
        p_r.push_back(p_r_);

        double uncert;
        uncert = Levenberg_Marquardt(1, 1e-8, reps, 1e-2, T0s, pr, p_r);
        cout << uncert << endl << endl;
        if(uncert > 1e-6){
           T0s = bT0s;
        }

        MatrixXd R(3, 3), t(3, 1);
        MatrixXd T = MatrixXd::Identity(4, 4);
        MatrixXd pT = MatrixXd::Identity(4, 4);

        MatrixXd _pT = poses.row(i);
        MatrixXd _T = poses.row(i + 1);

        _pT.resize(4, 3);
        _T.resize(4, 3);

        T.block<3, 4>(0, 0) = _T.transpose();
        pT.block<3, 4>(0, 0) = _pT.transpose();
        all_GT.push_back(pT);

        // point dT
        MatrixXd dT = (pT.inverse() * T).inverse();

        double scale = dT.block<3, 1>(0, 3).norm();
        T0s[0].block<3, 1>(0, 3) /= T0s[0].block<3, 1>(0, 3).norm();
        dT.block<3, 1>(0, 3) = T0s[0].block<3, 1>(0, 3) * scale;
        dT.block<3, 3>(0, 0) = T0s[0].block<3, 3>(0, 0);

        MatrixXd pT_ = cT;

        all_T.push_back(cT);
        cT = cT * dT.inverse();

        // point dT
        //dT = dT.inverse();
        R = dT.block<3, 3>(0, 0);
        t = dT.block<3, 1>(0, 3);

        MatrixXd p_(2, 3), p(3, 1), x_;
        MatrixXd A, B, cp(3, 1), cp_(3, 1);
        double d;
        limits.push_back(X.size());
        for(int j = 0; j < cpt0.size(); j++){
            cp_ << cpt1_[j].x, cpt1_[j].y, 1.0;
            cp << cpt0[j].x, cpt0[j].y, 1.0;
            cp_ = cam_ * cp_;
            cp = cam_ * cp;

            p_ << 1.0, 0.0, -cp_(0, 0), 0.0, 1.0, -cp_(1, 0);
            A = p_ * t;
            B = p_ * R * cp;
            if(B.norm() > 1E-2){
                d = A.norm() / B.norm();
                x_ = pT_.block<3, 3>(0, 0) * (d * cp)
                    + pT_.block<3, 1>(0, 3); // debug
                X.push_back(x_);
            }
        }
    }

    ofstream pt_cloud, lims, poses_f, poses_gt;
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

    poses_f.open("kitti.T");
    for(int i = 0; i < all_T.size(); i++){
        poses_f << all_T[i] << "\n\n";
    }

    poses_f.close();

    poses_gt.open("kitti.GT");
    for(int i = 0; i < all_GT.size(); i++){
        poses_gt << all_GT[i] << "\n\n";
    }

    poses_gt.close();
}
