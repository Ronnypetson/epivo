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


int extract_good_kp(const string base_img,
                    const int num_frames,
                    vector<vector<Point2f> > &key_points,
                    vector<string> &img_fns,
                    vector<Mat> &descs){
    assert(key_points.size() == 0);
    assert(img_fns.size() == 0);

    Mat src;
    //SurfFeatureDetector detector(minHessian);
    //SurfDescriptorExtractor extractor;
    //OrbFeatureDetector detector;
    //OrbDescriptorExtractor extractor;

    Ptr<ORB> orb = ORB::create(10000, 1.2f, 8, 15, 0, 2, ORB::FAST_SCORE);

    //Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);

    for(int i = 0; i < num_frames; i++){
        string src_fn;
        src_fn = base_img;
        stringstream ss0;
        ss0 << setw(6) << setfill('0') << i;
        src_fn += ss0.str() + ".png";

        src = imread(src_fn, IMREAD_GRAYSCALE);

        vector<KeyPoint> kp0;
        orb->detect(src, kp0, Mat());
        //detector.detect(src, kp0);

        vector<Point2f> pt0;
        cv::KeyPoint::convert(kp0, pt0);

        Mat desc0;
        //extractor.compute(src, kp0, desc0);
        orb->compute(src, kp0, desc0);

        key_points.push_back(pt0);
        img_fns.push_back(src_fn);
        descs.push_back(desc0);
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
            //Mat ess = findEssentialMat(_cpt0, _cpt1, cam, LMEDS, 0.99, 0.01, mask_ess);
            Mat ess = findEssentialMat(_cpt0, _cpt1, cam, RANSAC, 0.95, 0.01, mask_ess);

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


int really_robust_ass(const vector<pair<int, int> > window,
                      const int stride,
                      const int num_frames,
                      const vector<vector<Point2f> > &source_kp,
                      const vector<string> &img_fns,
                      const vector<Mat> &descs,
                      const Mat cam,
                      map<pair<int, int>, reproj> &reprojs){
    assert(stride > 0);

    vector<uchar> status;
    vector<float> err;

    Mat src, tgt;
    int i0, i1;
    string src_fn, tgt_fn;

    //FlannBasedMatcher matcher;
    BFMatcher matcher(NORM_HAMMING2, true); // , true

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
            vector<Point2f> _cpt0, _cpt1;

            Mat desc0, desc1;
            desc0 = descs[i0];
            desc1 = descs[i1];

            //cout << source_kp[i0].size() << " "
            //     << source_kp[i1].size() << " "
            //     << desc0.size() << " "
            //     << desc1.size() << endl << endl;

            //-- Step 3: Matching descriptor vectors using FLANN matcher
            vector<DMatch> matches;
            matcher.match(desc0, desc1, matches);

            //cout << "matches " << matches.size() << endl << endl;

            //-- Quick calculation of max and min distances between keypoints
            //double max_dist = 0;
            //double min_dist = 100;
            //for(int k = 0; k < matches.size(); k++){
            //    double dist = matches[k].distance;
            //    if( dist < min_dist ) min_dist = dist;
            //    if( dist > max_dist ) max_dist = dist;
            //}

            //cout << "dists " << min_dist << " " << max_dist << endl << endl;

            //vector<DMatch> good_matches;
            //for(int k = 0; k < matches.size(); k++){
            //    if(matches[k].distance <= max((min_dist + max_dist) / 3, 0.02)){
            //        good_matches.push_back(matches[k]);
            //    }
            //}
            vector<DMatch> good_matches = matches;

            //cout << "good matches " << good_matches.size() << endl << endl;

            pt0 = source_kp[i0];
            pt1 = source_kp[i1];

            vector<KeyPoint> kp0, kp1;

            for(int k = 0; k < pt0.size(); k++){
                kp0.push_back(KeyPoint(pt0[k], 1.0f));
            }

            for(int k = 0; k < pt1.size(); k++){
                kp1.push_back(KeyPoint(pt1[k], 1.0f));
            }

            //Mat drawn_matches;
            //drawMatches(src, kp0, tgt, kp1, good_matches, drawn_matches);
            //imshow("Matches", drawn_matches);
            //waitKey(0);

            int id0, id1;
            for(int k = 0; k < good_matches.size(); k++){
                id0 = good_matches[k].queryIdx;
                id1 = good_matches[k].trainIdx;

                //cout << "id " << id0 << " " << id1 << " " << pt0.size() << endl;

                _cpt0.push_back(pt0[id0]);
                _cpt1.push_back(pt1[id1]);
            }

            vector<uchar> mask_ess;
            vector<Point2f> cpt0, cpt1;
            Mat rot, tr;
            vector<uchar> rec_mask;
            MatrixXd erot(3, 3), etr(3, 1);
            vector<Point2f> fpt0, fpt1;
            if(_cpt0.size() >= 8){
                Mat ess = findEssentialMat(_cpt0, _cpt1, cam, LMEDS, 0.99, 0.1, mask_ess);
                //Mat ess = findEssentialMat(_cpt0, _cpt1, cam, RANSAC, 0.99, 0.1, mask_ess);

                for(int k = 0; k < mask_ess.size(); k++){
                    if((int)mask_ess[k] == 1){
                        cpt0.push_back(_cpt0[k]);
                        cpt1.push_back(_cpt1[k]);
                    }
                }

                //cout << "cpt0 " << cpt0.size() << endl << endl;

                // Initialize
                recoverPose(ess, cpt0, cpt1, cam, rot, tr, rec_mask);

                cv2eigen(rot, erot);
                cv2eigen(tr, etr);

                // if(erot.trace() < 3.0 * 0.9){
                //     erot = MatrixXd::Identity(3, 3);
                //     etr << 0.1, 0.1, -0.9;
                // }

                // if(etr.norm() < 1E-5){
                //     etr << 0.1, 0.1, -0.9;
                // }

                for(int k = 0; k < rec_mask.size(); k++){
                    //cout << (int)rec_mask[k] << " ";
                    if((int)rec_mask[k] == 255){
                        fpt0.push_back(cpt0[k]);
                        fpt1.push_back(cpt1[k]);
                    }
                }

                //cout << erot << endl << endl;
                //cout << etr << endl << endl;
                //cout << fpt0.size() << endl << endl;

            } else {
                erot = MatrixXd::Identity(3, 3);
                etr << 0.1, 0.1, -0.9;
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


int bundle_adjustment(map<pair<int, int>, reproj> &reprojs,
                      const vector<pair<int, int> > window,
                      const int stride,
                      const int num_frames,
                      const Mat cam,
                      vector<MatrixXd> &opt_T){
    assert(stride > 0);
    assert(opt_T.size() == 0);

    vector<bool> optimized;
    for(int i = 0; i < num_frames; i++){
        opt_T.push_back(MatrixXd::Identity(4, 4));
        optimized.push_back(false);
    }

    MatrixXd cam_;
    cv2eigen(cam, cam_);
    cam_ = cam_.inverse();

    int i0, i1, w0, w1, N;
    const int min_pt = 12;
    bool _exit = false;
    for(int i = 0; i < num_frames; i += stride){
        cout << i << endl;
        w0 = num_frames + 1;
        w1 = -1;
        // Check if the reprojs are available
        for(int j = 0; j < window.size(); j++){
            i0 = i + window[j].first;
            i1 = i + window[j].second;
            w0 = min(w0, min(i0, i1));
            w1 = max(w1, max(i0, i1));

            if(max(i0, i1) >= num_frames){
                _exit = true;
                break;
            }

            while(reprojs.find(make_pair(i0, i1)) == reprojs.end()){
                this_thread::sleep_for(chrono::milliseconds(20));
            }
        }

        if(_exit){
            break;
        }

        // Run Levenberg-Marquardt
        vector<pair<int, int> > reps;
        vector<MatrixXd> pr, p_r;
        vector<double> wreps;
        for(int j = 0; j < window.size(); j++){
            i0 = i + window[j].first;
            i1 = i + window[j].second;
            reps.push_back(make_pair(window[j].first, window[j].second - 1));

            reproj r = reprojs[make_pair(i0, i1)];
            N = min(min_pt, (int)r.p0.size());
            if(N < min_pt){
                //cout << "Bad pts" << endl << endl;
                wreps.push_back(0.0);
                pr.push_back(MatrixXd::Ones(min_pt, 3));
                p_r.push_back(MatrixXd::Ones(min_pt, 3));
            } else {
                if(i1 - i0 == 1 && optimized[i0]){
                    //cout << "Freezing pose" << endl << endl;
                    //wreps.push_back(0.0);
                    wreps.push_back(1.0);
                } else {
                    wreps.push_back(1.0);
                }
                MatrixXd pr_(N, 3), p_r_(N, 3);
                for(int k = 0; k < N; k++){
                    pr_.row(k) << r.p0[k].x, r.p0[k].y, 1.0;
                    p_r_.row(k) << r.p1[k].x, r.p1[k].y, 1.0;

                    pr_.row(k) = cam_ * pr_.row(k).transpose();
                    p_r_.row(k) = cam_ * p_r_.row(k).transpose();
                }
                
                pr.push_back(pr_);
                p_r.push_back(p_r_);
            }
        }

        vector<MatrixXd> T0s;
        double scale = 1.0;
        if(optimized[w0]){
            scale = opt_T[w0].block<3, 1>(0, 3).norm();
        }
        for(int j = w0; j < w1; j++){
            MatrixXd T0_0 = MatrixXd::Identity(4, 4);
            if(!optimized[j]){
                reproj r = reprojs[make_pair(j, j + 1)];
                T0_0.block<3, 3>(0, 0) = r.R;
                T0_0.block<3, 1>(0, 3) = r.t;
            } else {
                //T0_0 = opt_T[j];
                T0_0.block<3, 3>(0, 0) = opt_T[j].block<3, 3>(0, 0);
                T0_0.block<3, 1>(0, 3) = opt_T[j].block<3, 1>(0, 3) / scale;
            }
            T0s.push_back(T0_0);
        }
        
        vector<MatrixXd> bT0s(T0s);

        double uncert;
        int nzeta = w1 - w0;
        LM_res lm_res;

        Levenberg_Marquardt(nzeta, 1e-8, reps, wreps, 1e-2, T0s, pr, p_r, lm_res);

        cout << lm_res.H_norm << endl
             << lm_res.r_norm << endl
             << lm_res.lambda << endl << endl;
        
        //uncert = lm_res.lambda;
        //uncert = lm_res.r_norm;
        // || lm_res.H_norm > 1e-2

        if(lm_res.r_norm > 1e-7){
           T0s = bT0s;
        }

        for(int j = w0; j < w1; j++){
            opt_T[j] = T0s[j - w0];
            opt_T[j].block<3, 1>(0, 3) *= scale;
            optimized[j] = true;
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

    const int num_frames = 200;
    const int stride = 1;
    vector<vector<Point2f> > key_points;
    vector<string> img_fns;

    vector<Mat> descs;
    thread kp_extractor(extract_kp,
                        base_img,
                        num_frames,
                        ref(key_points),
                        ref(img_fns));
    // thread kp_extractor(extract_good_kp,
    //                     base_img,
    //                     num_frames,
    //                     ref(key_points),
    //                     ref(img_fns),
    //                     ref(descs));
    
    vector<pair<int, int> > window;
    int ws = 20;
    for(int i = 0; i < ws - 1; i++){
        window.push_back(make_pair(i, i + 1));
        //if(i > 0){
        //    window.push_back(make_pair(0, i + 1));
        //}
        if(i < ws - 2){
            window.push_back(make_pair(i, i + 2));
        }
    }

    map<pair<int, int>, reproj> reprojs;

    thread match_kp(robust_ass, window, stride, num_frames,
                    ref(key_points), ref(img_fns), cam, ref(reprojs));
    //thread match_kp(really_robust_ass, window, stride, num_frames,
    //                ref(key_points), ref(img_fns), ref(descs), cam, ref(reprojs));
    
    vector<MatrixXd> opt_T;
    thread ba_lm(bundle_adjustment, ref(reprojs), window,
                 ws - 1, num_frames, cam, ref(opt_T));

    kp_extractor.join();
    match_kp.join();
    ba_lm.join();

    cout << key_points.size() << endl << endl;
    cout << key_points[0].size() << endl << endl;
    cout << reprojs.size() << endl << endl;
    //cout << opt_T[95] << endl << endl;

    ofstream poses_ba;
    poses_ba.open("kitti.T");

    MatrixXd acc_T = MatrixXd::Identity(4, 4);
    double scale = 1.0;
    double tnorm = 1.0;
    for(int i = 0; i < opt_T.size(); i++){
        if(i % (ws - 1) == 0 || true){
            MatrixXd T = MatrixXd::Identity(4, 4);
            MatrixXd pT = MatrixXd::Identity(4, 4);

            MatrixXd _pT = poses.row(i);
            MatrixXd _T = poses.row(i + 1);

            _pT.resize(4, 3);
            _T.resize(4, 3);

            T.block<3, 4>(0, 0) = _T.transpose();
            pT.block<3, 4>(0, 0) = _pT.transpose();

            MatrixXd dT = pT.inverse() * T;

            scale = dT.block<3, 1>(0, 3).norm();
            tnorm = opt_T[i].block<3, 1>(0, 3).norm();
        }

        poses_ba << acc_T << "\n\n";

        if(tnorm != 0.0){
            opt_T[i].block<3, 1>(0, 3) *= scale / tnorm;
        }

        acc_T = acc_T * opt_T[i].inverse();
        //cout << opt_T[i] << endl << endl;

        cout << i << " " << scale / tnorm << endl << endl;
    }

    poses_ba.close();

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

        //double uncert;
        //uncert = Levenberg_Marquardt(1, 1e-8, reps, 1e-2, T0s, pr, p_r);
        //cout << uncert << endl << endl;
        //if(uncert > 1e-6){
        //   T0s = bT0s;
        //}

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