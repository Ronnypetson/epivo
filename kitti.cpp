#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <string>
//#include <opencv2/features2d.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

//#include "jac_Rt_gen_.cpp"

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

int main(){
    Mat src = imread("/home/ronnypetson/dataset/sequences/00/image_0/000010.png", IMREAD_GRAYSCALE);
    Mat tgt = imread("/home/ronnypetson/dataset/sequences/00/image_0/000011.png", IMREAD_GRAYSCALE);
    
    cout << " Image size :" << src.rows << " " << src.cols << "\n";
    cout << " Image size :" << tgt.rows << " " << tgt.cols << "\n";
    
    vector<KeyPoint> kp0, kp1, kp_;
    Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
    vector<Mat> descriptor;

    detector->detect(src, kp0, Mat());
    detector->detect(tgt, kp1, Mat());

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
    
    vector<Point2f> cpt0, cpt1_;
    for(int i = 0; i < status.size(); i++){
        if((int)status[i] == 1){
            cpt0.push_back(pt0[i]);
            cpt1_.push_back(pt1_[i]);
        }
    }

    cout << status.size() << endl;
    cout << cpt0.size() << " " << cpt1_.size() << endl;

    Mat cam = (Mat_<float>(3,3) << 718.8560, 0.0, 607.1928,
                                    0.0, 718.8560, 185.2157,
                                    0.0, 0.0,      1.0);
    MatrixXd cam_(3, 3);
    cam_ << 718.8560, 0.0,      607.1928,
            0.0,      718.8560, 185.2157,
            0.0,      0.0,      1.0;
    cam_ = cam_.inverse();

    MatrixXd poses = load_csv<MatrixXd>("/home/ronnypetson/dataset/poses/00.txt");

    MatrixXd R(3, 3), t(3, 1);
    MatrixXd T = MatrixXd::Identity(4, 4);
    MatrixXd pT = MatrixXd::Identity(4, 4);

    MatrixXd _pT = poses.row(10);
    MatrixXd _T = poses.row(11);

    _pT.resize(4, 3);
    _T.resize(4, 3);

    T.block<3, 4>(0, 0) = _T.transpose();
    pT.block<3, 4>(0, 0) = _pT.transpose();

    MatrixXd dT = pT.inverse() * T;
    dT = dT.inverse();
    R = dT.block<3, 3>(0, 0);
    t = dT.block<3, 1>(0, 3);

    vector<MatrixXd> X;
    MatrixXd p_(2, 3), p(3, 1), x_;
    MatrixXd A, B, cp(3, 1), cp_(3, 1);
    double d;
    for(int i = 0; i < cpt0.size(); i++){
        cp_ << cpt1_[i].x, cpt1_[i].y, 1.0;
        cp << cpt0[i].x, cpt0[i].y, 1.0;
        cp_ = cam_ * cp_;
        cp = cam_ * cp;

        p_ << 1.0, 0.0, -cp_(0, 0), 0.0, 1.0, -cp_(1, 0);
        A = p_ * t;
        B = p_ * R * cp;
        if(B.norm() > 1E-2){
            d = A.norm() / B.norm();
            x_ = d * cp;
            X.push_back(x_);
        }
    }

    ofstream pt_cloud;
    pt_cloud.open("pts.cld");

    for(int i = 0; i < X.size(); i++){
        pt_cloud << X[i].transpose() << "\n\n";
    }

    pt_cloud.close();

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
