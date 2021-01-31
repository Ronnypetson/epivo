#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

//#include "jac_Rt_gen_.cpp"

using namespace std;
using namespace cv;

int main(){
    Mat src = imread("/home/ronnypetson/dataset/sequences/00/image_0/000000.png", IMREAD_GRAYSCALE);
    Mat tgt = imread("/home/ronnypetson/dataset/sequences/00/image_0/000000.png", IMREAD_GRAYSCALE);
    
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

    float _cam[9] = {718.8560, 0.0, 607.1928,
                      0.0, 718.8560, 185.2157,
                      0.0, 0.0,      1.0};
    //Mat cam = new Mat(3, 3, CV_32FC1);
    //cam.put(0, 0, _cam);
    Mat cam = (Mat_<float>(3,3) << 718.8560, 0.0, 607.1928,
                                    0.0, 718.8560, 185.2157,
                                    0.0, 0.0,      1.0);
    
    Mat ess = cv::findEssentialMat(cpt0, cpt1_, cam);

    Mat rot, tr;
    recoverPose(ess, cpt0, cpt1_, cam, rot, tr);

    cout << rot << endl << endl;
    cout << tr << endl << endl;

    //drawKeypoints(src, keypointsD, src);
    //drawKeypoints(tgt, keypointsD2, tgt);

    //imshow("keypoints", src);
    //waitKey();
    //imshow("keypoints", tgt);
    //waitKey();
}
