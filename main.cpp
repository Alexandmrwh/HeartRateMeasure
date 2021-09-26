//
//  main.cpp
//  HRMeasure
//
//  Created by WuH on 2017/6/26.
//  Copyright © 2017年 WuH. All rights reserved.
//
#include "opencv2/core/core_c.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <cmath>
#include <time.h>
#include <chrono>
#include <pthread.h>
#include <armadillo>
#include <itpp/signal/fastica.h>
#include <stdlib.h>
#include "matplotlibcpp.h"

using namespace dlib;
using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace arma;
using namespace itpp;
namespace plt = matplotlibcpp;

struct Frame {
public:
    cv::Mat frame;
    double time;
};

struct ROI {
public:
    cv::Mat roi;
    double time;
};

#define MAXT 201
milliseconds start_time;
bool isCalculating;
bool isDetecting;
pthread_t cal_thread;
pthread_t roi_thread;
std::queue<Frame> frames;

CascadeClassifier faces_cascade;



struct RGB_color
{
public:
    double r, g, b;
    RGB_color() {}
    RGB_color(double _r,double _g, double _b):r(_r), g(_g), b(_b) {}
    RGB_color(const RGB_color & A) {r = A.r, g = A.g, b = A.b;}
};

class Signal
{
private:
    RGB_color sig[MAXT];
    double times[MAXT];
    int nowT, headT;
    bool flag;
public:
    Signal() {nowT = headT = 0; flag = 0;}
    /* Insert avg. color of ROI */
    void insert(const RGB_color & a, const double & time)
    {
        sig[nowT] = a;
        times[nowT] = time;
        nowT = (nowT + 1) % MAXT;
        if(nowT == headT) headT = (headT + 1) % MAXT, flag = 1;
    }
    bool full() {return flag;}
    /* Calculate BVP */
    double calc_BVP()
    {
        /* filter */
        /* undo */
        /* SOBI */
        int nows = (nowT - headT + MAXT) % MAXT;
        if(nows < 100) return 0.0;
        double sel = -10000.0;
        int sel_i = 0;
        arma::mat tmp;
        rowvec even_times = linspace<rowvec>(times[headT], times[nowT - 1], nows);
        rowvec timesv = zeros<rowvec>(nows);
        for(int i = headT, j = 0; i != nowT; i = (i + 1) % MAXT, j++) timesv(j) = times[i];
        tmp.set_size(3, nows);
        for(int i = headT, j = 0; i != nowT; i = (i + 1) % MAXT, j++)
        {
            tmp(0, j) = sig[i].r;
            tmp(1, j) = sig[i].g;
            tmp(2, j) = sig[i].b;
        }
        
        
        std::vector<double> rgbx(nows), rgby(nows);
        for (int i = 0; i < nows; i++) {
            rgbx.at(i) = i;
            rgby.at(i) = tmp(0,i);
        }
        
        plt::plot(rgbx, rgby);
        plt::title("rgb red");
        plt::legend();
        plt::save("/Users/apple/Desktop/testpics/rgbred.png");
        
        for (int i = 0; i < nows; i++) {
            rgbx.at(i) = i;
            rgby.at(i) = tmp(1,i);
        }
        
        plt::figure();
        plt::plot(rgbx, rgby);
        plt::title("rgb green");
        plt::legend();
        plt::save("/Users/apple/Desktop/testpics/rgbgreen.png");
        
        for (int i = 0; i < nows; i++) {
            rgbx.at(i) = i;
            rgby.at(i) = tmp(2,i);
        }
        
        plt::figure();
        plt::plot(rgbx, rgby);
        plt::title("rgb blue");
        plt::legend();
        plt::save("/Users/apple/Desktop/testpics/rgbblue.png");

        
        for(int i = 0; i < 3; ++i)
        {
            double avg = mean(tmp.row(i)), std = stddev(tmp.row(i));
            tmp.row(i).transform([avg, std](double x) {return (x - avg) / (std + 0.0001);});
        }
        cout << "START SOBI!" << endl;
        arma::mat H;
        SOBI(tmp, 3, 20, H);
        cout << "END SOBI!" << endl;
        arma::mat sur = H * tmp;
        cx_mat raws;
        for(int i = 0; i < 3; ++i)
        {
            rowvec tmp3;
            interp1(timesv, sur.row(i), even_times, tmp3);
            sur.row(i) = tmp3;
            raws = fft(sur.row(i));
            double tmp2 = kurt(abs(raws));
            if(tmp2 > sel)
                sel = tmp2, sel_i = i;
        }
        rowvec sor = sur.row(sel_i);
        cout << "End Select" << endl;
        /* End of SOBI */
        /* Calculate BVP */
        raws = fft(sor);
        
        cout<<"&&&&&&&&&"<< abs(raws) <<endl;
        double ans = 0, BVP = 0;
        for(int i = 0; i < nows / 2 + 1; ++i)
        {
            if (abs(raws(i)) > ans && ((double)i) / (even_times[nows - 1] - even_times[0]) * 60 > 40
                        && ((double)i) / (even_times[nows - 1] - even_times[0]) * 60< 200)
                ans = abs(raws(i)), BVP = ((double)i) / (even_times[nows - 1] - even_times[0]) * 60;
        }
        

        cout << "BVP is: " << BVP << endl;
        return BVP;
        /* End of Calculate */
    }
    
    /* Helper Function */
    double sqr(double x)
    {
        return x * x;
    }
    double kurt(const rowvec & A)
    {
        int n = A.size();
        double B4 = 0.0, B2 = 0.0;
        for(int i = 0; i < n; ++i)
            B4 += sqr(sqr(A(i))), B2 += sqr(A(i));
        return B4 * n / (B2 * B2) - 3;
    }
    
    void stdcov(const arma::mat & X, int tau, arma::mat & C)
    {
        int N = X.n_cols, m = X.n_rows;
        arma::vec m1 = zeros<arma::vec>(m), m2 = zeros<arma::vec>(m);
        arma::mat R = X.cols(0, N - tau - 1) * X.cols(tau, N - 1).t() / (N - tau);
        for(int i = 0; i < m; ++i)
        {
            m1[i] = mean(X.row(i).cols(0, N - tau - 1));
            m2[i] = mean(X.row(i).cols(tau, N - 1));
        }
        C = R - m1 * m2.t();
        C = (C + C.t()) / 2;
    }
    void joint_diag(const arma::mat & A, double jthresh, cx_mat & V, cx_mat & D)
    {
        int m = A.n_rows, nm = A.n_cols;
        arma::mat b1 = zeros<arma::mat>(3, 3), b2 = zeros<arma::mat>(3, 3);
        b1 << 1 << 0 << 0 << endr << 0 << 1 << 1 << endr << 0 << 0 << 0 << endr;
        b2 << 0 << 0 << 0 << endr << 0 << 0 << 0 << endr << 0 << -1 << 1 << endr;
        cx_mat B = cx_mat(b1, b2);
        cx_mat Bt = B.t();
        cx_mat Ip = zeros<cx_mat>(1, nm);
        cx_mat Iq = zeros<cx_mat>(1, nm);
        cx_mat g = zeros<cx_mat>(3, m);
        cx_mat G = zeros<cx_mat>(2, 2);
        arma::vec ev = zeros<arma::vec>(3);
        arma::mat vcp = zeros<arma::mat>(3, 3);
        double c = 0;
        cx_double s = 0;
        V = eye<cx_mat>(m, m);
        D = zeros<cx_mat>(m, nm);
        D.set_real(A);
        for(int encore = 1; encore;)
        {
            encore = 0;
            for(int p = 0; p < m - 1; ++p)
                for(int q = p + 1; q < m; ++q)
                {
                    cx_rowvec t1 = zeros<cx_rowvec>(nm / m), t2 = zeros<cx_rowvec>(nm / m);
                    cx_rowvec t3 = zeros<cx_rowvec>(nm / m), t4 = zeros<cx_rowvec>(nm / m);
                    for(int i = p; i < nm; i += m) t1((i - p) / m) = D(p, i), t2((i - p) / m) = D(q, i);
                    for(int i = q; i < nm; i += m) t3((i - q) / m) = D(q, i), t4((i - q) / m) = D(p, i);
                    g = join_vert(t1 - t3, join_vert(t4, t2));
                    eig_sym(ev, vcp, real((B * (g * g.t())) * B.t()));
                    arma::vec angles = vcp.col(2);
                    if(angles[0] < 0)
                        angles = angles * (-1);
                    c = sqrt(0.5 + angles[0] / 2);
                    s = cx_double(angles[1], -angles[2]) * 0.5 / c;
                    if(abs(s) > jthresh)
                    {
                        encore = 1;
                        G << c << -conj(s) << endr << s << c << endr;
                        cx_mat tmp = join_horiz(V.col(p), V.col(q)) * G;
                        V.col(p) = tmp.col(0), V.col(q) = tmp.col(1);
                        tmp = G.t() * join_vert(D.row(p), D.row(q));
                        D.row(p) = tmp.row(0), D.row(q) = tmp.row(1);
                        for(int ip = p, iq = q; ip < nm && iq < nm; ip += m, iq += m)
                        {
                            cx_colvec dip = D.col(ip), diq = D.col(iq);
                            D.col(ip) = (dip * c) + (diq * s); 
                            D.col(iq) = (diq * c) - (dip * conj(s));
                        } 
                    }
                }
        }
    }
    void SOBI(const arma::mat & A, int n, int num_tau, arma::mat & H)
    {
        int N = A.n_cols, m = A.n_rows;
        double tiny = 1e-8;
        arma::mat Rx, tmp;
        stdcov(A, 0, Rx);
        arma::mat uu, vv;
        arma::vec dd;
        svd(uu, dd, vv, Rx);
        arma::mat d = diagmat(dd);
        arma::mat Q = sqrtmat_sympd(pinv(d)) * uu.t();
        arma::mat z = Q * A;
        arma::mat Rz = zeros<arma::mat>(n, num_tau * n);
        
        for(int i = 1; i <= num_tau; ++i)
        {
            stdcov(z, i - 1, tmp);
            Rz.cols((i - 1) * n, i * n - 1) = tmp.cols(0, n - 1);
        }
        cx_mat v, d2;
        joint_diag(Rz, tiny, v, d2);
        H = real(v.t()) * Q;
    }
}HR;



Scalar roiMean(cv::Mat roi){
    Scalar avg;
    Scalar std;
    cv::meanStdDev(roi, avg, std);
    return avg;
}
cv::Mat getROI(cv::Mat img, Rect FOI) {
    

    Rect ROI;
    ROI.x = FOI.x+FOI.width*0.5-FOI.width*0.25/2;
    ROI.y = FOI.y+FOI.height*0.18-FOI.height*0.15/2;
    ROI.width = FOI.width*0.25;
    ROI.height = FOI.height*0.15;
    
    Point p1(ROI.x, ROI.y);
    Point p2(ROI.x+ROI.width, ROI.y+ROI.height);
    cv::rectangle(img, p1, p2, Scalar(255, 0, 0));
    
    cv::Mat ROIm = img(ROI);
    
    return ROIm;
}

void saveROI(ROI roi_tmp){
    
    double r, g, b;
    Scalar mean = roiMean(roi_tmp.roi);
    r = mean[2];
    g = mean[1];
    b = mean[0];
    HR.insert(RGB_color(r, g, b), roi_tmp.time);
}

void * calculate(void * arg) {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    while(true) {
        pthread_testcancel();
        cout << "Heart Rate" << HR.calc_BVP() << endl;
//        sleep(2);
    }
    return NULL;
}

void changeState() {
    isCalculating = !isCalculating;
}

void keyHandler(VideoCapture &cap) {
    char key = (char) cv::waitKey(10);
    if( key == 27 ) {
        pthread_cancel(roi_thread);
        if (isCalculating) {
            pthread_cancel(cal_thread);
        }
        cv::destroyAllWindows();
        cap.release();
    }
    if( key == 32 ) {
        changeState();
        if(isCalculating == true) {
            int succ = pthread_create(&cal_thread, NULL, calculate, NULL);
            if(succ == 0) {
                printf("start calculating in new thread\n");
            }
        }
        else {
            printf("stop calculating\n");
            pthread_cancel(cal_thread);
        }
    }
}

Rect getfacerect(cv::Mat frame){

    cv::Mat img = frame;
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);
    
    std::vector<Rect> faces;
    faces_cascade.detectMultiScale(img_gray,faces,1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
    
    Rect FOI = Rect(1,1,2,2);
    
    if(faces.size() > 1) {
        int maxsize = faces[0].area();
        
        FOI = faces[0];
        for ( size_t i = 1; i < faces.size(); ++i) {
            int tmp = faces[i].area();
            if( tmp > maxsize ) {
                maxsize = tmp;
                FOI = faces[i];
            }
        }
    }
    else if(faces.size() == 1) {
        FOI = faces[0];
    }
    
    return FOI;
}


void gettenface(Frame tenframe[]) {
    Rect facearea;
    int hasface = 0;
    
    for(int i = 0; i < 10; i++) {
        
        cv::Mat face = tenframe[i].frame;
        facearea = getfacerect(face);
        
        if(facearea != Rect(1,1,2,2)) {
            hasface = i;
            break;
        }
    }
    ROI roi_tmp;
    for(int i = hasface; i < 10; i++) {
        roi_tmp.time = tenframe[i].time;
        roi_tmp.roi = getROI(tenframe[i].frame, facearea);
        saveROI(roi_tmp);
        
        Point p1(facearea.x, facearea.y);
        Point p2(facearea.x+facearea.width, facearea.y+facearea.height);
        cv::rectangle(roi_tmp.roi, p1, p2, Scalar(255, 0, 0));
        
        char strchar[10];
        gcvt(roi_tmp.time, 15, strchar);
        string str = strchar;
        
        imwrite("/Users/apple/Library/Mobile\ Documents/com~apple~CloudDocs/Desktop/HRMeasure/roi"+str+".jpg", roi_tmp.roi);
    }
}
// In this function, take out the first frame in the queue and process it.
// Get the location of the top of eyebrow to calculate
// Call ROI calculate function in the end
ROI getRoiWithDlib(Frame frameparam) {
    
    cout << "getRoiWithDlib" << endl;
    
    dlib::frontal_face_detector face_detector = get_frontal_face_detector();
    
    shape_predictor sp;
    
    deserialize("/Users/apple/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/ceca/projects/HRMeasure/shape_predictors/shape_predictor_68_face_landmarks.dat") >> sp;
    
    
    Frame frametmp = frameparam;

    cv::Mat frame = frametmp.frame;
    array2d<dlib::rgb_pixel> img;
    assign_image(img, dlib::cv_image<rgb_pixel>(frame));
    
    //pyramid_up(img);
    
    image_window win;
    win.clear_overlay();
    win.set_image(img);
    
    cout << "test5" <<endl;
    
    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    std::vector<dlib::rectangle> dets = face_detector(img);
    
    cout << "test6" <<endl;
    
    // Choose the largest face to process
    unsigned long facenum = dets.size();
    
    cout << "face num" << facenum << endl;
    
    dlib::rectangle face;
    if(facenum > 1) {
        unsigned long maxsize = dets[0].area();
        
        face = dets[0];
        
        for(unsigned long j = 1; j < facenum; ++j) {
            unsigned long tmp = dets[j].area();
            if(tmp > maxsize) {
                maxsize = tmp;
                face = dets[j];
            }
         }
    }
    else if(facenum == 1) {
        face = dets[0];
    }
    

    full_object_detection shape = sp(img, face);

    
    dlib::point rightpoint = shape.part(20);
    dlib::point leftpoint = shape.part(23);
    Rect ROI;
    
    ROI.x = rightpoint.x();
    ROI.width = leftpoint.x() - rightpoint.x();

    if (rightpoint.y() <= leftpoint.y()) {
        ROI.y = 0.66 *face.top() + 0.33 * rightpoint.y();
        ROI.height = ROI.y - rightpoint.y();
    }
    else {
        ROI.y = 0.66 *face.top() + 0.33 * leftpoint.y();
        ROI.height = ROI.y - leftpoint.y();
    }
    
    
    struct ROI ROItmp;
    ROItmp.time = frametmp.time;
    ROItmp.roi = frame(ROI);

    // For debug
    Point p1(ROI.x, ROI.y);
    Point p2(ROI.x+ROI.width, ROI.y+ROI.height);
    cv::rectangle(frame, p1, p2, Scalar(255, 0, 0));
    imshow("frame2", frame);

    char strchar[10];
    gcvt(frametmp.time, 15, strchar);
    string str = strchar;
    
    imwrite("/Users/apple/Desktop/testpics/rois/roi"+str+".jpg", ROItmp.roi);
    // For debug

    return ROItmp;
}
void * processThreadDlib(void * arg) {
    while (frames.size() > 0) {
        cout << "frame num" << frames.size() << endl;
        Frame frame = frames.front();
        frames.pop();
        
        ROI roi_tmp;
        roi_tmp = getRoiWithDlib(frame);
        saveROI(roi_tmp);
    }
    return NULL;
}
void * processThreadTenFrames(void * arg) {
    while (frames.size() > 0) {
        //cout << "frame num" << frames.size() << endl;
        if(frames.size() > 10) {
            Frame tenframe[10];
            Frame tenframeout[10];
            for(int i = 0; i < 10; ++i) {
                tenframe[i] = frames.front();
                frames.pop();
            }
            gettenface(tenframe);
        }
    }
    return NULL;
}
int saveFrame() {
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;
    
    while(cap.isOpened()) {
        Frame frame_in;
        cap >> frame_in.frame;
        
        milliseconds ms = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        frame_in.time = ((double)(ms.count() - start_time.count())) / 1000;

        
        frames.push(frame_in);
        imshow("frame", frame_in.frame);
        keyHandler(cap);
        if (isDetecting == false) {
            isDetecting = true;
            int out_thread;
            //out_thread = pthread_create(&roi_thread, NULL, processThreadTenFrames, NULL);
            out_thread = pthread_create(&roi_thread, NULL, processThreadDlib, NULL);
            if (out_thread == 0) {
                printf("start show thread\n");
            }
        }
    }
    return 1;
}

int main(int argc, const char * argv[]) {
    start_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    isCalculating = false;
    isDetecting = false;
    
    faces_cascade.load("/Users/apple/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/ceca/projects/HRMeasure/Haarcascades/haarcascade_frontalface_alt2.xml");
    
    if(faces_cascade.empty()) {
        printf("fail loading cascade\n");
    }

    int openVideo = saveFrame();
    if( openVideo < 0 ) {
        printf("error open camera\n");
    }
    return 0;
}




//cv::Mat getFace(cv::Mat frame_in) {
//
//
//    cv::Mat img = frame_in;
//    cv::Mat img_gray;
//    cv::cvtColor(img, img_gray, COLOR_BGR2GRAY);
//    equalizeHist(img_gray, img_gray);
//
//    vector<Rect> faces;
//    faces_cascade.detectMultiScale(img_gray,faces,1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
//
//    Rect FOI;
//
//    if(faces.size() > 1) {
//        int maxsize = faces[0].area();
//
//        FOI = faces[0];
//        for ( size_t i = 1; i < faces.size(); ++i) {
//            int tmp = faces[i].area();
//            if( tmp > maxsize ) {
//                maxsize = tmp;
//                FOI = faces[i];
//            }
//        }
//    }
//    else if(faces.size() == 1) {
//        FOI = faces[0];
//    }
//
//    cv::Mat ROI = getROI(img, FOI);
//    saveROI(ROI);
//
//    Point p1(FOI.x, FOI.y);
//    Point p2(FOI.x+FOI.width, FOI.y+FOI.height);
//    rectangle(img, p1, p2, Scalar(255, 0, 0));
//
//    return img;
//}
