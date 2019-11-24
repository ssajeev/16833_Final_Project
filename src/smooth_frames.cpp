// #include "temp_filt.h"

// int main() {
//     using namespace librealsense;

//     temporal_filter temp_filter.process;
// }
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "temp_filt.h"
using namespace cv;
using namespace std;
using namespace librealsense;

/* Feed in 16-bit GRAYSCALE images */
const string in_path = "/Users/PapaYaw/Documents/16-833/Final_Project/image_0/disp_maps/";
const string out_path = "/Users/PapaYaw/Documents/16-833/Final_Project/image_0/smoothed_maps/";


/* Play around with the alpha and delta to change the smoothing */

// valid range from 0.f - 1.f
// higher alpha values means less dependence on previous frame
const float alpha_val = 0.2f;

// valid range from 1 - 100
const float delta_val = 60; 

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2){
    // treat two empty mat as identical as well
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    // if dimensionality of two mat is not identical, these two mat is not identical
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
        return false;
    }
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    int nz = cv::countNonZero(diff);

    return nz==0;
}

int main(int argc, char* argv[])
{   
    unsigned int i = 0;

    temporal_filter temp_filter;

    // used to set up initial parameters 
    Mat frame = imread(in_path + "1" + ".png", CV_LOAD_IMAGE_ANYDEPTH);

    temp_filter.update_configuration(frame);
    temp_filter.on_set_alpha(alpha_val);
    temp_filter.on_set_delta(delta_val);


    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0); // no compression
        
    int img_idx = 1;

    while (!((frame = imread(in_path + to_string(img_idx) + ".png", CV_LOAD_IMAGE_ANYDEPTH)).empty())) {
        temp_filter.process_frame(frame);

        imwrite(out_path + to_string(img_idx) + ".png", frame, compression_params);

        cout << "Smoothed image #: " << img_idx << endl;

        img_idx++;
    }
}