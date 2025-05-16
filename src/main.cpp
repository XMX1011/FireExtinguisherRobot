#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// #include "thermoCam.h"
// #include "visionModule.h"

using namespace std;
using namespace cv;
int main(char *argv[], int argc)
{
    // 读取红外相机获取的温度图片
    Mat image = imread("E:/Intern/FireExtinguisherRobot/testImage/01.JPG");
    cout << image.rows << " " << image.cols << endl;
    imshow("image", image);
    waitKey(0);
    
    return 0;
}