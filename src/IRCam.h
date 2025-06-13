// IRCam.h
#pragma once

#include "utils.h"

class IRCam
{
public:
    class IRCamImpl; // 声明嵌套实现类

    IRCam();
    ~IRCam();

    bool openCamera();
    bool closeCamera();
    bool isCameraOpened();
    bool readVideo(cv::Mat &frame);
    bool converetToTemperature(cv::Mat &frame, cv::Mat &temp_matrix);

private:
    IRCamImpl *pImpl;
};