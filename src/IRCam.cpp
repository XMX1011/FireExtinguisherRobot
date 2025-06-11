// src/IRCam.cpp
#include "IRCam.h"
#include <opencv2/opencv.hpp>
#include <iostream>

/**
 * @brief   IRCam 类的实现文件
 *  需要注意的是需要等待具体sdk
 *  现在里面的代码毫无意义
 * 
 */


// 将 IRCamImpl 定义为 IRCam 的嵌套类
class IRCam::IRCamImpl
{
public:
    cv::VideoCapture cap;
    bool isOpenedFlag = false;

    // 打开相机（可以是设备索引号或RTSP地址）
    bool openCamera(const std::string &source = "")
    {
        int deviceIndex = 0;
        if (!source.empty())
        {
            try
            {
                deviceIndex = std::stoi(source); // 如果是数字字符串，尝试转为整数作为设备索引
            }
            catch (...)
            {
                // 否则当作 RTSP 地址处理
                cap.open(source);
                if (!cap.isOpened())
                {
                    std::cerr << "Failed to open RTSP stream: " << source << std::endl;
                    return false;
                }
                isOpenedFlag = true;
                return true;
            }
        }

        cap.open(deviceIndex);
        if (!cap.isOpened())
        {
            std::cerr << "Failed to open camera at index: " << deviceIndex << std::endl;
            return false;
        }
        isOpenedFlag = true;
        return true;
    }

    // 关闭相机
    bool closeCamera()
    {
        if (cap.isOpened())
        {
            cap.release();
        }
        isOpenedFlag = false;
        return true;
    }

    // 是否已打开
    bool isCameraOpened() const
    {
        return isOpenedFlag;
    }

    // 读取视频帧
    bool readVideo(cv::Mat &frame)
    {
        if (!cap.isOpened())
        {
            std::cerr << "Camera not opened!" << std::endl;
            return false;
        }
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Failed to capture frame." << std::endl;
            return false;
        }
        return true;
    }
};

// 根据数据手册：图像格式为 Cb,Y,Cr,Y 或 Y,Cb,Y,Cr
// 我们需要将该格式转换为标准 BGR 以便后续处理
bool convertYCbYCrToBGR(const cv::Mat &ycbcrFrame, cv::Mat &bgrFrame)
{
    if (ycbcrFrame.channels() != 1 || ycbcrFrame.type() != CV_8UC1)
    {
        std::cerr << "Input frame must be single-channel 8-bit grayscale." << std::endl;
        return false;
    }

    // 检查尺寸必须是偶数宽度
    if (ycbcrFrame.cols % 2 != 0)
    {
        std::cerr << "Image width must be even for YCbYCr format conversion." << std::endl;
        return false;
    }

    cv::Mat interleaved;
    cv::cvtColor(ycbcrFrame, interleaved, cv::COLOR_GRAY2BGR); // 简化处理，实际应解析像素结构
    bgrFrame = interleaved.clone();
    return true;
}

// 温度矩阵转换函数（示例逻辑，需替换为真实映射逻辑）
// 测温精度：±3℃或±3%(取大值) @23℃±5℃，测温距离 5 米
// 测温范围：-20℃~+150℃，0~550℃，支持测温范围拓展定制
// 区域测温：支持任意区域测温，输出区域最大值，最小值及平均值
bool convertToTemperature(cv::Mat &frame, cv::Mat &temp_matrix, float min_temp = 0.0f, float max_temp = 550.0f)
{
    if (frame.empty())
    {
        std::cerr << "Input frame is empty." << std::endl;
        return false;
    }

    // 转换为灰度图
    cv::Mat grayFrame;
    if (frame.channels() == 3)
    {
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayFrame = frame.clone();
    }

    // 归一化并映射到温度范围
    grayFrame.convertTo(temp_matrix, CV_32FC1);
    float scale = (max_temp - min_temp) / 255.0f;
    temp_matrix = temp_matrix.mul(scale) + min_temp;

    return true;
}

// 构造函数和析构函数实现
IRCam::IRCam() : pImpl(new IRCamImpl()) {}

IRCam::~IRCam()
{
    delete pImpl;
}

bool IRCam::openCamera()
{
    return pImpl->openCamera(); // 可传入参数如 RTSP 地址
}

bool IRCam::closeCamera()
{
    return pImpl->closeCamera();
}

bool IRCam::isCameraOpened()
{
    return pImpl->isCameraOpened();
}

bool IRCam::readVideo(cv::Mat &frame)
{
    return pImpl->readVideo(frame);
}

bool IRCam::converetToTemperature(cv::Mat &frame, cv::Mat &temp_matrix)
{
    return convertToTemperature(frame, temp_matrix);
}