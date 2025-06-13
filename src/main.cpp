#include "vision_processing.h"
#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// 将相机参数封装为结构体
struct CameraParams
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 500.0, 0.0, 320.0,
                             0.0, 500.0, 240.0,
                             0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    float hfov_degrees = 60.0f;
    float vfov_degrees = 45.0f;
    float nozzle_azimuth_offset = 0.0f;
    float nozzle_pitch_offset = 0.0f;
};

// utils.h 中声明的辅助函数实现

/**
 * @brief 将像素坐标转换为近似的世界坐标
 *
 * @param pixel_coord 像素坐标
 * @param cam_matrix 相机内参矩阵
 * @param distance_to_plane 到平面的距离
 * @return 返回近似的世界坐标
 *
 * 此函数根据相机内参和像素坐标，计算出近似的世界坐标。如果相机内参为空或焦距为0，则直接返回像素坐标。
 */
cv::Point3f pixelToApproxWorld(const cv::Point2f &pixel_coord, const cv::Mat &cam_matrix, float distance_to_plane)
{
    if (cam_matrix.empty() || cam_matrix.at<double>(0, 0) == 0)
    {
        return cv::Point3f(pixel_coord.x, pixel_coord.y, 0.0f);
    }
    double fx = cam_matrix.at<double>(0, 0);
    double fy = cam_matrix.at<double>(1, 1);
    double cx = cam_matrix.at<double>(0, 2);
    double cy = cam_matrix.at<double>(1, 2);

    double X = (pixel_coord.x - cx) * distance_to_plane / fx;
    double Y = (pixel_coord.y - cy) * distance_to_plane / fy;
    return cv::Point3f(static_cast<float>(X), static_cast<float>(Y), distance_to_plane);
}

/**
 * @brief 算两点在真实世界中的距离
 *
 * @param p1 第一个点的世界坐标
 * @param p2 第二个点的世界坐标
 * @return 返回两点之间的距离，如果任一点的z坐标为0，则返回最大浮点数
 *
 * 此函数计算两个三维点在真实世界中的欧氏距离。如果任一点的z坐标为0，则表示该点无效，函数返回最大浮点数。
 */
float calculateRealWorldDistance(const cv::Point3f &p1, const cv::Point3f &p2)
{
    if (p1.z == 0.0f || p2.z == 0.0f)
        return FLT_MAX;
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

/**
 * @brief 加载相机参数
 *
 * @param filename 参数文件路径
 * @param cam_matrix_out 相机内参矩阵输出
 * @param dist_coeffs_out 相机畸变系数输出
 * @param hfov_out 水平视场角输出
 * @param vfov_out 垂直视场角输出
 * @param nozzle_az_offset_out 喷嘴偏移方位角输出
 * @param nozzle_p_offset_out 喷嘴偏移俯仰角输出
 * @return 如果成功加载参数文件则返回true，否则返回false
 *
 * 此函数从指定的文件中加载相机参数，包括内参矩阵、畸变系数和视场角等。如果文件打开失败或某些参数缺失，则函数会输出错误或警告信息，并使用默认参数。
 */
bool loadCameraParameters(const std::string &filename,
                          cv::Mat &cam_matrix_out, cv::Mat &dist_coeffs_out,
                          float &hfov_out, float &vfov_out,
                          float &nozzle_az_offset_out, float &nozzle_p_offset_out)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open parameters file: " << filename << std::endl;
        std::cerr << "Using default/hardcoded parameters." << std::endl;
        return false;
    }
    fs["camera_matrix"] >> cam_matrix_out;
    fs["distortion_coefficients"] >> dist_coeffs_out;

    if (fs["HFOV_degrees"].isReal())
        fs["HFOV_degrees"] >> hfov_out;
    else
        std::cout << "Warning: HFOV_degrees not found in " << filename << std::endl;

    if (fs["VFOV_degrees"].isReal())
        fs["VFOV_degrees"] >> vfov_out;
    else
        std::cout << "Warning: VFOV_degrees not found in " << filename << std::endl;

    if (fs["nozzle_offset_azimuth_degrees"].isReal())
        fs["nozzle_offset_azimuth_degrees"] >> nozzle_az_offset_out;
    else
        std::cout << "Warning: nozzle_offset_azimuth_degrees not found in " << filename << std::endl;

    if (fs["nozzle_offset_pitch_degrees"].isReal())
        fs["nozzle_offset_pitch_degrees"] >> nozzle_p_offset_out;
    else
        std::cout << "Warning: nozzle_offset_pitch_degrees not found in " << filename << std::endl;

    fs.release();
    std::cout << "Parameters loaded from " << filename << std::endl;
    return true;
}

/**
 * @brief 将热成像图像转换为温度矩阵
 *
 * @param image_path 图像文件路径
 * @param temp_matrix 输出的温度矩阵
 * @param min_temp 图像中的最低温度
 * @param max_temp 图像中的最高温度
 * @param target_size 目标图像分辨率，默认为384x288
 * @return 如果成功加载图像并转换为温度矩阵则返回true，否则返回false
 *
 * 此函数读取灰度图像，并将其转换为指定分辨率的温度矩阵。转换过程中，会根据给定的温度范围将灰度值映射到温度值。
 */
bool getThermalImageAsTemperatureMatrix(const std::string &image_path,
                                        cv::Mat &temp_matrix,
                                        float min_temp,
                                        float max_temp,
                                        const cv::Size &target_size = cv::Size(384, 288))
{
    // 1. 读取灰度图
    cv::Mat gray_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (gray_image.empty())
    {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return false;
    }

    // 2. 缩放图像到指定分辨率
    cv::Mat resized_image;
    cv::resize(gray_image, resized_image, target_size, 0, 0, cv::INTER_LINEAR);

    // 3. 预先计算缩放因子
    float scale = (max_temp - min_temp) / 255.0f;
    resized_image.convertTo(temp_matrix, CV_32FC1, scale, min_temp);

    return true;
}

int main()
{
    cv::Mat display_image;
    std::string thermal_image_path = "../testImage/02.JPG"; // 设置你的图像路径
    cv::Mat temperature_matrix;

    CameraParams params;
    std::string params_file = "../config/params.xml"; // 或者其他配置文件名
    if (!loadCameraParameters(params_file, params.camera_matrix, params.dist_coeffs,
                              params.hfov_degrees, params.vfov_degrees,
                              params.nozzle_azimuth_offset, params.nozzle_pitch_offset))
    {
        std::cout << "Using hardcoded default parameters due to load failure." << std::endl;
    }

    // 打印加载或使用的参数
    std::cout << "Using HFOV: " << params.hfov_degrees << ", VFOV: " << params.vfov_degrees << std::endl;
    std::cout << "Using Nozzle Offset Az: " << params.nozzle_azimuth_offset << ", Pitch: " << params.nozzle_pitch_offset << std::endl;

    std::cout << "Vision Processing for Fire Suppression Started." << std::endl;
    std::cout << "Press 'q' or ESC to exit." << std::endl;

    // 模拟云台当前角度 (实际应用中从云台反馈获取)
    float current_gimbal_azimuth = 0.0f;
    float current_gimbal_pitch = 0.0f;

    while (true)
    {
        if (!getThermalImageAsTemperatureMatrix(thermal_image_path, temperature_matrix, 20.0f, 500.0f))
        {
            std::cerr << "Error: Could not generate temperature matrix from image." << std::endl;
            break;
        }

        int frame_rows = temperature_matrix.rows;
        int frame_cols = temperature_matrix.cols;

        std::vector<HotSpot> hot_spots = detectAndFilterHotspots(temperature_matrix, params.camera_matrix, ASSUMED_DISTANCE_TO_FIRE_PLANE_METERS);
        std::vector<SprayTarget> spray_targets = determineSprayTargets(hot_spots, MAX_GROUPING_DISTANCE_METERS);

        cv::Mat normalized_temp;
        cv::normalize(temperature_matrix, normalized_temp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(normalized_temp, display_image, cv::COLORMAP_JET);
        visualizeResults(display_image, hot_spots, spray_targets);

        if (!spray_targets.empty())
        {
            const SprayTarget &primary_target = spray_targets[0]; // 取最严重的目标
            std::cout << "Primary Target Pixel: (" << primary_target.final_pixel_aim_point.x
                      << ", " << primary_target.final_pixel_aim_point.y << ")" << std::endl;

            CloudGimbalAngles desired_angles = calculateGimbalAngles(
                primary_target.final_pixel_aim_point,
                frame_cols, frame_rows,
                params.hfov_degrees, params.vfov_degrees,
                current_gimbal_azimuth, current_gimbal_pitch,
                params.nozzle_azimuth_offset, params.nozzle_pitch_offset);

            std::cout << "Calculated Gimbal Command -> Target Azimuth: " << desired_angles.target_azimuth_degrees
                      << ", Target Pitch: " << desired_angles.target_pitch_degrees << std::endl;

            // TODO: 在此处将 desired_angles 发送给云台控制器
            // TODO: 更新 current_gimbal_azimuth 和 current_gimbal_pitch 为云台移动后的实际角度
            // current_gimbal_azimuth = desired_angles.target_azimuth_degrees; // 简化模拟
            // current_gimbal_pitch = desired_angles.target_pitch_degrees;   // 简化模拟
        }
        else
        {
            std::cout << "No spray targets detected." << std::endl;
        }
        std::cout << "------------------------------------" << std::endl;

        cv::imshow("Fire Detection Visual Output", display_image);
        char key = (char)cv::waitKey(500); // 增加延时方便观察
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "Vision Processing Terminated." << std::endl;
    return 0;
}