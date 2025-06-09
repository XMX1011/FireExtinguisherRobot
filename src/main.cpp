#include "vision_processing.h"
#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// 将相机参数封装为结构体
struct CameraParams {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
                             500.0, 0.0, 320.0,
                             0.0, 500.0, 240.0,
                             0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    float hfov_degrees = 60.0f;
    float vfov_degrees = 45.0f;
    float nozzle_azimuth_offset = 0.0f;
    float nozzle_pitch_offset = 0.0f;
};

// utils.h 中声明的辅助函数实现
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

float calculateRealWorldDistance(const cv::Point3f &p1, const cv::Point3f &p2)
{
    if (p1.z == 0.0f || p2.z == 0.0f)
        return FLT_MAX;
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

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

bool getThermalImageAsTemperatureMatrix(const std::string& image_path, 
                                         cv::Mat& temp_matrix, 
                                         float min_temp, 
                                         float max_temp,
                                         const cv::Size& target_size = cv::Size(384, 288)) {
    // 1. 读取灰度图
    cv::Mat gray_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (gray_image.empty()) {
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