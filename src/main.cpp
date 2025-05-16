// src/main.cpp
#include "vision_processing.h" // 主要的视觉处理函数
#include "utils.h"             // 结构体, 全局参数, 辅助函数声明
#include <iostream>
#include <opencv2/opencv.hpp> // 确保包含OpenCV

// --- 全局相机参数定义 (在 utils.h 中声明为 extern) ---
// 这些值理想情况下从 config/camera_params.xml 加载
cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3, 3) << 500.0, 0.0, 320.0,
                         0.0, 500.0, 240.0,
                         0.0, 0.0, 1.0);
cv::Mat DIST_COEFFS = cv::Mat::zeros(4, 1, CV_64F);

// --- utils.h 中声明的辅助函数实现 ---
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

bool getSimulatedTemperatureMatrix(cv::Mat &temp_matrix, int rows, int cols)
{
    temp_matrix = cv::Mat(rows, cols, CV_32FC1);
    cv::randu(temp_matrix, cv::Scalar(20.0f), cv::Scalar(40.0f));
    cv::circle(temp_matrix, cv::Point(cols / 4, rows / 3), 15, cv::Scalar(250.0f), -1);
    cv::circle(temp_matrix, cv::Point(cols / 4 + 30, rows / 3 + 20), 12, cv::Scalar(200.0f), -1);
    cv::circle(temp_matrix, cv::Point(cols * 3 / 4, rows / 2), 20, cv::Scalar(300.0f), -1);
    cv::circle(temp_matrix, cv::Point(cols / 2, rows * 3 / 4), 3, cv::Scalar(180.0f), -1);
    return !temp_matrix.empty();
}

// --- 配置加载函数 (示例) ---
bool loadCameraParameters(const std::string &filename, cv::Mat &cam_matrix, cv::Mat &dist_coeffs_out)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open camera parameters file: " << filename << std::endl;
        std::cerr << "Using default/hardcoded camera parameters." << std::endl;
        return false;
    }
    fs["camera_matrix"] >> cam_matrix;
    fs["distortion_coefficients"] >> dist_coeffs_out;
    fs.release();
    std::cout << "Camera parameters loaded from " << filename << std::endl;
    return true;
}

int main()
{
    cv::Mat temperature_matrix;
    cv::Mat display_image;

    int frame_rows = 480;
    int frame_cols = 640;

    // 尝试从文件加载相机参数
    // 注意：全局的 CAMERA_MATRIX 和 DIST_COEFFS 变量会被这个函数修改
    std::string camera_params_file = "../config/camera_params.xml"; // 假设config在项目根目录的上一级
    if (!loadCameraParameters(camera_params_file, CAMERA_MATRIX, DIST_COEFFS))
    {
        // 如果加载失败，将使用在 main.cpp 顶部定义的硬编码值
    }

    std::cout << "Vision Processing for Fire Suppression Started." << std::endl;
    std::cout << "Press 'q' or ESC to exit." << std::endl;

    while (true)
    {
        // 在实际应用中，这里会调用相机SDK的函数
        if (!getSimulatedTemperatureMatrix(temperature_matrix, frame_rows, frame_cols))
        {
            std::cerr << "Error: Could not get temperature matrix." << std::endl;
            break;
        }

        // 核心视觉处理步骤
        // 将全局的相机参数传递给处理函数
        std::vector<HotSpot> hot_spots = detectAndFilterHotspots(temperature_matrix, CAMERA_MATRIX, ASSUMED_DISTANCE_TO_FIRE_PLANE_METERS);
        std::vector<SprayTarget> spray_targets = determineSprayTargets(hot_spots, MAX_GROUPING_DISTANCE_METERS);

        // 可视化
        cv::Mat normalized_temp;
        cv::normalize(temperature_matrix, normalized_temp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(normalized_temp, display_image, cv::COLORMAP_JET);
        visualizeResults(display_image, hot_spots, spray_targets);

        // 输出结果到控制台
        if (spray_targets.empty())
        {
            std::cout << "No spray targets detected." << std::endl;
        }
        else
        {
            int rank = 1;
            for (const auto &target : spray_targets)
            {
                std::cout << "Target ID: " << target.id << ", Rank: " << rank++
                          << ", Pixel: (" << target.final_pixel_aim_point.x << ", " << target.final_pixel_aim_point.y << ")"
                          << ", Approx World: (" << target.final_world_aim_point_approx.x << ", "
                          << target.final_world_aim_point_approx.y << ", "
                          << target.final_world_aim_point_approx.z << ")"
                          << ", Severity: " << target.estimated_severity << std::endl;
            }
        }
        std::cout << "------------------------------------" << std::endl;

        cv::imshow("Fire Detection Visual Output", display_image);
        char key = (char)cv::waitKey(500);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "Vision Processing Terminated." << std::endl;
    return 0;
}