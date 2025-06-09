// src/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath> // For fmod if needed for angle normalization

// --- 配置参数 ---
const float FIRE_TEMPERATURE_THRESHOLD_CELSIUS = 250.0f;
const double MIN_HOTSPOT_AREA_PIXELS = 30.0;
const float MAX_GROUPING_DISTANCE_METERS = 1.0f;
const float ASSUMED_DISTANCE_TO_FIRE_PLANE_METERS = 8.0f; // !!! 强假设 !!!

// --- 相机内参和FOV (理想情况下从 camera_params.xml 或专门的相机配置文件加载) ---
extern cv::Mat CAMERA_MATRIX;
extern cv::Mat DIST_COEFFS;
extern float CAMERA_HFOV_DEGREES; // 水平视场角
extern float CAMERA_VFOV_DEGREES; // 垂直视场角

// --- 喷嘴偏移参数 (通过标定获得) ---
extern float NOZZLE_OFFSET_AZIMUTH_DEGREES; // 喷嘴相对相机的回转角偏移
extern float NOZZLE_OFFSET_PITCH_DEGREES;   // 喷嘴相对相机的俯仰角偏移


// --- 结构体定义 ---
struct HotSpot {
    int id;
    cv::Point2f pixel_centroid;
    cv::Point3f world_coord_approx;
    double area_pixels;
    float max_temperature;
    std::vector<cv::Point> contour_pixels;
    bool grouped = false;

    HotSpot() : id(-1), area_pixels(0.0), max_temperature(0.0f), grouped(false) {}
};

struct SprayTarget {
    int id;
    cv::Point2f final_pixel_aim_point;
    cv::Point3f final_world_aim_point_approx;
    std::vector<int> source_hotspot_ids;
    float estimated_severity;

    SprayTarget() : id(-1), estimated_severity(0.0f) {}

    bool operator<(const SprayTarget& other) const {
        return estimated_severity > other.estimated_severity;
    }
};

// 新增：云台角度结构体
struct CloudGimbalAngles {
    float target_azimuth_degrees;
    float target_pitch_degrees;

    CloudGimbalAngles() : target_azimuth_degrees(0.0f), target_pitch_degrees(0.0f) {}
     // 带参数的构造函数
    CloudGimbalAngles(float azimuth, float pitch)
        : target_azimuth_degrees(azimuth), target_pitch_degrees(pitch) {}
};


// --- 通用辅助函数声明 ---
cv::Point3f pixelToApproxWorld(const cv::Point2f& pixel_coord, const cv::Mat& cam_matrix, float distance_to_plane);
float calculateRealWorldDistance(const cv::Point3f& p1, const cv::Point3f& p2);
bool getSimulatedTemperatureMatrix(cv::Mat& temp_matrix, int rows, int cols); // 保持模拟数据函数


#endif // UTILS_H