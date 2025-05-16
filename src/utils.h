// src/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string> // For std::string if needed later

// --- 配置参数 (可以考虑移到专门的config类或从文件加载) ---
// 但为了简单，先放在这里，实际项目中这些值可能来自 camera_params.xml 或其他配置
const float FIRE_TEMPERATURE_THRESHOLD_CELSIUS = 150.0f;
const double MIN_HOTSPOT_AREA_PIXELS = 30.0;
const float MAX_GROUPING_DISTANCE_METERS = 1.0f;
const float ASSUMED_DISTANCE_TO_FIRE_PLANE_METERS = 5.0f; // !!! 强假设 !!!

// --- 相机内参 (理想情况下从 camera_params.xml 加载) ---
// 示例值，您必须替换为您的真实相机标定结果
// 在实际应用中，这些应该在初始化时从文件读取
extern cv::Mat CAMERA_MATRIX; // 声明为 extern，定义在 main.cpp 或 config加载模块
extern cv::Mat DIST_COEFFS;   // 声明为 extern

// --- 结构体定义 ---
struct HotSpot
{
    int id;
    cv::Point2f pixel_centroid;
    cv::Point3f world_coord_approx;
    double area_pixels;
    float max_temperature;
    std::vector<cv::Point> contour_pixels;
    bool grouped = false;

    HotSpot() : id(-1), area_pixels(0.0), max_temperature(0.0f), grouped(false) {}
};

struct SprayTarget
{
    int id;
    cv::Point2f final_pixel_aim_point;
    cv::Point3f final_world_aim_point_approx;
    std::vector<int> source_hotspot_ids;
    float estimated_severity;

    SprayTarget() : id(-1), estimated_severity(0.0f) {}

    // 比较函数，用于排序 (按严重性降序)
    bool operator<(const SprayTarget &other) const
    {
        return estimated_severity > other.estimated_severity;
    }
};

// --- 通用辅助函数声明 ---
// (如果 pixelToApproxWorld 和 calculateRealWorldDistance 是纯粹的数学转换，
// 并且不依赖于 vision_processing 内部状态，可以放在这里)

cv::Point3f pixelToApproxWorld(const cv::Point2f &pixel_coord, const cv::Mat &cam_matrix, float distance_to_plane);
float calculateRealWorldDistance(const cv::Point3f &p1, const cv::Point3f &p2);

// 模拟从SDK获取温度矩阵 (实际项目中替换为真实SDK调用, 可能放在更专门的camera类中)
bool getSimulatedTemperatureMatrix(cv::Mat &temp_matrix, int rows, int cols);

#endif // UTILS_H