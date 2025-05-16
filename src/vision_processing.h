// src/vision_processing.h
#ifndef VISION_PROCESSING_H
#define VISION_PROCESSING_H

#include "utils.h" // 包含结构体和参数定义
#include <opencv2/opencv.hpp>
#include <vector>

// --- 核心视觉处理函数声明 ---

/**
 * @brief 从温度矩阵中检测并过滤高温热点。
 * @param temp_matrix 输入的温度矩阵 (CV_32FC1)。
 * @param camera_matrix 相机内参矩阵 (用于坐标转换)。
 * @param assumed_distance_to_fire_plane 假设的火源平面距离 (用于坐标转换)。
 * @return 检测到的有效热点列表。
 */
std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat &temp_matrix,
    const cv::Mat &camera_matrix, // 传入相机参数
    float assumed_distance_to_fire_plane);

/**
 * @brief 根据检测到的热点确定最终的喷射目标。
 * @param hot_spots 输入的已检测和过滤的热点列表 (此函数可能会修改其 grouped 状态)。
 * @param max_grouping_distance 热点群组化的最大真实世界距离。
 * @return 计算得到的喷射目标列表，按严重性排序。
 */
std::vector<SprayTarget> determineSprayTargets(
    std::vector<HotSpot> &hot_spots, // 传入引用以修改 grouped 状态
    float max_grouping_distance      // 传入分组距离参数
);

/**
 * @brief 在显示图像上绘制检测结果和目标。
 * @param display_image 用于绘制的图像 (通常是伪彩色的温度图)。
 * @param hot_spots 检测到的热点列表。
 * @param spray_targets 计算得到的喷射目标列表。
 */
void visualizeResults(
    cv::Mat &display_image, // 传入引用以直接绘制
    const std::vector<HotSpot> &hot_spots,
    const std::vector<SprayTarget> &spray_targets);

#endif // VISION_PROCESSING_H