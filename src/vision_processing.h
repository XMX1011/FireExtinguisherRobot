// src/vision_processing.h
#ifndef VISION_PROCESSING_H
#define VISION_PROCESSING_H

#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>

// --- 核心视觉处理函数声明 ---

/**
 * @brief 检测并过滤热点区域
 * 
 * @param temp_matrix 温度矩阵，包含每个像素的温度信息
 * @param camera_matrix 相机内参矩阵，用于校正相机畸变
 * @param assumed_distance_to_fire_plane 假定的火源平面距离，用于深度计算
 * 
 * @return 返回过滤后的热点区域向量，这些热点被认为是潜在的火源
 */
std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat& temp_matrix,
    const cv::Mat& camera_matrix,
    float assumed_distance_to_fire_plane
);

/**
 * @brief 确定喷射目标
 * 
 * @param hot_spots 热点区域向量，由detectAndFilterHotspots函数提供
 * @param max_grouping_distance 最大分组距离，用于决定哪些热点可以被分组为一个喷射目标
 * 
 * @return 返回喷射目标向量，每个目标包含一组靠近的热点
 */
std::vector<SprayTarget> determineSprayTargets(
    std::vector<HotSpot>& hot_spots,
    float max_grouping_distance
);

/**
 * @brief 可视化结果
 * 
 * @param display_image 显示图像，将在该图像上绘制热点和喷射目标
 * @param hot_spots 热点区域向量，用于在图像上绘制热点
 * @param spray_targets 喷射目标向量，用于在图像上绘制喷射目标
 * 
 * 此函数将热点和喷射目标可视化，以便用户可以直观地看到检测结果
 */
void visualizeResults(
    cv::Mat& display_image,
    const std::vector<HotSpot>& hot_spots,
    const std::vector<SprayTarget>& spray_targets
);

// 新增：计算云台角度函数声明
/**
 * @brief  计算云台角度
 * 
 * @param target_pixel_coords 目标像素坐标，表示感兴趣点在图像中的位置
 * @param image_width 图像宽度，用于计算角度
 * @param image_height 图像高度，用于计算角度
 * @param camera_hfov_degrees 相机水平视场角，用于计算角度
 * @param camera_vfov_degrees 相机垂直视场角，用于计算角度
 * @param current_cloud_azimuth_degrees 云台当前的回转角 (从外部获取)
 * @param current_cloud_pitch_degrees 云台当前的俯仰角 (从外部获取)
 * @param nozzle_offset_azimuth_degrees 喷嘴相对于云台的回转角偏移
 * @param nozzle_offset_pitch_degrees 喷嘴相对于云台的俯仰角偏移
 * 
 * @return 返回云台角度，包括回转角和俯仰角，用于对准目标
 * 
 * 此函数根据目标在图像中的位置和相机参数计算云台需要调整到的角度
 */
CloudGimbalAngles calculateGimbalAngles(
    const cv::Point2f& target_pixel_coords,
    int image_width,
    int image_height,
    float camera_hfov_degrees,
    float camera_vfov_degrees,
    float current_cloud_azimuth_degrees, // 云台当前的回转角 (从外部获取)
    float current_cloud_pitch_degrees,   // 云台当前的俯仰角 (从外部获取)
    float nozzle_offset_azimuth_degrees,
    float nozzle_offset_pitch_degrees
);
#endif