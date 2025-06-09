// src/vision_processing.h
#ifndef VISION_PROCESSING_H
#define VISION_PROCESSING_H

#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>

// --- 核心视觉处理函数声明 ---

std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat& temp_matrix,
    const cv::Mat& camera_matrix,
    float assumed_distance_to_fire_plane
);

std::vector<SprayTarget> determineSprayTargets(
    std::vector<HotSpot>& hot_spots,
    float max_grouping_distance
);

void visualizeResults(
    cv::Mat& display_image,
    const std::vector<HotSpot>& hot_spots,
    const std::vector<SprayTarget>& spray_targets
);

// 新增：计算云台角度函数声明
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