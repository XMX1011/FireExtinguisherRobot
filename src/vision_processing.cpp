// src/vision_processing.cpp
#include "vision_processing.h"
#include <iostream>
#include <algorithm>

// --- 核心视觉处理函数实现 ---

/**
 * @brief 检测并过滤热点
 * 
 * @param temp_matrix 温度矩阵
 * @param camera_matrix_param 相机矩阵参数
 * @param assumed_distance_to_fire_plane_param 假设的到火源平面的距离
 * @return 返回检测到的热点向量
 * 
 * 此函数负责从温度矩阵中检测出热点，并根据相机参数和假设的距离进行过滤和处理
 * 它首先检查输入的温度矩阵和相机矩阵的有效性，然后通过阈值处理和形态学操作来提取和优化热点区域
 * 最后，它计算每个热点的特征，如面积、最大温度等，并将其转换为世界坐标系中的近似位置
 */
std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat &temp_matrix,
    const cv::Mat &camera_matrix_param,
    float assumed_distance_to_fire_plane_param)
{
    std::vector<HotSpot> detected_spots;

    // 检测温度矩阵是否有效
    if (temp_matrix.empty() || temp_matrix.type() != CV_32FC1)
    {
        std::cerr << "Error: Temperature matrix is empty or not CV_32FC1 type." << std::endl;
        return detected_spots;
    }

    // 检查 camera_matrix_param 是否有效
    if (camera_matrix_param.empty() || camera_matrix_param.rows != 3 || camera_matrix_param.cols != 3)
    {
        std::cerr << "Error: Invalid camera matrix provided." << std::endl;
        return detected_spots;
    }

    cv::Mat binary_mask;
    cv::threshold(temp_matrix, binary_mask, temperature_threshold, 255.0, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int spot_id_counter = 0;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area < min_hotspot_area_pixels)
            continue;

        cv::Moments M = cv::moments(contour);
        if (M.m00 == 0)
            continue;
        cv::Point2f centroid(static_cast<float>(M.m10 / M.m00), static_cast<float>(M.m01 / M.m00));

        cv::Rect bounding_box = cv::boundingRect(contour);
        cv::Mat spot_roi_mask = cv::Mat::zeros(temp_matrix.size(), CV_8U);
        std::vector<std::vector<cv::Point>> contour_vec{contour};
        cv::drawContours(spot_roi_mask, contour_vec, -1, cv::Scalar(255), cv::FILLED);

        double min_temp, max_temp_in_roi;
        cv::minMaxLoc(temp_matrix, &min_temp, &max_temp_in_roi, nullptr, nullptr, spot_roi_mask);

        HotSpot spot;
        spot.id = spot_id_counter++;
        spot.pixel_centroid = centroid;
        spot.area_pixels = area;
        spot.max_temperature = static_cast<float>(max_temp_in_roi);
        spot.contour_pixels = contour;
        spot.world_coord_approx = pixelToApproxWorld(centroid, camera_matrix_param, assumed_distance_to_fire_plane_param);
        detected_spots.push_back(spot);
    }

    return detected_spots;
}

/**
 * @brief 确定喷洒目标
 * 
 * @param hot_spots 热点向量
 * @param max_grouping_distance_param 最大分组距离参数
 * @return 返回最终的目标向量
 * 
 * 此函数根据热点的分布情况确定喷洒目标它首先检查热点是否为空，然后对每个热点进行分组处理
 * 分组是根据热点之间的距离来决定的，如果距离小于最大分组距离参数，则将它们归为一组
 * 每个目标的严重性指标是根据其包含的所有热点的面积和最大温度来计算的
 */
std::vector<SprayTarget> determineSprayTargets(
    std::vector<HotSpot> &hot_spots,
    float max_grouping_distance_param)
{
    std::vector<SprayTarget> final_targets;

    if (hot_spots.empty())
        return final_targets;

    for (auto &spot : hot_spots)
        spot.grouped = false;

    int target_id_counter = 0;

    for (size_t i = 0; i < hot_spots.size(); ++i)
    {
        if (hot_spots[i].grouped)
            continue;

        SprayTarget current_target;
        current_target.id = target_id_counter++;
        current_target.source_hotspot_ids.push_back(hot_spots[i].id);
        hot_spots[i].grouped = true;

        cv::Point2f sum_pixel_centroids = hot_spots[i].pixel_centroid;
        cv::Point3f sum_world_centroids_approx = hot_spots[i].world_coord_approx;
        float total_severity_metric = static_cast<float>(hot_spots[i].area_pixels * hot_spots[i].max_temperature);
        int num_in_group = 1;

        for (size_t j = i + 1; j < hot_spots.size(); ++j)
        {
            if (hot_spots[j].grouped)
                continue;

            if (calculateRealWorldDistance(hot_spots[i].world_coord_approx, hot_spots[j].world_coord_approx) < max_grouping_distance_param)
            {
                hot_spots[j].grouped = true;
                current_target.source_hotspot_ids.push_back(hot_spots[j].id);
                sum_pixel_centroids += hot_spots[j].pixel_centroid;
                sum_world_centroids_approx += hot_spots[j].world_coord_approx;
                total_severity_metric += static_cast<float>(hot_spots[j].area_pixels * hot_spots[j].max_temperature);
                num_in_group++;
            }
        }

        current_target.final_pixel_aim_point = sum_pixel_centroids * (1.0f / num_in_group);
        current_target.final_world_aim_point_approx = sum_world_centroids_approx * (1.0f / num_in_group);
        current_target.estimated_severity = total_severity_metric;
        final_targets.push_back(current_target);
    }

    std::sort(final_targets.begin(), final_targets.end());
    return final_targets;
}

/**
 * @brief 可视化结果
 * 
 * @param display_image 显示图像
 * @param temp_matrix 温度矩阵
 * @param hot_spots 热点向量
 * @param spray_targets 喷洒目标向量
 * 
 * 此函数负责将处理结果可视化到图像上它首先绘制每个热点的轮廓和中心点，然后绘制温度阈值处理后的轮廓
 * 最后，它绘制每个喷洒目标的瞄准点和组成该目标的热点的边界框
 */
void visualizeResults(
    cv::Mat &display_image,
    const cv::Mat &temp_matrix, 
    const std::vector<HotSpot> &hot_spots,
    const std::vector<SprayTarget> &spray_targets)
{
    for (const auto &spot : hot_spots)
    {
        std::vector<std::vector<cv::Point>> contour_vec{spot.contour_pixels};
        cv::drawContours(display_image, contour_vec, -1, cv::Scalar(0, 255, 0), 1);
        cv::circle(display_image, spot.pixel_centroid, 3, cv::Scalar(0, 0, 255), -1);
    }

    cv::Mat binary_mask;
    cv::threshold(temp_matrix, binary_mask, temperature_threshold, 255.0, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::drawContours(display_image, contours, -1, cv::Scalar(255, 255, 255), 1);

    int target_rank = 1;
    for (const auto &target : spray_targets)
    {
        cv::circle(display_image, target.final_pixel_aim_point, 8, cv::Scalar(255, 0, 255), 2);
        cv::putText(display_image, "T" + std::to_string(target_rank++),
                    target.final_pixel_aim_point + cv::Point2f(10, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        for (const auto &spot_id : target.source_hotspot_ids)
        {
            if (spot_id < hot_spots.size())
            {
                const auto &spot = hot_spots[spot_id];
                cv::Rect bounding_box = cv::boundingRect(spot.contour_pixels);
                cv::rectangle(display_image, bounding_box, cv::Scalar(0, 0, 0), 1);
            }
        }
    }
}