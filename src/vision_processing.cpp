// src/vision_processing.cpp
#include "vision_processing.h"
#include <iostream>
#include <algorithm>
#include <cmath> // For std::abs, fmod

// detectAndFilterHotspots, determineSprayTargets, visualizeResults 函数实现保持不变

std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat &temp_matrix,
    const cv::Mat &camera_matrix_param,
    float assumed_distance_to_fire_plane_param)
{
    std::vector<HotSpot> detected_spots;
    if (temp_matrix.empty() || temp_matrix.type() != CV_32FC1)
    {
        std::cerr << "Error: Temperature matrix is empty or not CV_32FC1 type." << std::endl;
        return detected_spots;
    }

    cv::Mat binary_mask;
    cv::threshold(temp_matrix, binary_mask, FIRE_TEMPERATURE_THRESHOLD_CELSIUS, 255.0, cv::THRESH_BINARY);
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
        if (area < MIN_HOTSPOT_AREA_PIXELS)
            continue;

        cv::Moments M = cv::moments(contour);
        if (M.m00 == 0)
            continue;
        cv::Point2f centroid(static_cast<float>(M.m10 / M.m00), static_cast<float>(M.m01 / M.m00));

        cv::Rect bounding_box = cv::boundingRect(contour);
        cv::Mat spot_roi_mask = cv::Mat::zeros(temp_matrix.size(), CV_8U);
        cv::drawContours(spot_roi_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
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
        if (sum_world_centroids_approx.z != 0.0f && num_in_group > 0)
        {
            current_target.final_world_aim_point_approx = sum_world_centroids_approx * (1.0f / num_in_group);
        }
        else
        {
            current_target.final_world_aim_point_approx = cv::Point3f(0, 0, 0);
        }
        current_target.estimated_severity = total_severity_metric;
        final_targets.push_back(current_target);
    }

    std::sort(final_targets.begin(), final_targets.end());
    return final_targets;
}

void visualizeResults(
    cv::Mat &display_image,
    const std::vector<HotSpot> &hot_spots,
    const std::vector<SprayTarget> &spray_targets)
{
    for (const auto &spot : hot_spots)
    {
        cv::drawContours(display_image, std::vector<std::vector<cv::Point>>{spot.contour_pixels}, -1, cv::Scalar(0, 255, 0), 1);
        cv::circle(display_image, spot.pixel_centroid, 3, cv::Scalar(0, 0, 255), -1);
    }

    int target_rank = 1;
    for (const auto &target : spray_targets)
    {
        cv::circle(display_image, target.final_pixel_aim_point, 8, cv::Scalar(255, 0, 255), 2);
        cv::putText(display_image, "T" + std::to_string(target_rank++),
                    target.final_pixel_aim_point + cv::Point2f(10, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    }
}

// 新增：计算云台角度函数实现
CloudGimbalAngles calculateGimbalAngles(
    const cv::Point2f &target_pixel_coords,
    int image_width,
    int image_height,
    float camera_hfov_degrees,
    float camera_vfov_degrees,
    float current_cloud_azimuth_degrees,
    float current_cloud_pitch_degrees,
    float nozzle_offset_azimuth_degrees,
    float nozzle_offset_pitch_degrees)
{
    if (image_width <= 0 || image_height <= 0 || camera_hfov_degrees <= 0 || camera_vfov_degrees <= 0)
    {
        std::cerr << "Error: Invalid image dimensions or FOV for angle calculation." << std::endl;
        return {current_cloud_azimuth_degrees, current_cloud_pitch_degrees}; // Return current if params invalid
    }

    float cx = static_cast<float>(image_width) / 2.0f;
    float cy = static_cast<float>(image_height) / 2.0f;

    float delta_azimuth_pixels = target_pixel_coords.x - cx;
    // 像素偏移 / 半宽度像素 * 半FOV = 角度偏移
    float delta_azimuth_degrees = (delta_azimuth_pixels / cx) * (camera_hfov_degrees / 2.0f);

    float delta_pitch_pixels = target_pixel_coords.y - cy;
    // 注意：图像Y轴通常向下为正。如果云台向上为正Pitch，则需要反转符号。
    // 假设：目标在图像下方 (target_pixel_coords.y > cy) -> delta_pitch_pixels 为正
    // 此时如果希望云台向上运动 (正Pitch)，则delta_pitch_degrees应为正。
    // 如果希望云台向下运动 (负Pitch)，则delta_pitch_degrees应为负。
    // 这里的实现：Y向下为正像素偏移，假设对应云台PITCH向上为正角度。
    // (如果云台向上为负Pitch，则delta_pitch_degrees需要乘以-1)
    float delta_pitch_degrees = (delta_pitch_pixels / cy) * (camera_vfov_degrees / 2.0f);
    // 示例：如果图像Y向下对应云台俯仰角向下（负值），则需要反转。
    // delta_pitch_degrees *= -1.0f; // 根据您的云台定义调整

    CloudGimbalAngles angles;
    angles.target_azimuth_degrees = current_cloud_azimuth_degrees + delta_azimuth_degrees - nozzle_offset_azimuth_degrees;
    angles.target_pitch_degrees = current_cloud_pitch_degrees + delta_pitch_degrees - nozzle_offset_pitch_degrees;

    // 简单的角度归一化 (示例，实际云台可能有特定范围和处理方式)
    // angles.target_azimuth_degrees = fmod(fmod(angles.target_azimuth_degrees, 360.0f) + 360.0f, 360.0f);
    // 俯仰角通常有机械限位，例如 [-90, 90] 或 [-45, 45]
    // if (angles.target_pitch_degrees > 45.0f) angles.target_pitch_degrees = 45.0f;
    // if (angles.target_pitch_degrees < -45.0f) angles.target_pitch_degrees = -45.0f;

    // 调试信息
    std::cout << "Target Pixel: (" << target_pixel_coords.x << ", " << target_pixel_coords.y << ")" << std::endl;
    std::cout << "Image Center: (" << cx << ", " << cy << ")" << std::endl;
    std::cout << "Delta Pixels (Az, P): (" << delta_azimuth_pixels << ", " << delta_pitch_pixels << ")" << std::endl;
    std::cout << "Delta Degrees (Az, P): (" << delta_azimuth_degrees << ", " << delta_pitch_degrees << ")" << std::endl;
    std::cout << "Nozzle Offset (Az, P): (" << nozzle_offset_azimuth_degrees << ", " << nozzle_offset_pitch_degrees << ")" << std::endl;
    std::cout << "Current Cloud (Az, P): (" << current_cloud_azimuth_degrees << ", " << current_cloud_pitch_degrees << ")" << std::endl;
    std::cout << "Calculated Target Cloud (Az, P): (" << angles.target_azimuth_degrees << ", " << angles.target_pitch_degrees << ")" << std::endl;

    return angles;
}