// src/vision_processing.cpp
#include "vision_processing.h"
#include <iostream>  // For std::cerr, std::cout (可以考虑使用日志库)
#include <algorithm> // For std::sort

// --- 通用辅助函数实现 (如果定义在 utils.h, 则不需要在这里重复) ---
// (如果 pixelToApproxWorld 和 calculateRealWorldDistance 在 utils.h 中声明并在 utils.cpp 中定义，
// 或者作为内联函数直接在 utils.h 中定义，那么这里就不需要了。
// 为简单起见，如果它们是 utils.h 中的内联或模板，就不需要 utils.cpp)
// 这里假设它们已在 utils.h 中声明，并在 utils.cpp 或 utils.h (内联) 中定义。
// 我们把它们的实现放到 utils.cpp (或者如果简单，直接内联到 utils.h)

// (getSimulatedTemperatureMatrix 也是，假设它在 utils.h/utils.cpp)

// --- 核心视觉处理函数实现 ---

std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat &temp_matrix,
    const cv::Mat &camera_matrix_param, // 接收相机参数
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
        // 使用传入的相机参数进行转换
        spot.world_coord_approx = pixelToApproxWorld(centroid, camera_matrix_param, assumed_distance_to_fire_plane_param);
        detected_spots.push_back(spot);
    }
    return detected_spots;
}

std::vector<SprayTarget> determineSprayTargets(
    std::vector<HotSpot> &hot_spots, // 接收引用
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