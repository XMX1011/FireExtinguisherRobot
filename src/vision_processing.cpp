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

/**
 * @brief 检测并过滤热点区域，返回检测到的热点信息。
 *
 * 该函数接收温度矩阵、相机参数和假设的火焰平面距离，通过图像处理技术检测温度矩阵中的热点区域，
 * 并根据面积和温度阈值进行过滤，最终返回符合条件的热点信息。
 *
 * @param temp_matrix 输入的温度矩阵，类型应为 CV_32FC1，表示浮点型单通道矩阵。
 *                    矩阵中的每个元素表示对应像素点的温度值。
 * @param camera_matrix_param 相机参数矩阵，用于将像素坐标转换为近似的世界坐标。
 * @param assumed_distance_to_fire_plane_param 假设的火焰平面距离，用于计算热点的世界坐标。
 * @return std::vector<HotSpot> 返回检测到的热点信息列表，每个热点包含其 ID、像素坐标、面积、
 *         最高温度、轮廓点以及近似的世界坐标。
 */
std::vector<HotSpot> detectAndFilterHotspots(
    const cv::Mat &temp_matrix,
    const cv::Mat &camera_matrix_param, // 接收相机参数
    float assumed_distance_to_fire_plane_param)
{
    std::vector<HotSpot> detected_spots;

    // 检查输入温度矩阵是否为空或类型不正确
    if (temp_matrix.empty() || temp_matrix.type() != CV_32FC1)
    {
        std::cerr << "Error: Temperature matrix is empty or not CV_32FC1 type." << std::endl;
        return detected_spots;
    }

    // 使用温度阈值生成二值掩码，标记高于阈值的区域
    cv::Mat binary_mask;
    cv::threshold(temp_matrix, binary_mask, FIRE_TEMPERATURE_THRESHOLD_CELSIUS, 255.0, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U);

    // 对二值掩码进行形态学开运算和闭运算，去除噪声并填充空洞
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

    // 查找二值掩码中的外部轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int spot_id_counter = 0;

    // 遍历所有检测到的轮廓，筛选符合条件的热点区域
    for (const auto &contour : contours)
    {
        // 计算轮廓面积，过滤掉面积过小的区域
        double area = cv::contourArea(contour);
        if (area < MIN_HOTSPOT_AREA_PIXELS)
            continue;

        // 计算轮廓的质心
        cv::Moments M = cv::moments(contour);
        if (M.m00 == 0)
            continue;
        cv::Point2f centroid(static_cast<float>(M.m10 / M.m00), static_cast<float>(M.m01 / M.m00));

        // 获取轮廓的边界框，并创建一个掩码以提取该区域的温度信息
        cv::Rect bounding_box = cv::boundingRect(contour);
        cv::Mat spot_roi_mask = cv::Mat::zeros(temp_matrix.size(), CV_8U);
        cv::drawContours(spot_roi_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        double min_temp, max_temp_in_roi;
        cv::minMaxLoc(temp_matrix, &min_temp, &max_temp_in_roi, nullptr, nullptr, spot_roi_mask);

        // 构造 HotSpot 对象并填充相关信息
        HotSpot spot;
        spot.id = spot_id_counter++;
        spot.pixel_centroid = centroid;
        spot.area_pixels = area;
        spot.max_temperature = static_cast<float>(max_temp_in_roi);
        spot.contour_pixels = contour;

        // 使用相机参数和假设的距离将像素坐标转换为近似的世界坐标
        spot.world_coord_approx = pixelToApproxWorld(centroid, camera_matrix_param, assumed_distance_to_fire_plane_param);
        detected_spots.push_back(spot);
    }

    return detected_spots;
}

/**
 * @brief 根据热点数据确定喷洒目标。
 *
 * 该函数接收一组热点数据，并根据最大分组距离参数将热点分组为喷洒目标。
 * 每个喷洒目标包含一组热点的聚合信息，例如像素中心点、世界坐标近似值和严重性估计。
 *
 * @param hot_spots 热点数据的引用，每个热点包含像素坐标、世界坐标、面积、温度等信息。
 *                  函数会修改热点的 `grouped` 属性以标记是否已被分组。
 * @param max_grouping_distance_param 最大分组距离参数，用于判断两个热点是否属于同一组。
 *                                    单位为世界坐标系中的距离。
 * @return std::vector<SprayTarget> 返回一个包含所有喷洒目标的向量，每个目标包含聚合后的信息。
 */
std::vector<SprayTarget> determineSprayTargets(
    std::vector<HotSpot> &hot_spots,
    float max_grouping_distance_param)
{
    std::vector<SprayTarget> final_targets;

    // 如果热点列表为空，直接返回空的目标列表
    if (hot_spots.empty())
        return final_targets;

    // 初始化所有热点的分组状态为未分组
    for (auto &spot : hot_spots)
        spot.grouped = false;

    int target_id_counter = 0;

    // 遍历每个热点，尝试将其作为新分组的起点
    for (size_t i = 0; i < hot_spots.size(); ++i)
    {
        // 如果当前热点已被分组，则跳过
        if (hot_spots[i].grouped)
            continue;

        // 创建一个新的喷洒目标并初始化其属性
        SprayTarget current_target;
        current_target.id = target_id_counter++;
        current_target.source_hotspot_ids.push_back(hot_spots[i].id);
        hot_spots[i].grouped = true;

        // 初始化聚合变量，用于计算分组的中心点和严重性
        cv::Point2f sum_pixel_centroids = hot_spots[i].pixel_centroid;
        cv::Point3f sum_world_centroids_approx = hot_spots[i].world_coord_approx;
        float total_severity_metric = static_cast<float>(hot_spots[i].area_pixels * hot_spots[i].max_temperature);
        int num_in_group = 1;

        // 遍历后续热点，寻找可以加入当前分组的热点
        for (size_t j = i + 1; j < hot_spots.size(); ++j)
        {
            if (hot_spots[j].grouped)
                continue;

            // 如果两个热点之间的世界坐标距离小于最大分组距离，则将其加入当前分组
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

        // 计算当前分组的最终像素瞄准点和世界坐标近似值
        current_target.final_pixel_aim_point = sum_pixel_centroids * (1.0f / num_in_group);
        if (sum_world_centroids_approx.z != 0.0f && num_in_group > 0)
        {
            current_target.final_world_aim_point_approx = sum_world_centroids_approx * (1.0f / num_in_group);
        }
        else
        {
            current_target.final_world_aim_point_approx = cv::Point3f(0, 0, 0);
        }

        // 设置当前分组的严重性估计并将其添加到最终目标列表中
        current_target.estimated_severity = total_severity_metric;
        final_targets.push_back(current_target);
    }

    // 对所有喷洒目标按严重性进行排序
    std::sort(final_targets.begin(), final_targets.end());
    return final_targets;
}

/**
 * @brief 可视化热点和喷洒目标的结果。
 *
 * 该函数在输入图像上绘制热点的轮廓和中心点，以及喷洒目标的瞄准点和排名。
 *
 * @param display_image 输入的显示图像，函数会在此图像上绘制结果。
 * @param hot_spots 热点数据的引用，用于绘制热点的轮廓和中心点。
 * @param spray_targets 喷洒目标的引用，用于绘制目标的瞄准点和排名。
 */
void visualizeResults(
    cv::Mat &display_image,
    const std::vector<HotSpot> &hot_spots,
    const std::vector<SprayTarget> &spray_targets)
{
    // 绘制每个热点的轮廓和像素中心点
    for (const auto &spot : hot_spots)
    {
        cv::drawContours(display_image, std::vector<std::vector<cv::Point>>{spot.contour_pixels}, -1, cv::Scalar(0, 255, 0), 1);
        cv::circle(display_image, spot.pixel_centroid, 3, cv::Scalar(0, 0, 255), -1);
    }

    int target_rank = 1;

    // 绘制每个喷洒目标的瞄准点和排名
    for (const auto &target : spray_targets)
    {
        cv::circle(display_image, target.final_pixel_aim_point, 8, cv::Scalar(255, 0, 255), 2);
        cv::putText(display_image, "T" + std::to_string(target_rank++),
                    target.final_pixel_aim_point + cv::Point2f(10, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    }
}