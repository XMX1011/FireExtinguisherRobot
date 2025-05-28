// src/main.cpp
#include "vision_processing.h" // 主要的视觉处理函数
#include "utils.h"             // 结构体, 全局参数, 辅助函数声明
#include <iostream>
#include <opencv2/opencv.hpp> // 确保包含OpenCV

// --- 全局相机参数定义 (在 utils.h 中声明为 extern) ---
// // 这些值理想情况下从 config/camera_params.xml 加载
// cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3, 3) << 500.0, 0.0, 238.0,
//                          0.0, 500.0, 236.0,
//                          0.0, 0.0, 1.0);
// cv::Mat DIST_COEFFS = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat CAMERA_MATRIX;
cv::Mat DIST_COEFFS;
float temperature_threshold;
double  min_hotspot_area_pixels;
float max_grouping_distance_meters;

// --- utils.h 中声明的辅助函数实现 ---

// ! 具体距离、角度等内容的坐标转换需要之后根据实际情况实现
/**
 * @brief 将像素坐标转换为近似的世界坐标。
 *
 * 该函数根据相机内参矩阵和指定的平面距离，将图像中的二维像素坐标映射到三维世界坐标。
 * 假设世界坐标系的Z轴与相机光轴对齐，并且平面距离表示目标点到相机平面的距离。
 *
 * @param pixel_coord 输入的像素坐标 (x, y)。
 * @param cam_matrix 相机内参矩阵，3x3矩阵，包含焦距和主点偏移。
 * @param distance_to_plane 目标点到相机平面的距离（沿Z轴）。
 * @return cv::Point3f 转换后的三维世界坐标 (X, Y, Z)。
 */
cv::Point3f pixelToApproxWorld(const cv::Point2f &pixel_coord, const cv::Mat &cam_matrix, float distance_to_plane)
{
    // 如果相机内参矩阵为空或焦距为0，则返回默认值
    if (cam_matrix.empty() || cam_matrix.at<double>(0, 0) == 0)
    {
        return cv::Point3f(pixel_coord.x, pixel_coord.y, 0.0f);
    }

    // 提取相机内参矩阵中的焦距和主点偏移
    double fx = cam_matrix.at<double>(0, 0);
    double fy = cam_matrix.at<double>(1, 1);
    double cx = cam_matrix.at<double>(0, 2);
    double cy = cam_matrix.at<double>(1, 2);

    // 根据相机模型计算世界坐标
    double X = (pixel_coord.x - cx) * distance_to_plane / fx;
    double Y = (pixel_coord.y - cy) * distance_to_plane / fy;
    return cv::Point3f(static_cast<float>(X), static_cast<float>(Y), distance_to_plane);
}

/**
 * @brief 计算两个三维点之间的欧几里得距离。
 *
 * 该函数用于计算三维空间中两点之间的实际距离。如果任意一点的Z坐标为0，
 * 则认为输入无效，返回FLT_MAX以表示错误。
 *
 * @param p1 第一个三维点。
 * @param p2 第二个三维点。
 * @return float 两点之间的欧几里得距离。如果输入无效，返回FLT_MAX。
 */
float calculateRealWorldDistance(const cv::Point3f &p1, const cv::Point3f &p2)
{
    // 如果任意一点的Z坐标为0，返回最大浮点值表示无效
    if (p1.z == 0.0f || p2.z == 0.0f)
        return FLT_MAX;

    // 计算两点之间的欧几里得距离
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

// ! 这个作为初期测试使用，目前可以废弃或者删除
/**
 * @brief 生成模拟的温度矩阵。
 *
 * 该函数生成一个随机的温度矩阵，并在其中添加几个圆形热点区域。
 * 矩阵的值范围为20.0到40.0，热点区域的值更高，用于模拟异常温度分布。
 *
 * @param temp_matrix 输出的温度矩阵，类型为CV_32FC1。
 * @param rows 矩阵的行数。
 * @param cols 矩阵的列数。
 * @return bool 如果矩阵成功生成且非空，返回true；否则返回false。
 */
bool getSimulatedTemperatureMatrix(cv::Mat &temp_matrix, int rows, int cols)
{
    // 创建随机温度矩阵，值范围为20.0到40.0
    temp_matrix = cv::Mat(rows, cols, CV_32FC1);
    cv::randu(temp_matrix, cv::Scalar(20.0f), cv::Scalar(40.0f));

    // 在矩阵中添加多个圆形热点区域
    cv::circle(temp_matrix, cv::Point(cols / 4, rows / 3), 15, cv::Scalar(250.0f), -1);
    cv::circle(temp_matrix, cv::Point(cols / 4 + 30, rows / 3 + 20), 12, cv::Scalar(200.0f), -1);
    cv::circle(temp_matrix, cv::Point(cols * 3 / 4, rows / 2), 20, cv::Scalar(300.0f), -1);
    cv::circle(temp_matrix, cv::Point(cols / 2, rows * 3 / 4), 3, cv::Scalar(180.0f), -1);

    // 返回矩阵是否成功生成
    return !temp_matrix.empty();
}

// --- 配置加载函数 (示例) ---
/**
 * @brief 加载相机参数文件并解析相机内参矩阵和畸变系数。
 *
 * 该函数尝试从指定的文件中加载相机参数，包括相机内参矩阵和畸变系数。
 * 如果文件无法打开或读取失败，将输出错误信息并返回 false。
 *
 * @param filename 相机参数文件的路径（输入）。
 * @param cam_matrix 用于存储加载的相机内参矩阵（输出）。
 * @param dist_coeffs_out 用于存储加载的畸变系数矩阵（输出）。
 * @return bool 返回 true 表示加载成功，false 表示加载失败。
 */
bool loadCameraParameters(const std::string &filename, 
                            cv::Mat &cam_matrix, 
                            cv::Mat &dist_coeffs_out, 
                            float &temperature_threshold,
                            double &min_hotspot_area_pixels,
                            float &max_grouping_distance_meters)
{
    // 尝试打开指定的相机参数文件
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open camera parameters file: " << filename << std::endl;
        std::cerr << "Using default/hardcoded camera parameters." << std::endl;
        return false;
    }

    // 从文件中读取相机内参矩阵和畸变系数
    fs["camera_matrix"] >> cam_matrix;
    fs["distortion_coefficients"] >> dist_coeffs_out;
    fs["temperature_threshold"] >> temperature_threshold;
    fs["min_hotspot_area_pixels"] >> min_hotspot_area_pixels;
    fs["max_grouping_distance_meters"] >> max_grouping_distance_meters ; 
    fs.release();

    // 输出成功加载的信息
    std::cout << "Camera parameters loaded from " << filename << std::endl;
    return true;
}

// ! 下面的函数是作为使用网络图片的测试用函数，具体温度矩阵等待红外相机
/**
 * @brief 将 RGB 图像转换为温度矩阵。
 *
 * 该函数将输入的 RGB 图像转换为灰度图像，并将灰度值映射到指定的温度范围（例如 20°C 到 500°C）。
 * 转换后的温度矩阵以 CV_32FC1 格式存储。
 *
 * @param rgb_image 输入的 RGB 图像（输入）。
 * @param temperature_matrix 用于存储生成的温度矩阵（输出）。
 * @return bool 返回 true 表示转换成功，false 表示转换失败。
 */
bool convertRGBToTemperatureMatrix(const cv::Mat &rgb_image, cv::Mat &temperature_matrix)
{
    // 检查输入图像是否为空
    if (rgb_image.empty())
    {
        std::cerr << "Error: Input RGB image is empty." << std::endl;
        return false;
    }

    // 将 RGB 图像转换为灰度图像
    cv::Mat gray_image;
    cv::cvtColor(rgb_image, gray_image, cv::COLOR_BGR2GRAY);

    // 将灰度值线性映射到温度范围（20°C 到 500°C）
    gray_image.convertTo(temperature_matrix, CV_32FC1, (500.0f - 20.0f) / 255.0f, 20.0f);

    // 验证生成的温度矩阵是否有效
    if (temperature_matrix.empty() || temperature_matrix.type() != CV_32FC1)
    {
        std::cerr << "Error: Failed to generate valid temperature matrix." << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief 主函数，用于执行火灾检测与灭火目标定位的视觉处理流程。
 *
 * 该函数的主要功能包括：
 * 1. 加载输入图像并将其转换为温度矩阵；
 * 2. 加载相机参数（如果文件加载失败，则使用硬编码值）；
 * 3. 检测和过滤热点区域，并确定喷洒目标；
 * 4. 可视化处理结果并输出到控制台；
 * 5. 提供用户交互功能，按 'q' 或 ESC 键退出程序。
 *
 * @return int 返回 0 表示程序正常结束。
 */
int main(int argc, char **argv)
{
    if (argc != 2) // 检查命令行参数个数
    {
        std::cout << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }

    cv::Mat temperature_matrix; // 存储温度矩阵
    cv::Mat display_image;      // 用于显示的图像

    // 加载测试图像并将其转换为温度矩阵
    // cv::Mat image = cv::imread("../testImage/02.JPG");
    // 修改为从程序参数中获取图像路径
    cv::Mat image = cv::imread(argv[1]);
    convertRGBToTemperatureMatrix(image, temperature_matrix);

    // 假设帧的分辨率数值
    // 通过读取的图片的本身来修改frame尺寸
    int frame_rows = image.rows;
    int frame_cols = image.cols;

    // 尝试从文件加载相机参数
    // 注意：全局的 CAMERA_MATRIX 和 DIST_COEFFS 变量会被这个函数修改
    // 但是实际只有一个相机，并且从配置文件里读方便针对不同的相机进行修改
    std::string camera_params_file = "../config/camera_params.xml"; // config在项目根目录的上一级
    if (!loadCameraParameters(camera_params_file, CAMERA_MATRIX, DIST_COEFFS, temperature_threshold, min_hotspot_area_pixels, max_grouping_distance_meters))
    {
        // 如果加载失败，将使用在 main.cpp 顶部定义的硬编码值
        std::cout << "Failed to load camera parameters from " << camera_params_file << std::endl;
        std::cout << "Using hardcoded camera parameters instead." << std::endl;
    }

    // 打印程序启动信息
    std::cout << "Vision Processing for Fire Detection Started." << std::endl;
    std::cout << "Press 'q' or ESC to exit." << std::endl;

    // 主循环：持续处理温度矩阵并检测热点
    while (true)
    {
        // 在实际应用中，这里会调用相机SDK的函数
        // if (!getSimulatedTemperatureMatrix(temperature_matrix, frame_rows, frame_cols))
        // {
        //     std::cerr << "Error: Could not get temperature matrix." << std::endl;
        //     break;
        // }

        // 核心视觉处理步骤
        // 将全局的相机参数传递给处理函数
        std::vector<HotSpot> hot_spots = detectAndFilterHotspots(temperature_matrix, CAMERA_MATRIX, ASSUMED_DISTANCE_TO_FIRE_PLANE_METERS);
        std::vector<SprayTarget> spray_targets = determineSprayTargets(hot_spots, max_grouping_distance_meters);

        // 可视化处理结果
        cv::Mat normalized_temp;
        cv::normalize(temperature_matrix, normalized_temp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(normalized_temp, display_image, cv::COLORMAP_JET);
        visualizeResults(display_image, temperature_matrix, hot_spots, spray_targets);

        // 输出喷洒目标信息到控制台
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

        // 显示可视化结果并检查用户输入
        cv::imshow("Fire Detection Visual Output", display_image);
        char key = (char)cv::waitKey(500);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

    // 清理资源并打印程序终止信息
    cv::destroyAllWindows();
    std::cout << "Vision Processing Terminated." << std::endl;
    return 0;
}