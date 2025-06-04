// src/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string> // For std::string if needed later

/**
 * @brief 配置参数部分
 * 这些参数控制热点检测和分组的逻辑，建议在实际项目中从配置文件加载。
 * 现在已经修改为从配置文件xml中加载
 */
// const float FIRE_TEMPERATURE_THRESHOLD_CELSIUS = 210.0f; ///< 热点温度阈值（摄氏度），高于此值的区域被视为潜在热点。
// const double MIN_HOTSPOT_AREA_PIXELS = 30.0; ///< 热点的最小像素面积，小于该值的区域将被忽略。
// const float MAX_GROUPING_DISTANCE_METERS = 1.0f; ///< 热点分组的最大距离（米），用于合并相邻热点。
const float ASSUMED_DISTANCE_TO_FIRE_PLANE_METERS = 9.0f; ///< 假设的热点平面距离（米），用于近似世界坐标计算。

/**
 * @brief 相机内参声明
 * 这些参数应通过相机标定获得，并在初始化时从配置文件加载。
 */
extern cv::Mat CAMERA_MATRIX;              ///< 相机内参矩阵，包含焦距和主点偏移。
extern cv::Mat DIST_COEFFS;                ///< 相机畸变系数，用于校正图像畸变。
extern float temperature_threshold;        ///< 阈值，用于过滤温度低于阈值的热点。
extern double min_hotspot_area_pixels;     ///< 最小热点面积，用于过滤太小的热点。
extern float max_grouping_distance_meters; ///< 最大分组距离，用于将相似位置的热点分组。

extern float HFOV;                       ///< 相机水平视场角，用于计算像素坐标和世界坐标之间的转换。
extern float VFOV;                       ///< 相机垂直视场角，用于计算像素坐标和世界坐标之间的转换。
extern float Horizontal_angle_per_pixel; ///< 水平角度每像素，用于基于角度像素间关系进行坐标转换和运动指令
extern float Vertical_angle_per_pixel;   ///< 垂直角度每像素

/**
 * @brief HotSpot 结构体
 * 表示一个检测到的热点，包含像素坐标、近似世界坐标、面积、最高温度等信息。
 */
struct HotSpot
{
    int id;                                ///< 热点的唯一标识符，初始化为 -1。
    cv::Point2f pixel_centroid;            ///< 热点在图像中的像素中心坐标。
    cv::Point3f world_coord_approx;        ///< 热点的近似世界坐标（基于假设的平面距离）。
    double area_pixels;                    ///< 热点的像素面积。
    float max_temperature;                 ///< 热点区域内的最高温度。
    std::vector<cv::Point> contour_pixels; ///< 热点的轮廓像素点集合。
    bool grouped;                          ///< 标记该热点是否已被分组，默认为 false。

    /**
     * @brief 默认构造函数
     * 初始化热点的默认值。
     */
    HotSpot() : id(-1), area_pixels(0.0), max_temperature(0.0f), grouped(false) {}
};

/**
 * @brief SprayTarget 结构体
 * 表示一个喷洒目标，包含最终的瞄准点、相关热点 ID 和估计的严重性。
 */
struct SprayTarget
{
    int id;                                   ///< 喷洒目标的唯一标识符，初始化为 -1。
    cv::Point2f final_pixel_aim_point;        ///< 最终的像素瞄准点。
    cv::Point3f final_world_aim_point_approx; ///< 最终的世界坐标瞄准点（近似值）。
    std::vector<int> source_hotspot_ids;      ///< 构成该目标的相关热点 ID 列表。
    float estimated_severity;                 ///< 估计的严重性，用于优先级排序。

    /**
     * @brief 默认构造函数
     * 初始化喷洒目标的默认值。
     */
    SprayTarget() : id(-1), estimated_severity(0.0f) {}

    /**
     * @brief 比较运算符
     * 用于按严重性降序排序喷洒目标。
     *
     * @param other 另一个喷洒目标对象。
     * @return true 如果当前目标的严重性高于另一个目标。
     * @return false 如果当前目标的严重性低于或等于另一个目标。
     */
    bool operator<(const SprayTarget &other) const
    {
        return estimated_severity > other.estimated_severity;
    }
};

/**
 * @brief 定义球坐标结构体
 *
 * @param radius    距离原点的距离 (r)
 * @param azimuth   水平方向的角度 (θ)，单位为弧度或角度。
 * @param elevation 垂直方向的角度 (φ)，单位为弧度或角度
 */
struct SphericalCoordinate
{
    float radius;    // 距离原点的距离 (r)
    float azimuth;   // 水平方向的角度 (θ)，单位为弧度或角度
    float elevation; // 垂直方向的角度 (φ)，单位为弧度或角度

    // 构造函数
    SphericalCoordinate(float r = 0.0f, float theta = 0.0f, float phi = 0.0f)
        : radius(r), azimuth(theta), elevation(phi) {}
};

/**
 * @brief 将像素坐标转换为近似的世界坐标
 * 使用相机内参和假设的平面距离计算像素坐标对应的世界坐标。
 *
 * @param pixel_coord 输入的像素坐标 (x, y)。
 * @param cam_matrix 相机内参矩阵。
 * @param distance_to_plane 假设的平面距离（米）。
 * @return cv::Point3f 近似的世界坐标 (X, Y, Z)。
 */
cv::Point3f pixelToApproxWorld(const cv::Point2f &pixel_coord, const cv::Mat &cam_matrix, float distance_to_plane);

/**
 * @brief 计算两个世界坐标点之间的实际距离
 * 使用欧几里得距离公式计算两点之间的距离。
 * TODO: 转化为使用球坐标系的距离计算方式
 *
 * @param p1 第一个世界坐标点 (X1, Y1, Z1)。
 * @param p2 第二个世界坐标点 (X2, Y2, Z2)。
 * @return float 两点之间的实际距离（米）。
 */
float calculateRealWorldDistance(const cv::Point3f &p1, const cv::Point3f &p2);

/**
 * @brief 模拟从 SDK 获取温度矩阵
 * 生成一个模拟的温度矩阵，用于测试和开发。实际项目中应替换为真实的 SDK 调用。
 *
 * @param temp_matrix 输出的温度矩阵，大小为 rows x cols。
 * @param rows 温度矩阵的行数。
 * @param cols 温度矩阵的列数。
 * @return bool 返回 true 表示成功生成温度矩阵，false 表示失败。
 */
bool getSimulatedTemperatureMatrix(cv::Mat &temp_matrix, int rows, int cols);


/**
 * @brief 将笛卡尔坐标转换为球坐标
 * 
 * @param point 输入的笛卡尔坐标点 (x, y, z)
 * @return SphericalCoordinate 转换后的球坐标
 */
SphericalCoordinate cartesianToSpherical(const cv::Point3f &point);
#endif // UTILS_H