#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

// ===================== 数据结构定义 =====================
// YOLOv5 检测的动态目标边界框
struct BoundingBox {
    int x1, y1, x2, y2;       // 边界框坐标
    std::string category;     // 目标类别（如car, person）
    float confidence;         // 检测置信度
};

// NeRF 生成的关键语义特征
struct NeRFSemanticFeature {
    Eigen::Vector3d contour_center;   // 目标几何轮廓中心
    Eigen::Vector3d motion_trend;     // 运动趋势特征（相邻帧建模差异）
    std::map<std::string, float> category_prob; // 类别概率分布（如car:0.95, person:0.05）
    Eigen::Matrix4d obj_pose;         // 目标3D位姿
};

// SLAM 前端输出的初始位姿
struct SLAMPose {
    Eigen::Matrix4d pose;      // 相机位姿 (Tcw)
    double timestamp;          // 时间戳
    bool is_keyframe;          // 是否为关键帧
};

// ===================== 核心类：NeRF-SLAM 协同优化器 =====================
class NeRFSLAMOptimizer {
public:
    NeRFSLAMOptimizer() {
        // 初始化联合优化权重（对应公式14的λ1-λ4）
        lambda1 = 1.0;  // 相机位姿约束权重
        lambda2 = 0.8;  // NeRF 3D模型约束权重
        lambda3 = 0.9;  // 目标3D位置约束权重
        lambda4 = 0.7;  // 语义约束损失权重
        ms = 1000;      // 光度约束采样像素数（公式17的Ms）
    }

    // 核心接口：执行NeRF-SLAM协同优化
    // 输入：SLAM初始位姿、YOLO检测结果、原始图像、参考NeRF模型
    // 输出：优化后的相机位姿
    Eigen::Matrix4d optimize(
        const SLAMPose& slam_pose,
        const std::vector<BoundingBox>& dynamic_boxes,
        const cv::Mat& image,
        const NeRFSemanticFeature& ref_nerf_feature) {
        
        // 步骤1：轻量化NeRF建模（仅对动态目标区域建模，稀疏时间采样）
        NeRFSemanticFeature curr_nerf_feature = lightweightNeRFModeling(image, dynamic_boxes, slam_pose);
        
        // 步骤2：计算各类约束损失
        double E_geo = calculateGeometricConstraint(slam_pose.pose, curr_nerf_feature, ref_nerf_feature); // 公式15
        double L_sem = calculateSemanticConstraint(curr_nerf_feature, ref_nerf_feature);                 // 公式16
        double L_photo = calculatePhotometricLoss(image, curr_nerf_feature, dynamic_boxes);              // 公式17
        
        // 步骤3：构建联合优化目标函数（公式14）并求解
        Eigen::Matrix4d optimized_pose = solveJointOptimization(
            slam_pose.pose, curr_nerf_feature, ref_nerf_feature, E_geo, L_sem, L_photo);
        
        return optimized_pose;
    }

private:
    // ===================== 子模块1：轻量化NeRF建模（按需提取+稀疏采样） =====================
    NeRFSemanticFeature lightweightNeRFModeling(
        const cv::Mat& image,
        const std::vector<BoundingBox>& dynamic_boxes,
        const SLAMPose& slam_pose) {
        
        NeRFSemanticFeature nerf_feature;
        
        // 1. 仅对动态目标区域进行有限隐式3D建模（跳过纹理细节恢复）
        for (const auto& box : dynamic_boxes) {
            // 裁剪动态目标区域（避免全图建模）
            cv::Mat roi = image(cv::Rect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1));
            if (roi.empty()) continue;

            // 2. 稀疏射线采样：仅采样动态目标核心区域
            std::vector<Eigen::Vector3d> rays = sampleCoreRegionRays(box, slam_pose.pose);
            
            // 3. 提取关键语义特征（几何轮廓、运动趋势、类别概率）
            nerf_feature.contour_center = calculateContourCenter(roi, box, slam_pose.pose);
            nerf_feature.motion_trend = calculateMotionTrend(box, slam_pose);
            nerf_feature.category_prob = calculateCategoryProb(box);
            nerf_feature.obj_pose = estimateObjectPose(nerf_feature.contour_center, slam_pose.pose);
        }

        // 4. 关键帧策略：仅在关键帧更新动态目标建模，复用历史信息
        if (!slam_pose.is_keyframe) {
            nerf_feature = reuseHistoricalFeature(nerf_feature);
        }

        return nerf_feature;
    }

    // 稀疏射线采样：仅采样动态目标核心区域（避免无效计算）
    std::vector<Eigen::Vector3d> sampleCoreRegionRays(
        const BoundingBox& box,
        const Eigen::Matrix4d& cam_pose) {
        
        std::vector<Eigen::Vector3d> rays;
        // 计算动态目标中心像素
        int cx = (box.x1 + box.x2) / 2;
        int cy = (box.y1 + box.y2) / 2;
        // 仅采样中心区域5x5像素的射线（稀疏采样）
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                int x = cx + dx;
                int y = cy + dy;
                if (x < 0 || x >= 640 || y < 0 || y >= 480) continue; // 假设图像分辨率640x480
                
                // 将像素转换为相机坐标系射线（简化版）
                Eigen::Vector3d ray_dir(x - 320, y - 240, 1.0); // 内参假设：cx=320, cy=240, fx=fy=1.0
                ray_dir = ray_dir.normalized();
                // 转换到世界坐标系
                Eigen::Vector3d ray_world = cam_pose.block<3,3>(0,0) * ray_dir;
                rays.push_back(ray_world);
            }
        }
        return rays;
    }

    // 提取目标几何轮廓中心（区分动态/静态边界）
    Eigen::Vector3d calculateContourCenter(
        const cv::Mat& roi,
        const BoundingBox& box,
        const Eigen::Matrix4d& cam_pose) {
        
        // 简化实现：基于ROI的质心计算轮廓中心
        cv::Moments moments = cv::moments(roi);
        if (moments.m00 < 1e-6) {
            return Eigen::Vector3d(0, 0, 0);
        }
        int cx_roi = static_cast<int>(moments.m10 / moments.m00);
        int cy_roi = static_cast<int>(moments.m01 / moments.m00);
        int cx_world = box.x1 + cx_roi;
        int cy_world = box.y1 + cy_roi;
        
        // 像素坐标转世界坐标系（简化深度假设：z=5.0）
        Eigen::Vector3d pixel_coord(cx_world, cy_world, 5.0);
        return cam_pose.block<3,3>(0,0) * pixel_coord + cam_pose.block<3,1>(0,3);
    }

    // 计算运动趋势特征（相邻帧NeRF建模差异）
    Eigen::Vector3d calculateMotionTrend(
        const BoundingBox& box,
        const SLAMPose& slam_pose) {
        
        // 简化实现：基于关键帧间的边界框位移计算运动趋势
        static Eigen::Vector3d last_frame_center(0, 0, 0);
        Eigen::Vector3d curr_center((box.x1+box.x2)/2.0, (box.y1+box.y2)/2.0, 5.0);
        
        if (slam_pose.is_keyframe) {
            Eigen::Vector3d motion = curr_center - last_frame_center;
            last_frame_center = curr_center;
            return motion;
        }
        return Eigen::Vector3d(0, 0, 0);
    }

    // 计算目标类别概率分布（无需像素级标注，仅类别级约束）
    std::map<std::string, float> calculateCategoryProb(const BoundingBox& box) {
        std::map<std::string, float> prob;
        // 简化实现：基于YOLO检测结果生成概率分布
        prob[box.category] = box.confidence;
        // 其他类别概率设为小值
        for (const auto& cat : {"car", "person", "bike", "bus"}) {
            if (cat != box.category) {
                prob[cat] = (1.0 - box.confidence) / 3.0;
            }
        }
        return prob;
    }

    // 估计目标3D位姿
    Eigen::Matrix4d estimateObjectPose(
        const Eigen::Vector3d& contour_center,
        const Eigen::Matrix4d& cam_pose) {
        Eigen::Matrix4d obj_pose = Eigen::Matrix4d::Identity();
        obj_pose.block<3,1>(0,3) = contour_center;
        return obj_pose;
    }

    // 复用历史NeRF建模信息（非关键帧时）
    NeRFSemanticFeature reuseHistoricalFeature(const NeRFSemanticFeature& curr) {
        static NeRFSemanticFeature last_nerf_feature;
        NeRFSemanticFeature reused = curr;
        // 复用运动趋势和类别概率（减少计算）
        reused.motion_trend = 0.7 * last_nerf_feature.motion_trend + 0.3 * curr.motion_trend;
        reused.category_prob = last_nerf_feature.category_prob;
        last_nerf_feature = reused;
        return reused;
    }

    // ===================== 子模块2：约束损失计算 =====================
    // 计算几何约束损失 E_geo（公式15）
    double calculateGeometricConstraint(
        const Eigen::Matrix4d& slam_pose,
        const NeRFSemanticFeature& curr_nerf,
        const NeRFSemanticFeature& ref_nerf) {
        
        // 公式15: Egeo = Σ||Pi-Pj||² + Σ||Xkt-Xkt_ref||²
        double pose_error = (slam_pose - ref_nerf.obj_pose).squaredNorm();
        double pos_error = (curr_nerf.contour_center - ref_nerf.contour_center).squaredNorm();
        return pose_error + pos_error;
    }

    // 计算语义约束损失 ℒ(Ck,Ck_ref)（公式16，交叉熵）
    double calculateSemanticConstraint(
        const NeRFSemanticFeature& curr_nerf,
        const NeRFSemanticFeature& ref_nerf) {
        
        double cross_entropy = 0.0;
        // 公式16: ℒ = -Σ[P(Ck=c) * log(P_ref(Ck=c))]
        for (const auto& [category, prob] : curr_nerf.category_prob) {
            if (ref_nerf.category_prob.find(category) == ref_nerf.category_prob.end()) {
                continue;
            }
            float ref_prob = ref_nerf.category_prob.at(category);
            if (ref_prob < 1e-6) ref_prob = 1e-6; // 避免log(0)
            cross_entropy -= prob * std::log(ref_prob);
        }
        return cross_entropy;
    }

    // 计算光度约束损失 Lc（公式17）：仅采样静态区域像素
    double calculatePhotometricLoss(
        const cv::Mat& image,
        const NeRFSemanticFeature& nerf_feature,
        const std::vector<BoundingBox>& dynamic_boxes) {
        
        double loss = 0.0;
        int sampled = 0;
        
        // 仅采样静态区域像素（排除动态目标边界框）
        while (sampled < ms) {
            // 随机采样像素
            int x = rand() % image.cols;
            int y = rand() % image.rows;
            
            // 检查是否在动态目标区域内，若是则跳过
            bool is_dynamic = false;
            for (const auto& box : dynamic_boxes) {
                if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {
                    is_dynamic = true;
                    break;
                }
            }
            if (is_dynamic) continue;
            
            // 计算原始像素值与NeRF渲染像素值的差（公式17: |Ii - Ii'|）
            cv::Vec3b Ii = image.at<cv::Vec3b>(y, x);
            cv::Vec3b Ii_prime = renderPixelByNeRF(x, y, nerf_feature); // NeRF渲染像素
            loss += std::abs(Ii[0] - Ii_prime[0]) + 
                    std::abs(Ii[1] - Ii_prime[1]) + 
                    std::abs(Ii[2] - Ii_prime[2]);
            sampled++;
        }
        
        // 公式17: Lc = (1/Ms) * Σ|Ii - Ii'|
        return loss / ms;
    }

    // NeRF渲染指定像素（简化实现）
    cv::Vec3b renderPixelByNeRF(int x, int y, const NeRFSemanticFeature& nerf_feature) {
        // 简化：返回静态背景的平均颜色（实际需NeRF射线追踪）
        return cv::Vec3b(128, 128, 128);
    }

    // ===================== 子模块3：联合优化求解（高斯牛顿法） =====================
    Eigen::Matrix4d solveJointOptimization(
        const Eigen::Matrix4d& init_pose,
        const NeRFSemanticFeature& curr_nerf,
        const NeRFSemanticFeature& ref_nerf,
        double E_geo,
        double L_sem,
        double L_photo) {
        
        Eigen::Matrix4d optimized_pose = init_pose;
        const int max_iter = 10; // 高斯牛顿迭代次数
        const double eps = 1e-6; // 收敛阈值

        for (int iter = 0; iter < max_iter; ++iter) {
            // 构建联合优化目标函数（公式14）
            // min [Egeo + λ1||Pi-Pj||² + λ2||Mk-Mk_ref||² + λ3||Xkt-Xkt_ref||² + λ4ℒ(Ck,Ck_ref)]
            
            // 1. 计算目标函数值
            double obj_value = E_geo + 
                               lambda1 * (optimized_pose - ref_nerf.obj_pose).squaredNorm() +
                               lambda2 * (curr_nerf.obj_pose - ref_nerf.obj_pose).squaredNorm() +
                               lambda3 * (curr_nerf.contour_center - ref_nerf.contour_center).squaredNorm() +
                               lambda4 * L_sem + L_photo;
            
            // 2. 计算梯度（简化：数值梯度）
            Eigen::Matrix<double, 6, 1> grad = computeNumericGradient(optimized_pose, curr_nerf, ref_nerf, E_geo, L_sem, L_photo);
            
            // 3. 计算海森矩阵（简化：对角矩阵）
            Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Identity();
            
            // 4. 求解增量 Δx = H^-1 * grad
            Eigen::Matrix<double, 6, 1> delta = hessian.inverse() * grad;
            
            // 5. 更新位姿（基于李代数更新）
            optimized_pose = updatePose(optimized_pose, delta);
            
            // 6. 检查收敛
            if (delta.norm() < eps) {
                break;
            }
        }

        return optimized_pose;
    }

    // 数值法计算梯度（简化实现）
    Eigen::Matrix<double, 6, 1> computeNumericGradient(
        const Eigen::Matrix4d& pose,
        const NeRFSemanticFeature& curr_nerf,
        const NeRFSemanticFeature& ref_nerf,
        double E_geo,
        double L_sem,
        double L_photo) {
        
        Eigen::Matrix<double, 6, 1> grad;
        const double h = 1e-4; // 数值微分步长
        
        for (int i = 0; i < 6; ++i) {
            // 扰动位姿
            Eigen::Matrix4d pose_plus = pose;
            Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
            delta(i) = h;
            pose_plus = updatePose(pose, delta);
            
            // 计算扰动后的目标函数值
            double obj_plus = E_geo + 
                              lambda1 * (pose_plus - ref_nerf.obj_pose).squaredNorm() +
                              lambda2 * (curr_nerf.obj_pose - ref_nerf.obj_pose).squaredNorm() +
                              lambda3 * (curr_nerf.contour_center - ref_nerf.contour_center).squaredNorm() +
                              lambda4 * L_sem;
            
            // 原目标函数值
            double obj_original = E_geo + 
                                  lambda1 * (pose - ref_nerf.obj_pose).squaredNorm() +
                                  lambda2 * (curr_nerf.obj_pose - ref_nerf.obj_pose).squaredNorm() +
                                  lambda3 * (curr_nerf.contour_center - ref_nerf.contour_center).squaredNorm() +
                                  lambda4 * L_sem;
            
            // 数值梯度：(f(x+h) - f(x))/h
            grad(i) = (obj_plus - obj_original) / h;
        }
        
        return grad;
    }

    // 基于李代数更新位姿（简化版）
    Eigen::Matrix4d updatePose(const Eigen::Matrix4d& pose, const Eigen::Matrix<double, 6, 1>& delta) {
        Eigen::Matrix4d new_pose = pose;
        // delta前3维：旋转增量（角轴），后3维：平移增量
        Eigen::Vector3d rot_delta = delta.head(3);
        Eigen::Vector3d trans_delta = delta.tail(3);
        
        // 更新旋转
        Eigen::Matrix3d R = pose.block<3,3>(0,0);
        Eigen::Matrix3d delta_R = Eigen::AngleAxisd(rot_delta.norm(), rot_delta.normalized()).toRotationMatrix();
        new_pose.block<3,3>(0,0) = delta_R * R;
        
        // 更新平移
        new_pose.block<3,1>(0,3) += trans_delta;
        
        return new_pose;
    }

private:
    // 联合优化权重参数
    double lambda1, lambda2, lambda3, lambda4;
    int ms; // 光度约束采样像素数
};

// ===================== 测试代码 =====================
int main() {
    try {
        // 1. 初始化测试数据
        // SLAM初始位姿（单位矩阵）
        SLAMPose slam_pose;
        slam_pose.pose = Eigen::Matrix4d::Identity();
        slam_pose.timestamp = 1620000000.0;
        slam_pose.is_keyframe = true;

        // YOLOv5检测的动态目标
        BoundingBox box;
        box.x1 = 100, box.y1 = 200, box.x2 = 300, box.y2 = 400;
        box.category = "car";
        box.confidence = 0.95;
        std::vector<BoundingBox> dynamic_boxes = {box};

        // 原始图像（640x480，3通道）
        cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);

        // 参考NeRF语义特征
        NeRFSemanticFeature ref_nerf;
        ref_nerf.contour_center = Eigen::Vector3d(2.0, 3.0, 5.0);
        ref_nerf.motion_trend = Eigen::Vector3d(0.1, 0.0, 0.0);
        ref_nerf.category_prob = {{"car", 0.9}, {"person", 0.1}};
        ref_nerf.obj_pose = Eigen::Matrix4d::Identity();

        // 2. 创建优化器并执行优化
        NeRFSLAMOptimizer optimizer;
        Eigen::Matrix4d optimized_pose = optimizer.optimize(slam_pose, dynamic_boxes, image, ref_nerf);

        // 3. 输出结果
        std::cout << "优化后的相机位姿：" << std::endl;
        std::cout << optimized_pose << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << std::endl;
        return 1;
    }
    return 0;
}