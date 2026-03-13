#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

// 特征数据结构定义
struct PointFeature {
    Eigen::Vector3d coord;          // 点特征坐标 (Xi)
    Eigen::Vector3d velocity;       // 点运动速度向量 (vi)
    double corner_response;         // 角点响应值（用于置信度计算）
    double match_stability;         // 匹配稳定性（用于置信度计算）
    double confidence;              // 点特征置信度 (α_k 相关)
};

struct LineFeature {
    Eigen::Vector3d start;          // 线特征起点 (Lj,start)
    Eigen::Vector3d end;            // 线特征终点 (Lj,end)
    double dir_angle;               // 方向角 (θj,t)
    double length_stability;        // 长度稳定性（用于置信度计算）
    double dir_consistency;         // 方向一致性（用于置信度计算）
    double confidence;              // 线特征置信度
};

// 约束项类型枚举
enum ConstraintType {
    PROJ_ERROR = 0,    // 点线投影误差约束 (Eproj)
    DIR_ANGLE = 1,     // 方向角一致性约束 (Edir)
    POSE_CONSISTENCY = 2,// 全局位姿一致性约束 (Epose)
    SCALE_INVARIANCE = 3 // 尺度不变性约束（简化实现）
};

// 分层约束模型核心类
class HierarchicalFeatureConstraint {
public:
    HierarchicalFeatureConstraint() {
        // 初始化参数
        sigma0 = 0.01;          // 基线标准差 σ0
        lambda = 0.05;          // 辅助约束系数 λ
        alpha = {0.8, 0.7, 0.9, 0.6}; // 各约束项优先级系数 αk
        max_iter = 10;          // 最大迭代次数
    }

    // 核心优化函数：分层迭代求解位姿和特征约束
    Eigen::Matrix4d optimize(const std::vector<PointFeature>& points_t, 
                             const std::vector<LineFeature>& lines_t,
                             const std::vector<PointFeature>& points_t1,
                             const std::vector<LineFeature>& lines_t1,
                             Eigen::Matrix4d T_init) {
        Eigen::Matrix4d T = T_init; // 初始位姿 Tt (t到t-1的变换矩阵)
        std::vector<double> weights = {1.0, 1.0, 1.0, 1.0}; // 初始权重 ωk
        
        // 1. 初始优化阶段：固定权重，高斯牛顿求解初始位姿
        for (int iter = 0; iter < max_iter / 2; ++iter) {
            T = gaussNewtonOptimization(points_t, lines_t, points_t1, lines_t1, T, weights);
        }

        // 2. 权重更新阶段：基于初始优化结果更新动态权重
        updateDynamicWeights(points_t, lines_t, points_t1, lines_t1, T, weights);

        // 3. 精细优化阶段：使用更新后的权重二次优化
        for (int iter = 0; iter < max_iter / 2; ++iter) {
            T = gaussNewtonOptimization(points_t, lines_t, points_t1, lines_t1, T, weights);
        }

        return T;
    }

private:
    // 点线投影误差约束计算 Eproj
    double calcProjError(const PointFeature& point, const LineFeature& line) {
        Eigen::Vector3d Xi = point.coord;
        Eigen::Vector3d Lj_start = line.start;
        Eigen::Vector3d Lj_end = line.end;
        
        // 公式7: Eproj = ||(Xi-Lj_start) × (Xi-Lj_end)||² / ||Lj_end-Lj_start||
        Eigen::Vector3d cross = (Xi - Lj_start).cross(Xi - Lj_end);
        double numerator = cross.squaredNorm();
        double denominator = (Lj_end - Lj_start).norm();
        return denominator > 1e-8 ? numerator / denominator : 0.0;
    }

    // 方向角一致性约束计算 Edir
    double calcDirAngleError(const std::vector<PointFeature>& points, 
                             const std::vector<LineFeature>& lines_t,
                             const std::vector<LineFeature>& lines_t1) {
        double line_angle_error = 0.0;
        // 线特征方向角变化项：Σ(θj,t - θj,t-1)²
        for (size_t j = 0; j < lines_t.size() && j < lines_t1.size(); ++j) {
            double d_theta = lines_t[j].dir_angle - lines_t1[j].dir_angle;
            line_angle_error += d_theta * d_theta;
        }

        double point_line_dir_error = 0.0;
        // 点运动方向与线方向一致性项：Σ(1 - cos(vi, dj))
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < lines_t.size(); ++j) {
                Eigen::Vector3d vi = points[i].velocity.normalized();
                Eigen::Vector3d dj = (lines_t[j].end - lines_t[j].start).normalized();
                double cos_theta = vi.dot(dj);
                point_line_dir_error += 1.0 - cos_theta;
            }
        }

        return line_angle_error + point_line_dir_error; // 公式8
    }

    // 全局位姿一致性约束计算 Epose
    double calcPoseError(const std::vector<PointFeature>& points_t,
                         const std::vector<LineFeature>& lines_t,
                         const std::vector<PointFeature>& points_t1,
                         const std::vector<LineFeature>& lines_t1,
                         const Eigen::Matrix4d& T) {
        double point_pose_error = 0.0;
        // 点特征位姿变换误差：Σ||Tt*Xit - Xit-1||²
        for (size_t i = 0; i < points_t.size() && i < points_t1.size(); ++i) {
            Eigen::Vector3d Xt_transformed = T.block<3,3>(0,0) * points_t[i].coord + T.block<3,1>(0,3);
            point_pose_error += (Xt_transformed - points_t1[i].coord).squaredNorm();
        }

        double line_pose_error = 0.0;
        // 线特征位姿变换误差：Σ||Tt*Ljt - Lit-1||²（简化为端点误差和）
        for (size_t j = 0; j < lines_t.size() && j < lines_t1.size(); ++j) {
            Eigen::Vector3d L_start_transformed = T.block<3,3>(0,0) * lines_t[j].start + T.block<3,1>(0,3);
            Eigen::Vector3d L_end_transformed = T.block<3,3>(0,0) * lines_t[j].end + T.block<3,1>(0,3);
            line_pose_error += (L_start_transformed - lines_t1[j].start).squaredNorm();
            line_pose_error += (L_end_transformed - lines_t1[j].end).squaredNorm();
        }

        return point_pose_error + line_pose_error; // 公式9
    }

    // 计算约束项残差标准差 σk
    double calcResidualStd(const std::vector<double>& residuals) {
        if (residuals.empty()) return 0.0;
        double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
        double sq_sum = 0.0;
        for (double r : residuals) {
            sq_sum += (r - mean) * (r - mean);
        }
        return std::sqrt(sq_sum / residuals.size());
    }

    // 更新动态权重 ωk = exp(-σk/σ0) * αk （公式10）
    void updateDynamicWeights(const std::vector<PointFeature>& points_t,
                              const std::vector<LineFeature>& lines_t,
                              const std::vector<PointFeature>& points_t1,
                              const std::vector<LineFeature>& lines_t1,
                              const Eigen::Matrix4d& T,
                              std::vector<double>& weights) {
        // 1. 计算各约束项残差
        std::vector<double> proj_residuals, dir_residuals, pose_residuals;
        for (size_t i = 0; i < points_t.size(); ++i) {
            for (size_t j = 0; j < lines_t.size(); ++j) {
                proj_residuals.push_back(calcProjError(points_t[i], lines_t[j]));
            }
        }
        dir_residuals.push_back(calcDirAngleError(points_t, lines_t, lines_t1));
        pose_residuals.push_back(calcPoseError(points_t, lines_t, points_t1, lines_t1, T));

        // 2. 计算各约束项残差标准差 σk
        double sigma_proj = calcResidualStd(proj_residuals);
        double sigma_dir = calcResidualStd(dir_residuals);
        double sigma_pose = calcResidualStd(pose_residuals);

        // 3. 更新权重
        weights[PROJ_ERROR] = exp(-sigma_proj / sigma0) * alpha[PROJ_ERROR];
        weights[DIR_ANGLE] = exp(-sigma_dir / sigma0) * alpha[DIR_ANGLE];
        weights[POSE_CONSISTENCY] = exp(-sigma_pose / sigma0) * alpha[POSE_CONSISTENCY];
        weights[SCALE_INVARIANCE] = 0.7; // 简化：尺度不变性约束权重固定（可扩展）
    }

    // 高斯牛顿法求解位姿优化（公式11 核心）
    Eigen::Matrix4d gaussNewtonOptimization(const std::vector<PointFeature>& points_t,
                                            const std::vector<LineFeature>& lines_t,
                                            const std::vector<PointFeature>& points_t1,
                                            const std::vector<LineFeature>& lines_t1,
                                            Eigen::Matrix4d T,
                                            const std::vector<double>& weights) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        // 遍历所有特征，构建高斯牛顿法的H和b
        for (size_t i = 0; i < points_t.size(); ++i) {
            for (size_t j = 0; j < lines_t.size(); ++j) {
                // 1. 投影误差约束的雅可比（简化：仅对位姿求导）
                Eigen::Matrix<double, 1, 6> J_proj;
                // 此处省略雅可比的详细计算（可基于李代数推导）
                double e_proj = calcProjError(points_t[i], lines_t[j]);
                H += weights[PROJ_ERROR] * J_proj.transpose() * J_proj;
                b += weights[PROJ_ERROR] * J_proj.transpose() * e_proj;

                // 2. 方向角约束的雅可比（简化）
                Eigen::Matrix<double, 1, 6> J_dir;
                double e_dir = calcDirAngleError({points_t[i]}, {lines_t[j]}, {lines_t1[j]});
                H += weights[DIR_ANGLE] * J_dir.transpose() * J_dir;
                b += weights[DIR_ANGLE] * J_dir.transpose() * e_dir;
            }
        }

        // 3. 全局位姿约束的雅可比
        Eigen::Matrix<double, 1, 6> J_pose;
        double e_pose = calcPoseError(points_t, lines_t, points_t1, lines_t1, T);
        H += weights[POSE_CONSISTENCY] * J_pose.transpose() * J_pose;
        b += weights[POSE_CONSISTENCY] * J_pose.transpose() * e_pose;

        // 4. 求解增量 Δx = H^{-1} * b
        Eigen::Matrix<double, 6, 1> delta_x = H.inverse() * b;

        // 5. 更新位姿 T（基于李代数更新）
        Eigen::Matrix3d R = T.block<3,3>(0,0);
        Eigen::Vector3d t = T.block<3,1>(0,3);
        Eigen::Matrix3d delta_R = Eigen::AngleAxisd(delta_x.head(3).norm(), delta_x.head(3).normalized()).toRotationMatrix();
        Eigen::Vector3d delta_t = delta_x.tail(3);
        T.block<3,3>(0,0) = delta_R * R;
        T.block<3,1>(0,3) = delta_R * t + delta_t;

        return T;
    }

private:
    double sigma0;          // 基线标准差
    double lambda;          // 辅助约束系数
    std::vector<double> alpha; // 约束项优先级系数
    int max_iter;           // 优化迭代次数
};

// 测试代码
int main() {
    // 1. 初始化特征数据
    std::vector<PointFeature> points_t, points_t1;
    std::vector<LineFeature> lines_t, lines_t1;

    // 添加测试点特征
    PointFeature p1;
    p1.coord = Eigen::Vector3d(0.1, 0.2, 0.0);
    p1.velocity = Eigen::Vector3d(0.01, 0.02, 0.0);
    p1.corner_response = 0.8;
    p1.match_stability = 0.9;
    p1.confidence = 0.85;
    points_t.push_back(p1);

    // 添加测试线特征
    LineFeature l1;
    l1.start = Eigen::Vector3d(0.0, 0.0, 0.0);
    l1.end = Eigen::Vector3d(1.0, 1.0, 0.0);
    l1.dir_angle = M_PI_4;
    l1.length_stability = 0.9;
    l1.dir_consistency = 0.88;
    l1.confidence = 0.82;
    lines_t.push_back(l1);

    // t-1帧特征（简化：与t帧一致）
    points_t1 = points_t;
    lines_t1 = lines_t;
    lines_t1[0].dir_angle = M_PI_4 + 0.01; // 微小角度变化

    // 2. 初始化位姿（单位矩阵）
    Eigen::Matrix4d T_init = Eigen::Matrix4d::Identity();

    // 3. 运行分层约束优化
    HierarchicalFeatureConstraint constraint_model;
    Eigen::Matrix4d T_optimized = constraint_model.optimize(points_t, lines_t, points_t1, lines_t1, T_init);

    // 4. 输出结果
    std::cout << "优化后的位姿矩阵：" << std::endl;
    std::cout << T_optimized << std::endl;

    return 0;
}