# GDN-SLAM
GDN-SLAM 是一款面向复杂动态环境的高精度视觉SLAM系统，针对动态目标干扰、低纹理场景、运动模糊等实际场景问题进行了深度优化，在保持 ORB-SLAM2 核心框架稳定性的基础上，融合多特征约束、动态目标检测与语义优化策略，显著提升动态场景下的定位精度与鲁棒性。

# \## 核心特性
- 🎯 **动态特征点剔除**：基于 YOLOv8 目标检测+几何一致性约束，精准过滤动态目标特征点，降低运动干扰；
- 📈 **多特征融合优化**：融合点/线特征分层几何约束，解决低纹理区域特征匹配歧义问题；
- 🚀 **轻量化语义增强**：引入 NeRF 轻量化语义约束，无需精细纹理重建，仅提取关键语义特征辅助位姿优化；
- ⚡ **实时性优化**：稀疏时间采样+动态自适应权重策略，在保证精度的前提下控制计算开销；
- 🧩 **兼容ORB-SLAM2生态**：完全兼容 ORB-SLAM2 的数据集接口、配置文件格式，支持快速迁移。



\## 环境依赖

\### 基础依赖（必须） 

- 操作系统：Ubuntu 18.04/20.04 
- 编译器：GCC 5.4+ (支持 C++11/14) - ROS：Melodic (Ubuntu 18.04) / Noetic (Ubuntu 20.04) 
- 核心库：  
  - Pangolin ≥ 0.6 (可视化)  
  - OpenCV ≥ 3.4 (建议 4.5+)  
  -  Eigen3 ≥ 3.3.7  
  -  g2o (源码编译，已内置修改版)  
  - DBoW2 (词袋模型，已内置)



\### 扩展依赖（动态检测/语义优化）

- YOLOv5 (动态目标检测)
- PyTorch ≥ 1.10 (NeRF 轻量化语义建模，可选)

\## 环境配置

\### 1. 基础依赖安装

\```bash

\# 1. 安装系统依赖

```
sudo apt-get update && sudo apt-get install -y \    
	build-essential cmake git libgtk2.0-dev \    
	libboost-all-dev libssl-dev libusb-1.0-0-dev \   
    libprotobuf-dev protobuf-compiler libgoogle-glog-dev \    
    libgflags-dev libatlas-base-dev libeigen3-dev \    
    libsuitesparse-dev ros-${ROS_DISTRO}-cv-bridge \    
    ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-tf
```

\# 2. 安装 Pangolin

```
git clone https://github.com/stevenlovegrove/Pangolin.git 
cd Pangolin && mkdir build && cd build 
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j8 && sudo make install
```



\# 3. 安装 OpenCV 

```
git clone https://github.com/opencv/opencv.git 
git clone https://github.com/opencv/opencv_contrib.git 
cd opencv && mkdir build && cd build 
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \      
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. 
make -j8 && sudo make install
```

2、克隆并编译 GDN-SLAM

```bash
mkdir -p ~/gdn_slam_ws/src && cd ~/gdn_slam_ws/src
git clone https://github.com/~/GDN-SLAM.git
cd GDN-SLAM
```
```
cd GDN-SLAM
chmod +x build.sh
./build.sh
chmod +x build_ros.sh
.build_ros.sh
```
```
cd ~/gdn_slam_ws && catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

运行

```
wget  https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
tar -xvf rgbd_dataset_freiburg3_walking_xyz.tgz -C ./datasets/
```

```
python ./Examples/RGB-D/associate.py \
    ./datasets/rgbd_dataset_freiburg3_walking_xyz/rgb.txt \
    ./datasets/rgbd_dataset_freiburg3_walking_xyz/depth.txt > \
    ./Examples/RGB-D/associations/fr3_walking_xyz.txt
```

```
roslaunch GDN-SLAM yolov5.launch
rosrun GDN-SLAM RGBD Vocabulary/ORBvoc.txt Examples/RGB-D/tum_dynamic.yaml
rosbag play ./datasets/rgbd_dataset_freiburg3_walking_xyz.bag
rosrun GDN-SLAM rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/tum_dynamic.yaml ./datasets/rgbd_dataset_freiburg3_walking_xyz/ Examples/RGB-D/associations/fr3_walking_xyz.txt
```



## Acknowledgements
Our code builds on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

