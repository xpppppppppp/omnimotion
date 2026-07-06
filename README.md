## 本项目基于原版 OmniMotion 的修改

本分支在原版 [OmniMotion](https://github.com/qianqianwang68/omnimotion) (ICCV 2023) 基础上进行了以下扩展和优化：

### 1. 骨架关键点跟踪支持 (Keypoint Support)

- **`loaders/keypoint.py`**: 新增 `KeypointDataset`，支持加载每帧的骨架关键点标注作为训练数据
- **`util.load_keypoints()`**: 支持多种关键点格式，包括 LabelMe、OpenPose 和自定义 JSON 格式
- **`util.load_query_points()`**: 从关键点文件中加载查询点用于可视化
- 新增配置文件参数：`--keypoint_dir`, `--keypoint_format`, `--num_joints`, `--min_keypoint_conf`, `--patch_size`, `--foreground_hard_mask` 等

### 2. 点跟踪头部网络 (Point Tracking Head)

- **`networks/point_head.py`**: 新增 `PointMLPHead` 模块，一个可训练的 MLP 头部网络，在 OmniMotion 底层对应关系基础上预测残差修正
- 支持多种辅助特征输入：
  - **RGB 图像块特征** (`--point_use_rgb_patch`)：在源点和目标点周围采样 RGB 块
  - **DINO 特征** (`--point_use_dino_feature`)：使用 DINO 自监督视觉特征
  - **DINO 相关特征** (`--point_use_dino_correlation`)：在局部窗口内计算 DINO 特征的余弦相似度热力图，通过 soft-argmax 得到亚像素偏移
- 支持从 RAFT flow 或者关键点标注两种监督方式 (`--point_supervision`)
- 训练时可联合 OmniMotion 一起优化，也可独立加载进行推理

### 3. 滚动查询模式 (Rolling Query)

- 新增 `plot_correspondences_for_pixels_rolling()` 方法，使用帧间传播（frame-to-frame）替代直接从源帧到目标帧的查询方式
- 通过逐帧链式传播，在长序列上获得更稳定的跟踪结果
- 通过 `--rolling_query` 参数启用

### 4. 训练增强

- **前背景掩码过滤** (`--train_use_mask`)：限制 RAFT flow 监督采样到前景区域，可配置腐蚀内核 (`--train_mask_erosion`)
- **关键点偏置采样** (`--train_keypoint_bias`)：在训练时倾向于在关键点轨迹附近采样 flow 点
  - 支持动态关键点轨迹目录 (`--train_keypoint_track_dir`)
  - 可配置邻域半径 (`--train_keypoint_radius`) 和偏置比例 (`--train_keypoint_focus_ratio`)
- **源帧选择控制**：支持固定以查询帧为源帧 (`--train_query_frame_only`) 或按概率选择 (`--train_query_frame_prob`)
- 默认迭代次数从 100k 增加到 **200k**
- 默认采样点数从 256 减少到 **32** 以降低显存占用
- Checkpoint 保存频率从每 20000 步提高到每 **10000** 步
- 新增 `--skip_checkpoint_visualization` 可跳过 checkpoint 时的可视化渲染以加速保存

### 5. 独立可视化工具 (vistest.py)

- **`vistest.py`**: 独立的命令行点跟踪可视化脚本，直接从 `.pth` checkpoint 加载模型
- 支持三种查询点来源：手动指定坐标、随机采样、骨架关键点
- 输出每帧跟踪点位置为 JSON 文件（LabelMe 格式），方便后续在标注工具中查看和编辑
- 支持滚动查询模式和点头部网络的推理

### 6. 其他改进

- **优化器/调度器容错**：checkpoint 加载时若优化器或调度器状态不兼容（如参数组变化）会优雅跳过而非崩溃
- **数据集加载器简化**：移除了原始的 `WeightedRandomSampler` 多数据集加权采样逻辑，简化了 `create_training_dataset.py`
- **自动化脚本** (`auto.sh`)：一键完成预处理和训练的脚本
- **配置文件更新**：configs/default.txt 中的默认参数已针对实际训练需求调整

## Citation
```
@article{wang2023omnimotion,
    title   = {Tracking Everything Everywhere All at Once},
    author  = {Wang, Qianqian and Chang, Yen-Yu and Cai, Ruojin and Li, Zhengqi and Hariharan, Bharath and Holynski, Aleksander and Snavely, Noah},
    journal = {ICCV},
    year    = {2023}
}
```



