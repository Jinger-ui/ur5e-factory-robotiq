# UR5e 工业机械臂 + Robotiq 三指夹爪 仿真（复杂工厂环境版）

> 使用 Webots R2025a 构建的 UR5e + Robotiq 3F Gripper 完整工业仿真项目，包含复杂工厂环境、多目标抓取与放置任务。

## 项目简介

本项目是一个基于 **Webots 仿真平台** 的工业机器人仿真系统，使用 **Universal Robots UR5e** 六轴机械臂搭配 **Robotiq 三指自适应夹爪（3F Gripper）**，在一个完整的工厂环境中执行多目标物体的抓取与分拣任务。

控制器采用 **有限状态机（FSM）** 架构，集成了 **逆运动学（IK）求解**、**梯形速度规划**、**笛卡尔直线插值** 等核心算法，并通过力传感器、距离传感器、触觉传感器和摄像头实现多模态感知。

## 与 ur5e-robot-simulation 仓库的区别

| 特性 | 本仓库 (ur5e-factory-robotiq) | [ur5e-robot-simulation](https://github.com/Jinger-ui/ur5e-robot-simulation) |
|------|-------------------------------|----------------------------------------------------------------------------|
| **夹爪类型** | Robotiq 三指自适应夹爪 (3F Gripper) | Connector 磁吸方案 |
| **抓取方式** | 物理摩擦力抓取（真实仿真） | 磁性连接（100% 成功率） |
| **工厂环境** | 完整工厂（围墙、传送带、货架、信号灯等） | 简化场景 |
| **控制器复杂度** | 18 态 FSM + IK + 力反馈 | 精简版控制器 |
| **传感器** | 摄像头 + 距离传感器 + 触觉传感器 + 力反馈 | 基础传感器 |
| **适用场景** | 学术研究、算法验证 | 课程演示、快速原型 |
| **抓取成功率** | 受 IK 精度和物理仿真影响 | 极高（磁吸连接） |

**建议**：如果需要稳定的抓取演示，推荐使用 [ur5e-robot-simulation](https://github.com/Jinger-ui/ur5e-robot-simulation) 的 Connector 方案；如果需要研究真实抓取算法，使用本仓库。

## 工厂场景描述

本项目构建了一个 10m × 8m 的完整工厂环境，包含以下元素：

- **工厂围墙**：四面 Roughcast 材质围墙（高 2.6m），营造封闭厂房氛围
- **安全围栏**：黄色金属围栏包围机器人工作区域（3.65m × 2.8m）
- **传送带**：6m 长 ConveyorBelt，运行速度 0.15 m/s
- **工作台**：拾取台（pick_table）和放置台（place_table），不同颜色区分
- **货架**：镀锌金属货架，用于存储物料
- **信号灯塔**：三色信号灯（红/黄/绿），指示系统状态
- **木质托盘堆**：3 层叠放的标准木托盘
- **纸箱**：分布在传送带附近的包装箱
- **灭火器**：工业安全设备
- **机器人基座**：0.8m 高的镀锌金属底座

### 抓取目标物体（5 个）

| 名称 | 形状 | 颜色 | 质量 | 位置 |
|------|------|------|------|------|
| target_red | 立方体 (5cm) | 红色 | 0.20 kg | (0.40, -0.12) |
| target_green | 圆柱体 | 绿色 | 0.15 kg | (0.40, 0.00) |
| target_blue | 长方体 | 蓝色 | 0.18 kg | (0.40, 0.12) |
| target_yellow | 圆柱体 | 黄色 | 0.14 kg | (0.48, -0.08) |
| target_orange | 球体 | 橙色 | 0.12 kg | (0.48, 0.08) |

## 技术栈

- **仿真平台**：Webots R2025a
- **编程语言**：Python 3.11
- **机械臂**：Universal Robots UR5e（6 自由度）
- **夹爪**：Robotiq 3-Finger Adaptive Gripper
- **运动学求解**：[ikpy](https://github.com/Phylliade/ikpy) - 逆运动学库
- **数值计算**：NumPy

## 控制器功能

### 完整版控制器 (`ur5e_complete_controller`)

- **18 态有限状态机（FSM）**：INIT → WARMUP → GO_HOME → IDLE → PICK_APPROACH → PICK_DESCEND → GRASPING → LIFT → TRANSPORT → PLACE_APPROACH → PLACE_DESCEND → RELEASING → RETREAT → RETURN_HOME → FINAL_HOME → STANDBY 等
- **ikpy 逆运动学求解**：基于 DH 参数构建 UR5e 运动链，多种子 IK 搜索，前向运动学验证
- **梯形速度规划**：S 曲线平滑插值（`s = t² × (3 - 2t)`），可调速度因子
- **笛卡尔直线插值**：世界坐标到机器人坐标变换
- **传感器集成**：摄像头、距离传感器、触觉传感器
- **力反馈**：夹爪 `getForceFeedback()` 监测抓取力
- **可配置参数**：通过 `config.json` 调整目标位置、速度、时序等

### 早期版控制器 (`ur5e_controller`)

- 基础版 FSM（11 态）
- 预定义关节角度 + IK 辅助
- 三次插值轨迹规划

## 项目结构

```
ur5e-factory-robotiq/
├── worlds/
│   ├── ur5e_complete.wbt          # 完整工厂场景（最终版）
│   └── ur5e_demo.wbt              # 早期版本工厂场景
├── controllers/
│   ├── ur5e_complete_controller/
│   │   ├── ur5e_complete_controller.py   # 完整版控制器（18态FSM）
│   │   └── config.json                    # 配置文件
│   └── ur5e_controller/
│       └── ur5e_controller.py             # 早期版控制器
├── README.md
├── .gitignore
└── LICENSE
```

## 快速开始

### 前置条件

1. 安装 [Webots R2025a](https://cyberbotics.com/doc/guide/installation-procedure)
2. 安装 Python 3.11+
3. 安装依赖：

```bash
pip install ikpy numpy
```

### 运行仿真

1. 克隆本仓库：

```bash
git clone https://github.com/Jinger-ui/ur5e-factory-robotiq.git
cd ur5e-factory-robotiq
```

2. 使用 Webots 打开场景文件：

```bash
# 完整版场景（推荐）
webots worlds/ur5e_complete.wbt

# 早期版本场景
webots worlds/ur5e_demo.wbt
```

3. 仿真会自动启动控制器，机械臂将依次抓取 5 个目标物体并放置到指定位置。

### 控制台输出示例

```
[INIT] Motors=6 Gripper=3 Cam=OK Dist=OK Touch=OK

============================================================
  UR5e Robust Pick-and-Place Controller
  --------------------------------------------
  IK Engine       : ikpy (verification)
  Motion Method   : Pre-computed joint poses
  Targets         : 5
  Time Step       : 8 ms
============================================================

=======================================================
  CYCLE 1: 'target_red'
  World position: [0.40, -0.12, 0.765]
=======================================================
  -> Moving to approach above target
  -> Descending to grasp position
  -> Closing gripper
  -> Grasp complete (force=2.3N)
  -> Lifting object
  [OK] Object placed! (1 total)
```

## 已知问题

1. **抓取成功率**：由于使用真实物理摩擦力抓取（非磁吸），抓取成功率受 IK 求解精度、物体形状和摩擦系数影响。球体（target_orange）尤其难以稳定抓取。
2. **IK 精度**：ikpy 基于数值迭代，可能陷入局部最优，控制器使用多种子搜索缓解此问题。
3. **建议**：如需 100% 稳定的抓取演示，建议参考 [ur5e-robot-simulation](https://github.com/Jinger-ui/ur5e-robot-simulation) 仓库的 Connector（磁吸连接）方案。

## 引用的开源项目

- [Webots](https://github.com/cyberbotics/webots) - 开源机器人仿真平台 (Apache 2.0)
- [ikpy](https://github.com/Phylliade/ikpy) - Python 逆运动学库 (GPL-2.0)
- [Universal Robots UR5e PROTO](https://github.com/cyberbotics/webots/tree/master/projects/robots/universal_robots) - Webots 官方 UR5e 模型
- [Robotiq 3F Gripper PROTO](https://github.com/cyberbotics/webots/tree/master/projects/devices/robotiq) - Webots 官方 Robotiq 夹爪模型

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

**作者**：[Jinger-ui](https://github.com/Jinger-ui)
