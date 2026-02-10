<div align="center">

# 🚦 Smart Traffic Management System for Urban Congestion

### *Intelligent Traffic Signal Control using Deep Reinforcement Learning & Computer Vision*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN-009688?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Env-0081A5?style=for-the-badge)](https://gymnasium.farama.org/)
[![Unity](https://img.shields.io/badge/Unity-3D%20Sim-000000?style=for-the-badge&logo=unity&logoColor=white)](https://unity.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> **Built for Smart India Hackathon (SIH)** — A next-generation traffic control system that replaces fixed-timer signals with an AI agent that *sees* real-time traffic and *learns* optimal signal patterns to minimize congestion.

---

```
  🔴 ──────────────────────────────────────────── 🔴
  │  ██████╗  ██████╗ ███╗   ██╗    ████████╗██████╗  █████╗ ███████╗███████╗██╗ ██████╗│
  │  ██╔══██╗██╔═══██╗████╗  ██║    ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝│
  │  ██║  ██║██║   ██║██╔██╗ ██║       ██║   ██████╔╝███████║█████╗  █████╗  ██║██║     │
  │  ██║  ██║██║▄▄ ██║██║╚██╗██║       ██║   ██╔══██╗██╔══██║██╔══╝  ██╔══╝  ██║██║     │
  │  ██████╔╝╚██████╔╝██║ ╚████║       ██║   ██║  ██║██║  ██║██║     ██║     ██║╚██████╗│
  │  ╚═════╝  ╚══▀▀═╝ ╚═╝  ╚═══╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝ ╚═════╝│
  🟢 ──────────────────────────────────────────── 🟢
```

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [DQN vs Traditional Approach](#-dqn-vs-traditional-round-robin-approach)
- [Unity Integration](#-unity-3d-integration)
- [Future Scope](#-future-scope)
- [Contributors](#-contributors)

---

## 🔍 Problem Statement

<table>
<tr>
<td width="60%">

Urban traffic congestion is one of the most critical challenges facing modern cities:

- 🕐 **Average commuter loses 54+ hours/year** stuck in traffic
- ⛽ **Fuel wastage** of billions of liters annually due to idling vehicles
- 🌫️ **Air pollution spikes** at congested intersections
- 🚑 **Emergency vehicle delays** costing lives
- 💰 **Economic losses** exceeding $87 billion/year in the US alone

**Traditional traffic signals** use fixed timers or simple induction loops — they are **blind** to actual real-time traffic density and cannot adapt dynamically.

</td>
<td width="40%">

```
    Traditional System
    ┌─────────────────┐
    │  Fixed Timer: 30s│
    │  ⏱️ ⏱️ ⏱️ ⏱️    │
    │                  │
    │  🚗🚗🚗🚗🚗🚗   │ ← Heavy traffic
    │  🚗🚗🚗🚗🚗🚗   │    STILL WAITING
    │  🚗🚗🚗🚗       │
    │─────────────────│
    │                  │ ← Empty lane
    │  🟢 GREEN 30s    │    WASTING TIME
    └─────────────────┘
```

</td>
</tr>
</table>

---

## 💡 Our Solution

We built a **two-stage intelligent system** that combines:

<div align="center">

| Stage | Technology | Purpose |
|:-----:|:----------:|:-------:|
| **👁️ Vision** | TensorFlow + SSD MobileNet V2 | Real-time vehicle detection & counting from CCTV feeds |
| **🧠 Brain** | Deep Q-Network (DQN) | Learns optimal signal switching to minimize wait times |

</div>

### Key Innovation

```
  📹 Camera Feed ──→ 🔍 Object Detection ──→ 📊 Vehicle Count ──→ 🧠 DQN Agent ──→ 🚦 Signal Control
       │                    │                       │                    │                │
       │              SSD MobileNet            Per-lane count      Optimal action     Green/Red
       │              + OpenCV                  [12, 3, 8, 15]    "Open Lane 4"      signals
       ▼                    ▼                       ▼                    ▼                ▼
   Live Video        Bounding Boxes          Traffic Density      RL Decision       Less Waiting!
```

The DQN agent **learns from experience** — it's rewarded for clearing high-density lanes and penalized for switching to empty ones, naturally developing efficient traffic management strategies.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SMART TRAFFIC SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌───────────────────┐    ┌────────────────┐  │
│   │  📹 CCTV      │───→│  🔍 Object         │───→│  📊 Vehicle     │  │
│   │  Camera Feed  │    │  Detection Module  │    │  Count Module  │  │
│   │  (OpenCV)     │    │  (TF + SSD MNet)   │    │  (Per Lane)    │  │
│   └──────────────┘    └───────────────────┘    └───────┬────────┘  │
│                                                         │           │
│                                                         ▼           │
│   ┌──────────────┐    ┌───────────────────┐    ┌────────────────┐  │
│   │  🚦 Traffic   │←──│  🧠 DQN Agent       │←──│  🎮 Gymnasium   │  │
│   │  Signal       │    │  (Stable-          │    │  Environment   │  │
│   │  Controller   │    │   Baselines3)      │    │  (4-Lane Sim)  │  │
│   └──────────────┘    └───────────────────┘    └────────────────┘  │
│         │                                                           │
│         ▼                                                           │
│   ┌──────────────┐                                                  │
│   │  🎮 Unity 3D  │  ← Optional 3D visualization via socket        │
│   │  Simulation   │                                                  │
│   └──────────────┘                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology | Role |
|:--------:|:----------:|:----:|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) Python 3.11 | Core programming language |
| **RL Framework** | ![SB3](https://img.shields.io/badge/Stable--Baselines3-009688?style=flat-square) | DQN implementation & training |
| **RL Environment** | ![Gym](https://img.shields.io/badge/Gymnasium-0081A5?style=flat-square) | Custom 4-lane traffic environment |
| **Object Detection** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | SSD MobileNet V2 (COCO pre-trained) |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | Video processing, ROI, drawing |
| **Numerical** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Array ops, state management |
| **3D Visualization** | ![Unity](https://img.shields.io/badge/Unity-000000?style=flat-square&logo=unity&logoColor=white) | Optional 3D traffic simulation |

</div>

---

## 📁 Project Structure

```
📦 Smart-Traffic-Management-System/
├── 📄 README.md                          # You are here!
├── 📄 .gitignore                         # Git ignore rules
│
├── 🤖 ML_Model/                          # Core AI & ML code
│   ├── 🏋️ model.py                       # Initial DQN training (10K steps)
│   ├── 🏋️ model_trainingext.py            # Extended training (100K steps)
│   ├── 🎮 traffic_light_env.py            # Custom Gymnasium RL environment
│   ├── 👁️ object_detection.py             # Real-time vehicle detection & counting
│   ├── 🧪 Test without Unity.py           # CLI-based model testing
│   ├── 🔴 Realtimetest1.py               # Live camera + DQN integration test
│   ├── 🌐 ServerUnity.py                  # Socket server for Unity 3D communication
│   └── 📊 WhyBetter.py                   # Comparison: DQN vs round-robin baseline
│
├── 🧠 models/                            # TensorFlow models & research
│   ├── ssdlite_mobilenet_v2_coco_2018_05_09/  # Pre-trained SSD MobileNet V2
│   │   ├── saved_model/                   # SavedModel format for inference
│   │   └── pipeline.config               # Model pipeline configuration
│   ├── official/                          # TF Model Garden - official models
│   ├── research/                          # TF Model Garden - research models
│   │   └── object_detection/             # TF Object Detection API
│   └── tensorflow_models/                # TF Models package
│
└── 🔧 .venv/                            # Python virtual environment (git-ignored)
```

---

## ⚙️ How It Works

### 1. 👁️ Computer Vision — Vehicle Detection

The system uses **SSD MobileNet V2** (pre-trained on COCO dataset) to detect vehicles in real-time from camera feeds:

```python
# Detects cars (COCO class ID: 3) in each frame
# Splits the frame into LEFT and RIGHT zones
# Counts vehicles per zone using centroid tracking with departure threshold

📹 Video Frame → Resize 320x320 → SSD MobileNet V2 → Filter Cars → Count per Lane
```

**Key features:**
- **ROI-based detection** — Only counts vehicles in the defined region of interest
- **Centroid tracking** — Prevents double-counting using 50px proximity threshold
- **Departure detection** — Removes cars from count after 2-second absence
- **Directional counting** — Splits into left/right lanes using vertical centerline

### 2. 🧠 Reinforcement Learning — DQN Agent

The traffic signal controller is a **Deep Q-Network** trained using Stable-Baselines3:

```
┌─────────────────────────────────────────────────┐
│              DQN ARCHITECTURE                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  Observation Space: Box(0, 20, shape=(4,))      │
│  ├── Lane 1 car count: [0-20]                   │
│  ├── Lane 2 car count: [0-20]                   │
│  ├── Lane 3 car count: [0-20]                   │
│  └── Lane 4 car count: [0-20]                   │
│                                                  │
│  Action Space: Discrete(4)                       │
│  ├── Action 0 → Open Lane 1                     │
│  ├── Action 1 → Open Lane 2                     │
│  ├── Action 2 → Open Lane 3                     │
│  └── Action 3 → Open Lane 4                     │
│                                                  │
│  Reward Function:                                │
│  ├── +100  → All lanes cleared                  │
│  ├── +10   → A lane fully cleared               │
│  ├── -1    → Step penalty (encourage speed)     │
│  └── -5    → Chose an empty lane (wasteful)     │
│                                                  │
│  Policy: MlpPolicy (Multi-Layer Perceptron)     │
│  Training: 100,000 timesteps                     │
│  Avg Reward: 128 | Avg Episode Length: ~9.8      │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 3. 🔗 Integration Pipeline

```
Step 1: Camera captures live traffic video
Step 2: Object detection counts cars per lane  →  [12, 3, 8, 15]
Step 3: DQN observes the state                 →  "Lane 4 has most cars"
Step 4: DQN selects action                     →  Action: Open Lane 4
Step 5: 5 cars pass through Lane 4             →  [12, 3, 8, 10]
Step 6: Repeat until congestion is resolved
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Git**
- *(Optional)* Unity 2021+ for 3D visualization
- *(Optional)* Webcam or video file for real-time detection

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rajatpundir7/Smart-Traffic-Management-Systern-for-Urban-Congestion.git
cd Smart-Traffic-Management-Systern-for-Urban-Congestion

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install numpy stable-baselines3 tensorflow opencv-python gymnasium shimmy
```

### Quick Verify

```bash
python -c "import stable_baselines3; import tensorflow; import cv2; print('All dependencies OK!')"
```

---

## 🎯 Usage

### 🏋️ Train the DQN Model

```bash
cd ML_Model

# Initial training (10,000 timesteps) — Quick test
python model.py

# Extended training (100,000 timesteps) — Full training
python model_trainingext.py
```

> The trained model is saved as `dqn_traffic_light.zip` in the `ML_Model/` directory.

### 🧪 Test without Unity (CLI Mode)

```bash
python "Test without Unity.py"
```

Enter the number of cars in each lane when prompted:
```
Enter the number of cars in lane 1 (0 to 20): 15
Enter the number of cars in lane 2 (0 to 20): 3
Enter the number of cars in lane 3 (0 to 20): 18
Enter the number of cars in lane 4 (0 to 20): 7
```

Watch the DQN agent intelligently prioritize high-traffic lanes!

### 📹 Real-time Test with Camera

```bash
python Realtimetest1.py
```

This launches:
- **Thread 1:** Object detection on video feed (counts cars in left/right zones)
- **Main thread:** DQN agent making decisions based on real-time counts

### 🌐 Unity 3D Integration

```bash
python ServerUnity.py
```

Starts a socket server on `localhost:65432` — connect your Unity traffic simulation client to visualize the AI decisions in 3D.

### 📊 Compare DQN vs Baseline

```bash
python WhyBetter.py
```

See how the round-robin (traditional) approach takes more steps than the DQN agent for the same traffic scenario.

---

## 📈 Model Performance

<div align="center">

| Metric | Value |
|:------:|:-----:|
| **Training Timesteps** | 100,000 |
| **Episodes Completed** | ~9,388 |
| **Avg Episode Reward** | **128** |
| **Avg Episode Length** | ~9.8 steps |
| **Training FPS** | ~602 |
| **Training Time** | ~166 seconds |
| **Exploration Rate** | 0.05 (final) |
| **Learning Rate** | 0.0001 |

</div>

---

## 🏆 DQN vs Traditional Round-Robin Approach

<table>
<tr>
<td width="50%">

### ❌ Round-Robin (Traditional)
```
Input: [15, 3, 18, 7]

Step 1: Lane 1 → [10, 3, 18, 7]
Step 2: Lane 2 → [10, 0, 18, 7]
Step 3: Lane 3 → [10, 0, 13, 7]
Step 4: Lane 4 → [10, 0, 13, 2]
Step 5: Lane 1 → [ 5, 0, 13, 2]
Step 6: Lane 2 → [ 5, 0, 13, 2]  ❌ WASTED
Step 7: Lane 3 → [ 5, 0,  8, 2]
Step 8: Lane 4 → [ 5, 0,  8, 0]
...
Total: ~15 steps
```

**Problems:** Visits empty lanes, no prioritization

</td>
<td width="50%">

### ✅ DQN (Our Approach)
```
Input: [15, 3, 18, 7]

Step 1: Lane 3 → [15, 3, 13, 7]  ← Highest!
Step 2: Lane 1 → [10, 3, 13, 7]  ← 2nd highest
Step 3: Lane 3 → [10, 3,  8, 7]
Step 4: Lane 1 → [ 5, 3,  8, 7]
Step 5: Lane 3 → [ 5, 3,  3, 7]
Step 6: Lane 4 → [ 5, 3,  3, 2]
Step 7: Lane 1 → [ 0, 3,  3, 2]  🎉 Cleared!
...
Total: ~9 steps
```

**Advantage:** Prioritizes busy lanes, skips empty ones

</td>
</tr>
</table>

<div align="center">

### 📉 Result: **~40% fewer steps** to clear all traffic

</div>

---

## 🎮 Unity 3D Integration

The system supports real-time 3D visualization through **Unity** via TCP sockets:

```
┌──────────────┐     TCP Socket      ┌──────────────┐
│  Python DQN  │ ──── :65432 ────→  │  Unity 3D    │
│  Server      │                     │  Simulation  │
│              │  Sends: Lane #      │              │
│              │  "Open Lane 3"      │  🚗 🚙 🚕    │
│              │                     │  🚦 Animates  │
│              │  Sends: TERMINATE   │  signals     │
└──────────────┘  when all clear     └──────────────┘
```

---

## 🔮 Future Scope

| Enhancement | Description |
|:----------:|:-----------:|
| 🚑 **Emergency Vehicle Priority** | Detect emergency vehicles and override signal timing |
| 🛰️ **Multi-Intersection Coordination** | Coordinate signals across multiple junctions using multi-agent RL |
| ☁️ **Cloud Deployment** | Edge computing + cloud dashboards for city-wide monitoring |
| 📱 **Mobile App** | Real-time traffic status and route suggestions for commuters |
| 🌙 **Night/Weather Adaptation** | Adjust detection model for low-light and adverse weather |
| 📊 **Analytics Dashboard** | Historical traffic patterns, peak-hour analysis, and predictions |
| 🚶 **Pedestrian Detection** | Include pedestrian crossing demands in signal optimization |

---

## 👥 Contributors

<div align="center">

| Name | Role |
|:----:|:----:|
| **Rajat Pundir** | Project Lead & Developer |

</div>

---

<div align="center">

### ⭐ If you found this project useful, give it a star!

[![GitHub Stars](https://img.shields.io/github/stars/Rajatpundir7/Smart-Traffic-Management-Systern-for-Urban-Congestion?style=social)](https://github.com/Rajatpundir7/Smart-Traffic-Management-Systern-for-Urban-Congestion)

---

**Built with ❤️ for Smart India Hackathon**

*Making cities smarter, one intersection at a time.*

</div>
