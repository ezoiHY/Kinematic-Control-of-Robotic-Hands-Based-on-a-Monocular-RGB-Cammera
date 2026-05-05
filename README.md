# Kinematic-Control-of-Robotic-Hands-Based-on-a-Monocular-RGB-Cammera

Vision-based teleoperation pipeline for dexterous robotic hands using only a monocular RGB camera.  
It combines hand detection, 3D hand pose reconstruction, coordinate normalization and optimization-based retargeting to control robotic hands in a physics simulator.

**Goal:** Map human hand motion from a standard RGB camera to joint commands of a dexterous robotic hand – no gloves, no depth cameras.

---

## Overview

- **MediaPipe Hands** – real-time hand detection
- **HaMeR** – monocular 3D hand pose reconstruction
- **Hand-centric normalization** – removes global wrist position/orientation
- **DexPilot-style retargeting** – maps human features to robot joint space
- **SAPIEN** – real-time robotic hand simulation

---

## Pipeline
Monocular RGB Camera
↓
MediaPipe Hand Detection
↓
HaMeR 3D Hand Pose Estimation
↓
Hand-Centric Coordinate Transformation
↓
Dexterous Hand Retargeting
↓
SAPIEN Robot Hand Simulation


---

## Features

- Monocular RGB-based dexterous hand teleoperation
- No data glove or depth camera needed
- Real-time hand detection (MediaPipe)
- 21 keypoints 3D reconstruction (HaMeR)
- Hand-centric coordinate normalization
- Vector-based / position-based retargeting
- Multi-process camera capture with latest-frame-only strategy
- Temporal smoothing for reduced jitter
- Supports multiple dexterous robotic hands

---

## Method (simplified)

1. **Hand Detection**  
   MediaPipe detects the hand region, 2D landmarks and handedness. Only one hand is processed at a time.

2. **3D Hand Pose Estimation**  
   The hand crop is sent to HaMeR, which outputs 21 3D keypoints (wrist + 4 finger joints per finger).

3. **Coordinate Transformation**  
   Keypoints are translated to a wrist-centred frame and rotated to a canonical hand coordinate system, preserving only finger articulation.

4. **Retargeting & Robot Control**  
   An optimisation problem minimises the difference between human hand features (e.g. keypoint vectors) and robot hand features. The resulting joint angles drive the SAPIEN simulation.

---

## Real-Time Design

- **Producer process**: continuously captures camera/video frames.
- **Consumer process**: runs detection, pose estimation, retargeting and rendering.
- **Latest-frame strategy**: only the newest frame is processed; outdated frames are dropped.
- **Temporal smoothing**: exponential smoothing on hand keypoints reduces jitter.

---

## Supported Robotic Hands

Allegro Hand, Shadow Hand, LEAP Hand, DClaw, Ability Hand, Barrett Hand, SVH Hand.  
Models are loaded from URDFs in the `hands/` directory.

---

## Project Structure
Kinematic-Control-of-Robotic-Hands-Based-on-a-Monocular-RGB-Camera/
├── DRBT2.py # Main teleoperation script
├── hands/ # Robot URDF & meshes
├── README.md
└── requirements.txt
