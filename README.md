# Car Control with Hand Gesture

A computer-vision-based system that enables controlling a robotic car using hand gestures. The project uses a camera for gesture recognition and sends corresponding driving commands to a robot or ROS-based system.

---

## Table of Contents
- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About
This project integrates computer vision and robotics to allow gesture-based control of a robotic car. A camera captures hand movements, interprets the gesture, and converts it into movement commands such as forward, backward, left, right, and stop. The repository structure indicates usage of ROS (Robot Operating System) through a Catkin workspace.

---

## Features
- Real-time hand gesture detection
- Maps gestures to control commands for a robotic car
- ROS-based workspace with Catkin build system
- Python-based gesture recognition using external libraries
- Modular code inside the `src/` directory

---

## Project Structure
```
Car-control-with-hand-gesture/
  build/
  devel/
  src/
  requirements.txt
  .catkin_workspace
```

**build/** – Auto-generated build files by Catkin

**devel/** – Development environment created after `catkin_make`

**src/** – Contains gesture detection and robot control packages

**requirements.txt** – Python dependencies

**.catkin_workspace** – Indicates ROS Catkin workspace

---

## Prerequisites
- Ubuntu (recommended for ROS)
- ROS Noetic / ROS1 with Catkin
- Python 3.x
- Webcam / USB camera
- OpenCV and other dependencies listed in `requirements.txt`
- A robotic car or simulated robot receiving control commands

---

## Installation
```bash
git clone https://github.com/GeekySalami/Car-control-with-hand-gesture.git
cd Car-control-with-hand-gesture
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### ROS Build
```bash
catkin_make
source devel/setup.bash
```

---

## Running the Project
Start ROS core:
```bash
roscore
```

Run the gesture recognition node (example):
```bash
rosrun <package_name> gesture_node.py
```

Run the robot control node (example):
```bash
rosrun <package_name> control_node.py
```

Make gestures in front of the camera; they will be interpreted and converted into movement commands.

---

## Usage
- Ensure proper lighting for the camera
- Hold your hand clearly in the camera frame
- Use predefined gestures such as:
  - Open palm → Move forward
  - Fist → Stop
  - Palm left → Turn left
  - Palm right → Turn right
  - Palm down → Reverse
- Modify gesture mappings inside the source code if needed

---

## Configuration
You may adjust:
- Camera index (0, 1, etc.)
- ROS topics (e.g., `/cmd_vel`, `/gesture_cmd`)
- Thresholds and model parameters in gesture detection logic
- Robot communication settings (serial, ROS topic, etc.)

---

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

All contributions are welcome.

---

## Contact
Author: **GeekySalami**
GitHub: https://github.com/GeekySalami
