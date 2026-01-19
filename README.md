# RoboMaster YOLO Vision Control System

## 🚀 Project Overview
An advanced 4-mode robotic control system for DJI RoboMaster with real-time computer vision, autonomous navigation, and intelligent person tracking capabilities.

## 📋 Features

### **4 Operational Modes:**
1. **MANUAL MODE** - Full keyboard control with real-time YOLO object detection
2. **AUTO MODE** - Autonomous obstacle avoidance using YOLO + DeepSORT tracking
3. **TRACK MODE** - Basic person following using RoboMaster's internal detection
4. **PID_TRACK MODE** - Advanced PID-controlled tracking with motion prediction

### **Key Capabilities:**
- Real-time YOLOv8 object detection (GPU accelerated)
- DeepSORT object tracking for persistent obstacle identification
- Intelligent obstacle avoidance with wall/corner detection
- PID-controlled smooth person tracking with velocity prediction
- Adjustable danger zones and safety boundaries
- IR sensor integration for emergency stop
- Customizable detection parameters
- Live camera feed with overlay information

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- CUDA-capable GPU (recommended for YOLO acceleration)
- DJI RoboMaster EP Robot
- WiFi connection to robot

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/robomaster-yolo-control.git
cd robomaster-yolo-control
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Key Dependencies**
- `robomaster` - Official DJI RoboMaster SDK
- `ultralytics` - YOLOv8 object detection
- `opencv-python` - Computer vision processing
- `torch` - PyTorch for YOLO inference
- `keyboard` - Keyboard input handling
- `deep-sort-realtime` - Object tracking (for AUTO mode)

## 📁 Project Structure
```
robomaster-yolo-control/
├── main.py              # Main controller and interface
├── manual_mode.py       # Manual control with YOLO detection
├── auto_mode.py         # Autonomous navigation system
├── track_mode.py        # Basic person tracking
├── pid_track_mode.py    # Advanced PID tracking
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎮 Control System

### **Mode Switching**
- Press `T` to cycle through all 4 modes:
  - MANUAL → AUTO → TRACK → PID_TRACK

### **Movement Controls (Manual Mode)**
```
W / S          - Forward / Backward
A / D          - Right / Left strafe
Z / C          - Rotate left / right
Arrow Keys     - Robotic arm control
```

### **Detection Controls**
```
SPACE          - Toggle object detection ON/OFF
CTRL           - Toggle bounding boxes
ALT            - Toggle distance lines
1 / 2          - Decrease / Increase speed
```

### **Danger Zone Controls**
```
5 / 6          - Decrease / Increase danger zone size
7 / 8          - Move danger zone left / right
9 / 0          - Move danger zone up / down
[ / ]          - Adjust ignore zone width
; / '          - Adjust ignore zone height
i              - Toggle ignore zone
```

## 🤖 Mode Details

### **1. MANUAL MODE**
- Full keyboard control of robot movement
- Real-time YOLO object detection
- Configurable detection zones
- Visual distance estimation

### **2. AUTO MODE**
- **Right-side Avoidance Strategy**: Always moves right when obstacles detected
- **Wall Detection**: Counts consecutive right movements
- **Recovery Maneuver**: 90° left turn after 2 consecutive rights
- **Safety Layers**: IR sensor (0.3m) + YOLO detection (pixel distance)
- **Tracking**: DeepSORT for persistent obstacle identification

### **3. TRACK MODE**
- Uses RoboMaster's internal person detection
- Simple distance-based following
- Target distance: 1.5 meters
- Automatic search behavior when person lost

### **4. PID_TRACK MODE** (Advanced)
- **PID Control**: Smooth distance and angle maintenance
- **Motion Prediction**: Tracks person velocity for better following
- **Adaptive Speed**: Slows down when person at rest, speeds up during fast movement
- **No Backward Bug Fix**: Prevents backward movement when person stops
- **Emergency Stop**: IR sensor triggered at 0.25m distance

## ⚙️ Configuration

### **Auto Mode Parameters**
```python
self.safe_distance_threshold = 0.3    # IR sensor safety distance (meters)
self.yolo_distance_threshold = 10     # YOLO obstacle detection threshold (pixels)
self.normal_forward_speed = 0.4       # Base forward speed
self.turn_duration = 1.5              # 90° turn duration (seconds)
```

### **PID Track Mode Parameters**
```python
self.target_distance = 1.5            # Optimal following distance (meters)
self.comfort_zone_min = 1.2           # Start slowing down distance
self.comfort_zone_max = 1.8           # Start speeding up distance
self.fast_movement_threshold = 40     # Person speed threshold (pixels/sec)
```

## 🚀 Quick Start

1. **Connect to RoboMaster:**
   - Power on RoboMaster
   - Connect computer to RoboMaster WiFi
   - Ensure robot is in FREE mode

2. **Launch Application:**
```bash
python main.py
```

3. **Initial Operation:**
   - System starts in MANUAL mode
   - Press `SPACE` to enable object detection
   - Use `T` to switch between modes
   - Press `O` to quit safely

## 🎯 Performance Tips

### **For Best Tracking:**
1. Ensure good lighting conditions
2. Maintain clear line of sight with tracked person
3. Start in open spaces for AUTO mode
4. Calibrate camera focal length for your environment

### **Optimization:**
- GPU acceleration recommended for YOLO
- Reduce detection frame size for higher FPS
- Adjust detection confidence thresholds based on environment

## 🔧 Technical Details

### **Computer Vision Pipeline**
1. **Frame Capture**: RoboMaster camera stream (720p)
2. **Detection**: YOLOv8-nano (optimized for real-time)
3. **Tracking**: DeepSORT for object persistence
4. **Navigation**: State machine with obstacle avoidance
5. **Control**: PID algorithms for smooth movement

### **Distance Estimation**
- Uses camera geometry and known object widths
- IR sensor provides ground truth for close objects
- Pixel-to-meter conversion based on focal length

## ⚠️ Safety Features

1. **Emergency Stop**: IR sensor triggers at 0.25m
2. **Speed Limits**: Capped forward/backward speeds
3. **Stuck Detection**: AUTO mode detects consecutive right movements
4. **Graceful Degradation**: Fallbacks for missing dependencies
5. **Clean Shutdown**: Proper resource cleanup on exit

## 🐛 Troubleshooting

### **Common Issues:**

1. **Robot Connection Failed**
   - Verify WiFi connection to RoboMaster
   - Check robot battery level
   - Ensure robot is in FREE mode

2. **Low Detection FPS**
   - Enable GPU acceleration
   - Reduce detection resolution
   - Lower max_det parameter in YOLO

3. **Person Not Detected (TRACK Mode)**
   - Ensure person is within camera view
   - Check lighting conditions
   - Verify RoboMaster vision subscription

4. **AUTO Mode Not Avoiding Obstacles**
   - Verify DeepSORT installation
   - Check YOLO detection is enabled
   - Adjust distance thresholds

### **Debug Mode:**
Run with verbose output:
```bash
python main.py --debug
```

## 📈 Future Enhancements

1. **Machine Learning**: Train custom object detectors
2. **Mapping**: SLAM integration for environment mapping
3. **Multi-Robot**: Cooperative multi-robot systems
4. **Gesture Control**: Hand gesture recognition
5. **Voice Commands**: Natural language control interface
6. **Web Interface**: Remote control via browser
7. **Data Logging**: Session recording and playback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- DJI for RoboMaster platform and SDK
- Ultralytics for YOLOv8 implementation
- Roboflow for computer vision resources
- OpenCV community for computer vision tools

