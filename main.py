import cv2
import time
import threading
import queue
from robomaster import robot, chassis, camera
import torch
import keyboard
from ultralytics import YOLO

# Import mode modules
from manual_mode import ManualMode
from auto_mode import AutoMode
from track_mode import TrackMode
from pid_track_mode import PidTrackMode  # NEW: Import PID Track Mode

# Try to import DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    print("Note: DeepSORT not installed. Run: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False


class OptimizedYOLORobotController:
    def __init__(self):
        print("\n==============================")
        print("   Optimized RoboMaster Init")
        print("==============================\n")

        self.is_running = True
        self.speed = 0.5
        self.rotation_speed = 45

        self.object_detection_enabled = False
        self.show_bounding_boxes = False
        self.show_lines = False
        self.ignore_zone_enabled = True
        
        # 4-mode system (UPDATED: Added PID_TRACK)
        self.modes = ['MANUAL', 'AUTO', 'TRACK', 'PID_TRACK']  # MODIFIED
        self.current_mode_idx = 0
        self.last_mode_press = 0
        self.previous_mode_idx = -1
        
        # Mode instances
        self.manual_mode = None
        self.auto_mode = None
        self.track_mode = None
        self.pid_track_mode = None  # NEW: PID tracking mode
        
        # Danger zone settings
        self.danger_zone_size = 10
        self.danger_zone_x = 0
        self.danger_zone_y = 0
        self.danger_zone_color = (0, 0, 255)
        self.danger_zone_alpha = 0.3
        
        # Ignore zone settings
        self.ignore_zone_width = 560
        self.ignore_zone_height = 120
        self.ignore_zone_alpha = 0.35
        self.ignore_zone_color = (200, 220, 255)

        # Key press cooldowns
        self.last_space_press = 0
        self.last_ctrl_press = 0
        self.last_alt_press = 0
        self.last_5_press = 0
        self.last_6_press = 0
        self.last_7_press = 0
        self.last_8_press = 0
        self.last_9_press = 0
        self.last_0_press = 0
        self.cooldown = 0.25

        # Frame and detection data
        self.latest_frame = None
        self.latest_detections = []  # For manual mode
        self.latest_tracks = []
        self.tracked_obstacles = {}
        
        # FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # IR sensor distance
        self.inf_distance = None
        
        # Movement flags
        self.movement = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'rotate_left': False,
            'rotate_right': False
        }
        
        # Arm flags
        self.arm_up_flag = False
        self.arm_down_flag = False
        self.arm_left_flag = False
        self.arm_right_flag = False
        self.arm_step = 50
        self.arm_position_x = 0
        self.arm_position_y = 0

        # Frame lock for camera thread
        self.frame_lock = threading.Lock()

        # Initialize components
        self._init_deepsort()
        self._init_yolo()
        self._init_robot()
        self._init_ir_sensor()
        self._init_camera()
        self._init_modes()
        self._start_threads()

    def _init_deepsort(self):
        """Initialize DeepSORT tracker"""
        if DEEPSORT_AVAILABLE:
            try:
                print("Initializing DeepSORT tracker...")
                self.tracker = DeepSort(
                    max_age=30,
                    n_init=3,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.4,
                    nn_budget=None,
                    override_track_class=None
                )
                print("✓ DeepSORT initialized")
            except Exception as e:
                print(f"✗ DeepSORT initialization failed: {e}")
                self.tracker = None
        else:
            self.tracker = None
            print("⚠ DeepSORT not available - AUTO mode may be limited")

    def _init_yolo(self):
        """Initialize YOLO model"""
        try:
            print("Loading YOLO model…")
            self.model = YOLO("yolov8n.pt")
            
            if torch.cuda.is_available():
                self.model.to("cuda")
                print("✓ YOLO running on GPU")
            else:
                print("✓ YOLO running on CPU")
            
            self.model.fuse()
            print(f"✓ YOLO loaded ({len(self.model.names)} classes)\n")
            
        except Exception as e:
            print(f"✗ ERROR loading YOLO: {e}")
            raise

    def _init_robot(self):
        """Initialize RoboMaster robot"""
        print("Connecting to RoboMaster robot…")
        try:
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type="ap")
            self.ep_robot.set_robot_mode(mode=robot.FREE)
            print("✓ RoboMaster connected\n")
            self.chassis = chassis.Chassis(self.ep_robot)
            self.arm = self.ep_robot.robotic_arm
        except Exception as e:
            print(f"✗ ERROR connecting to robot: {e}")
            raise

    def _init_ir_sensor(self):
        """Initialize IR sensor"""
        try:
            self.ir_sensor = self.ep_robot.sensor
            # Subscribe to distance sensor
            self.ir_sensor.sub_distance(freq=5, callback=self._ir_distance_callback)
            print("✓ IR sensor initialized")
        except Exception as e:
            print(f"⚠ IR sensor not available: {e}")
            self.ir_sensor = None

    def _ir_distance_callback(self, distance_info):
        """Callback for IR distance sensor"""
        if distance_info:
            distance_mm = distance_info[0]
            self.inf_distance = distance_mm / 1000.0

    def _init_camera(self):
        """Initialize camera"""
        print("Starting camera stream…")
        try:
            self.ep_robot.camera.start_video_stream(
                display=False,
                resolution=camera.STREAM_720P
            )
            print("✓ Camera stream started (720P)\n")
        except Exception as e:
            print(f"✗ ERROR starting camera: {e}")

    def _init_modes(self):
        """Initialize mode controllers"""
        self.manual_mode = ManualMode(self)
        self.auto_mode = AutoMode(self)
        self.track_mode = TrackMode(self)
        self.pid_track_mode = PidTrackMode(self)  # NEW: Initialize PID Track Mode
        
        print("✓ Mode controllers initialized")

    def _start_threads(self):
        """Start worker threads"""
        self.camera_thread = threading.Thread(
            target=self._camera_worker,
            daemon=True
        )
        self.camera_thread.start()

        self.keyboard_thread = threading.Thread(
            target=self._keyboard_listener_worker,
            daemon=True
        )
        self.keyboard_thread.start()

        print("✓ Worker threads started\n")

    def _camera_worker(self):
        """Camera worker thread"""
        while self.is_running:
            try:
                frame = self.ep_robot.camera.read_cv2_image()
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
            except Exception as e:
                time.sleep(0.005)

    def _keyboard_listener_worker(self):
        """Keyboard listener thread"""
        while self.is_running:
            # Movement keys
            self.movement['forward'] = keyboard.is_pressed('w')
            self.movement['backward'] = keyboard.is_pressed('s')
            self.movement['left'] = keyboard.is_pressed('d')
            self.movement['right'] = keyboard.is_pressed('a')
            self.movement['rotate_left'] = keyboard.is_pressed('c')
            self.movement['rotate_right'] = keyboard.is_pressed('z')

            # Arm keys
            self.arm_up_flag = keyboard.is_pressed('up')
            self.arm_down_flag = keyboard.is_pressed('down')
            self.arm_left_flag = keyboard.is_pressed('left')
            self.arm_right_flag = keyboard.is_pressed('right')

            # Toggle detection
            if keyboard.is_pressed('space'):
                now = time.time()
                if now - self.last_space_press > self.cooldown:
                    self.object_detection_enabled = not self.object_detection_enabled
                    self.last_space_press = now
                    status = "ON" if self.object_detection_enabled else "OFF"
                    print(f"Object Detection: {status}")

            # Cycle modes (UPDATED: Now cycles through 4 modes)
            if keyboard.is_pressed('t'):
                now = time.time()
                if now - self.last_mode_press > self.cooldown:
                    # Deactivate current mode
                    current_mode = self.modes[self.current_mode_idx]
                    if current_mode == 'TRACK':
                        self.track_mode.set_active(False)
                    elif current_mode == 'PID_TRACK':  # NEW
                        self.pid_track_mode.set_active(False)
                    
                    # Switch to new mode
                    self.previous_mode_idx = self.current_mode_idx
                    self.current_mode_idx = (self.current_mode_idx + 1) % 4  # UPDATED: 4 modes now
                    self.last_mode_press = now
                    new_mode = self.modes[self.current_mode_idx]
                    print(f"Mode changed to: {new_mode}")
                    
                    # Activate new mode
                    if new_mode == 'TRACK':
                        self.track_mode.set_active(True)
                    elif new_mode == 'PID_TRACK':  # NEW
                        self.pid_track_mode.set_active(True)

            # Toggle bounding boxes
            if keyboard.is_pressed('ctrl'):
                now = time.time()
                if now - self.last_ctrl_press > self.cooldown:
                    if self.object_detection_enabled:
                        self.show_bounding_boxes = not self.show_bounding_boxes
                        print("Bounding Boxes:", self.show_bounding_boxes)
                    self.last_ctrl_press = now

            # Toggle lines
            if keyboard.is_pressed('alt'):
                now = time.time()
                if now - self.last_alt_press > self.cooldown:
                    if self.object_detection_enabled:
                        self.show_lines = not self.show_lines
                        print("Lines:", self.show_lines)
                    self.last_alt_press = now

            # Speed adjustments
            if keyboard.is_pressed('1'):
                self.speed = max(0.1, self.speed - 0.1)
                print(f"Speed: {self.speed:.1f}")
            if keyboard.is_pressed('2'):
                self.speed = min(2.0, self.speed + 0.1)
                print(f"Speed: {self.speed:.1f}")

            # Danger zone controls
            self._handle_danger_zone_controls()
            
            # Ignore zone controls
            self._handle_ignore_zone_controls()

            time.sleep(0.02)

    def _handle_danger_zone_controls(self):
        """Handle danger zone control keys"""
        # 5: Decrease size
        if keyboard.is_pressed('5'):
            now = time.time()
            if now - self.last_5_press > self.cooldown:
                self.danger_zone_size = max(10, self.danger_zone_size - 10)
                self.last_5_press = now
                print(f"Danger Zone Size: {self.danger_zone_size}x{self.danger_zone_size}")
        
        # 6: Increase size
        if keyboard.is_pressed('6'):
            now = time.time()
            if now - self.last_6_press > self.cooldown:
                self.danger_zone_size = min(1000, self.danger_zone_size + 10)
                self.last_6_press = now
                print(f"Danger Zone Size: {self.danger_zone_size}x{self.danger_zone_size}")
        
        # 7: Move left
        if keyboard.is_pressed('7'):
            now = time.time()
            if now - self.last_7_press > self.cooldown:
                self.danger_zone_x -= 10
                self.last_7_press = now
                print(f"Danger Zone X offset: {self.danger_zone_x} (Left)")
        
        # 8: Move right
        if keyboard.is_pressed('8'):
            now = time.time()
            if now - self.last_8_press > self.cooldown:
                self.danger_zone_x += 10
                self.last_8_press = now
                print(f"Danger Zone X offset: {self.danger_zone_x} (Right)")
        
        # 9: Move up
        if keyboard.is_pressed('9'):
            now = time.time()
            if now - self.last_9_press > self.cooldown:
                self.danger_zone_y -= 10
                self.last_9_press = now
                print(f"Danger Zone Y offset: {self.danger_zone_y} (Up)")
        
        # 0: Move down
        if keyboard.is_pressed('0'):
            now = time.time()
            if now - self.last_0_press > self.cooldown:
                self.danger_zone_y += 10
                self.last_0_press = now
                print(f"Danger Zone Y offset: {self.danger_zone_y} (Down)")

    def _handle_ignore_zone_controls(self):
        """Handle ignore zone control keys"""
        if keyboard.is_pressed('['):
            self.ignore_zone_width = max(20, self.ignore_zone_width - 10)
            print(f"Ignore Zone Width: {self.ignore_zone_width}")
        if keyboard.is_pressed(']'):
            self.ignore_zone_width = min(2000, self.ignore_zone_width + 10)
            print(f"Ignore Zone Width: {self.ignore_zone_width}")
        if keyboard.is_pressed(';'):
            self.ignore_zone_height = max(20, self.ignore_zone_height - 10)
            print(f"Ignore Zone Height: {self.ignore_zone_height}")
        if keyboard.is_pressed("'"):
            self.ignore_zone_height = min(1000, self.ignore_zone_height + 10)
            print(f"Ignore Zone Height: {self.ignore_zone_height}")
        if keyboard.is_pressed('i'):
            self.ignore_zone_enabled = not self.ignore_zone_enabled
            print(f"Ignore Zone: {'ON' if self.ignore_zone_enabled else 'OFF'}")

    def update_movement(self):
        """Update movement based on current mode"""
        current_mode = self.modes[self.current_mode_idx]
        
        if current_mode == 'MANUAL':
            return self.manual_mode.update()
        elif current_mode == 'AUTO':
            return self.auto_mode.update()
        elif current_mode == 'TRACK':
            # Ensure TRACK mode is active
            if not self.track_mode.active:
                self.track_mode.set_active(True)
            return self.track_mode.update()
        elif current_mode == 'PID_TRACK':  # NEW
            # Ensure PID_TRACK mode is active
            if not self.pid_track_mode.active:
                self.pid_track_mode.set_active(True)
            return self.pid_track_mode.update()
        
        return 0, 0, 0

    def update_arm(self):
        """Update arm movement"""
        try:
            if self.arm is not None:
                if self.arm_up_flag:
                    self.arm.move(x=0, y=self.arm_step).wait_for_completed()
                    self.arm_position_y += self.arm_step
                    time.sleep(0.15)
                if self.arm_down_flag:
                    self.arm.move(x=0, y=-self.arm_step).wait_for_completed()
                    self.arm_position_y -= self.arm_step
                    time.sleep(0.15)
                if self.arm_left_flag:
                    self.arm.move(x=-self.arm_step, y=0).wait_for_completed()
                    self.arm_position_x -= self.arm_step
                    time.sleep(0.15)
                if self.arm_right_flag:
                    self.arm.move(x=self.arm_step, y=0).wait_for_completed()
                    self.arm_position_x += self.arm_step
                    time.sleep(0.15)
        except Exception as e:
            print("Arm movement error:", e)

    def draw_interface(self, frame):
        """Draw interface elements common to all modes"""
        h, w = frame.shape[:2]
        
        # Draw ignore zone
        if self.ignore_zone_enabled:
            self._draw_ignore_zone(frame)
        
        # Draw danger zone if detection enabled
        if self.object_detection_enabled:
            self._draw_danger_zone(frame)
        
        # Draw overlay info
        self._draw_overlay_info(frame)
        
        # Update FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 0.5:
            self.current_fps = self.frame_count / (now - self.last_fps_time)
            self.last_fps_time = now
            self.frame_count = 0

    def _draw_ignore_zone(self, frame):
        """Draw ignore zone"""
        h, w = frame.shape[:2]
        cx = w // 2

        x1 = int(cx - self.ignore_zone_width // 2)
        y1 = int(h - self.ignore_zone_height)
        x2 = int(cx + self.ignore_zone_width // 2)
        y2 = h

        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.ignore_zone_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def _draw_danger_zone(self, frame):
        """Draw danger zone square"""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        half_size = self.danger_zone_size // 2
        x1 = center_x + self.danger_zone_x - half_size
        y1 = center_y + self.danger_zone_y - half_size
        x2 = center_x + self.danger_zone_x + half_size
        y2 = center_y + self.danger_zone_y + half_size
        
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                     self.danger_zone_color, -1)
        cv2.addWeighted(overlay, self.danger_zone_alpha, frame, 
                       1 - self.danger_zone_alpha, 0, frame)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 0, 255), 2)
        
        # Draw cross at center
        cross_x = int(center_x + self.danger_zone_x)
        cross_y = int(center_y + self.danger_zone_y)
        cross_size = 10
        
        cv2.line(frame, (cross_x - cross_size, cross_y), 
                (cross_x + cross_size, cross_y), (0, 0, 255), 1)
        cv2.line(frame, (cross_x, cross_y - cross_size), 
                (cross_x, cross_y + cross_size), (0, 0, 255), 1)

    def _draw_overlay_info(self, frame):
        """Draw overlay information"""
        h, w = frame.shape[:2]

        # Control info
        lines = [
            "Controls: WASD ZC (move), Arrow Keys (arm)",
            "SPACE=Detect  CTRL=Boxes  ALT=Lines  O=Quit",
            "T=Cycle Mode  5=Size-  6=Size+  7=Left  8=Right",
            "9=Up  0=Down"
        ]

        y = 30
        for text in lines:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            y += 25

        # Current mode (UPDATED for 4 modes)
        current_mode = self.modes[self.current_mode_idx]
        if current_mode == 'MANUAL':
            mode_color = (0, 0, 255)  # Red
        elif current_mode == 'AUTO':
            mode_color = (0, 255, 0)  # Green
        elif current_mode == 'TRACK':
            mode_color = (0, 255, 255)  # Yellow
        else:  # PID_TRACK
            mode_color = (255, 0, 255)  # Purple
        
        mode_text = f"MODE: {current_mode}"
        cv2.putText(frame, mode_text, (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # System info
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                    (w - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        cv2.putText(frame, f"Speed: {self.speed:.1f}",
                    (w - 150, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # Danger zone info
        danger_zone_text = f"Danger Zone: {self.danger_zone_size}x{self.danger_zone_size}"
        cv2.putText(frame, danger_zone_text, (w - 250, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        position_text = f"Pos: X={self.danger_zone_x} Y={self.danger_zone_y}"
        cv2.putText(frame, position_text, (w - 250, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # IR distance
        if self.inf_distance is not None:
            ir_text = f"IR Distance: {self.inf_distance:.2f} m"
        else:
            ir_text = "IR: ---"
        
        cv2.putText(frame, ir_text, (w - 250, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detection status
        det_color = (0, 255, 0) if self.object_detection_enabled else (0, 0, 255)
        cv2.putText(frame,
                    f"DETECTION: {'ON' if self.object_detection_enabled else 'OFF'}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    det_color, 2)
        
        # Bounding boxes status
        if self.object_detection_enabled:
            boxes_color = (0, 255, 0) if self.show_bounding_boxes else (0, 0, 255)
            cv2.putText(frame,
                        f"BOXES: {'ON' if self.show_bounding_boxes else 'OFF'}",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        boxes_color, 2)
        
        # Lines status
        if self.object_detection_enabled:
            lines_color = (0, 255, 0) if self.show_lines else (0, 0, 255)
            cv2.putText(frame,
                        f"LINES: {'ON' if self.show_lines else 'OFF'}",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        lines_color, 2)

        # TRACK mode specific status
        if current_mode == 'TRACK':
            track_status = "ACTIVE" if self.track_mode.active else "INACTIVE"
            track_color = (0, 255, 0) if self.track_mode.active else (0, 0, 255)
            cv2.putText(frame, f"TRACK: {track_status}", (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)
        
        # PID_TRACK mode specific status (NEW)
        if current_mode == 'PID_TRACK':
            track_status = "ACTIVE" if self.pid_track_mode.active else "INACTIVE"
            track_color = (0, 255, 0) if self.pid_track_mode.active else (0, 0, 255)
            cv2.putText(frame, f"PID_TRACK: {track_status}", (10, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)

    def show_camera_feed(self):
        """Display camera feed with overlay"""
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame
        
        display_frame = frame.copy()
        
        # Draw common interface
        self.draw_interface(display_frame)
        
        # Let current mode draw its specific elements
        current_mode = self.modes[self.current_mode_idx]
        if current_mode == 'MANUAL':
            self.manual_mode.draw(display_frame)
        elif current_mode == 'AUTO':
            self.auto_mode.draw(display_frame)
        elif current_mode == 'TRACK':
            self.track_mode.draw(display_frame)
        elif current_mode == 'PID_TRACK':  # NEW
            self.pid_track_mode.draw(display_frame)
        
        # Resize if needed
        h, w = display_frame.shape[:2]
        if w > 1280 or h > 720:
            display_frame = cv2.resize(display_frame, (1280, 720))
        
        cv2.imshow("Robomaster - 4 Mode Control", display_frame)

    def cleanup(self):
        """Clean up all resources properly"""
        print("\nCleaning up resources...")
        
        try:
            # Clean up TRACK mode resources
            if hasattr(self, 'track_mode') and self.track_mode:
                if hasattr(self.track_mode, 'set_active'):
                    self.track_mode.set_active(False)
                if hasattr(self.track_mode, 'cleanup'):
                    self.track_mode.cleanup()
                print("✓ Track mode resources cleaned")
            
            # Clean up PID_TRACK mode resources (NEW)
            if hasattr(self, 'pid_track_mode') and self.pid_track_mode:
                if hasattr(self.pid_track_mode, 'set_active'):
                    self.pid_track_mode.set_active(False)
                if hasattr(self.pid_track_mode, 'cleanup'):
                    self.pid_track_mode.cleanup()
                print("✓ PID Track mode resources cleaned")
            
            # Unsubscribe from sensors
            if hasattr(self, 'ir_sensor') and self.ir_sensor:
                try:
                    self.ir_sensor.unsub_distance()
                    print("✓ IR sensor unsubscribed")
                except:
                    pass
            
            # Stop camera stream
            if hasattr(self, 'ep_robot') and self.ep_robot.camera:
                try:
                    self.ep_robot.camera.stop_video_stream()
                    print("✓ Camera stream stopped")
                except:
                    pass
            
            # Close robot connection
            if hasattr(self, 'ep_robot'):
                try:
                    self.ep_robot.close()
                    print("✓ Robot connection closed")
                except:
                    pass
                
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        
        cv2.destroyAllWindows()
        print("✓ Cleanup completed")

    def run(self):
        """Main loop"""
        print("Starting main loop…")
        print("Press 'o' to quit")
        print("\nControls:")
        print("  W/S/A/D/Z/C - Manual movement")
        print("  Arrow keys - Arm control")
        print("  SPACE - Toggle object detection")
        print("  T - Cycle modes (MANUAL → AUTO → TRACK → PID_TRACK)")  # UPDATED
        print("  CTRL - Toggle bounding boxes")
        print("  ALT - Toggle lines")
        print("  1/2 - Decrease/Increase speed")
        print("\nDanger Zone Controls:")
        print("  5 - Decrease square size")
        print("  6 - Increase square size")
        print("  7 - Move square left")
        print("  8 - Move square right")
        print("  9 - Move square up")
        print("  0 - Move square down")
        print("\nModes:")
        print("  MANUAL    - Full keyboard control")
        print("  AUTO      - Autonomous obstacle avoidance")
        print("  TRACK     - Basic person following (Robot's internal detection)")
        print("  PID_TRACK - Advanced PID tracking with fast movement response")  # NEW
        
        if not DEEPSORT_AVAILABLE:
            print("\n⚠ WARNING: DeepSORT not installed!")
            print("AUTO mode requires DeepSORT for best performance.")
            print("Install with: pip install deep-sort-realtime")

        try:
            while self.is_running:
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    print("\nExit requested via 'o' key")
                    self.is_running = False
                    break
                
                # Get movement from current mode
                x_speed, y_speed, z_speed = self.update_movement()
                
                # Apply movement
                try:
                    self.chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)
                except:
                    pass
                
                # Update arm
                self.update_arm()
                
                # Show camera feed
                self.show_camera_feed()

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.is_running = False
            self.cleanup()

if __name__ == "__main__":
    controller = OptimizedYOLORobotController()
    controller.run()