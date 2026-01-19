import math
import cv2
import numpy as np
from collections import deque
import time


class PersonInfo:
    """Convert normalized bounding box to pixel coordinates"""
    
    def __init__(self, x, y, w, h, image_width=1280, image_height=720):
        self._x = x  # Normalized center x (0-1)
        self._y = y  # Normalized center y (0-1)
        self._w = w  # Normalized width (0-1)
        self._h = h  # Normalized height (0-1)
        self.image_width = image_width
        self.image_height = image_height
        self.timestamp = time.time()

    @property
    def pt1(self):
        """Top-left corner of bounding box in pixels"""
        return (
            int((self._x - self._w / 2) * self.image_width),
            int((self._y - self._h / 2) * self.image_height),
        )

    @property
    def pt2(self):
        """Bottom-right corner of bounding box in pixels"""
        return (
            int((self._x + self._w / 2) * self.image_width),
            int((self._y + self._h / 2) * self.image_height),
        )

    @property
    def center(self):
        """Center of bounding box in pixels"""
        return (
            int(self._x * self.image_width),
            int(self._y * self.image_height),
        )

    @property
    def width_pixels(self):
        """Width in pixels"""
        return self.pt2[0] - self.pt1[0]

    @property
    def height_pixels(self):
        """Height in pixels"""
        return self.pt2[1] - self.pt1[1]


class PIDController:
    """PID Controller for smooth and responsive movement"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, windup_guard=20.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.windup_guard = windup_guard  # Anti-windup
        
        # State variables
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
        
        # Output limits
        self.output_min = -1.0
        self.output_max = 1.0
        
    def update(self, error, dt=None):
        """Update PID controller and return output"""
        
        # Calculate time delta
        current_time = time.time()
        if dt is None:
            dt = current_time - self.previous_time
            dt = max(dt, 0.001)  # Minimum 1ms to avoid division by zero
        self.previous_time = current_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        # Apply windup guard
        self.integral = max(min(self.integral, self.windup_guard), -self.windup_guard)
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            d_error = (error - self.previous_error) / dt
        else:
            d_error = 0
        d_term = self.kd * d_error
        self.previous_error = error
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(min(output, self.output_max), self.output_min)
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()


class PersonTracker:
    """Advanced person tracker with motion prediction"""
    
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.person_history = deque(maxlen=max_history)
        self.velocity_filter = deque(maxlen=5)
        self.last_update_time = time.time()
        
        # Motion tracking
        self.predicted_position = None
        self.predicted_velocity = (0, 0)
        
    def update(self, person):
        """Update tracker with new person detection"""
        current_time = time.time()
        
        if person is None:
            # No person detected
            self.person_history.append(None)
            return None
        
        # Calculate velocity if we have previous detection
        if len(self.person_history) > 0 and self.person_history[-1] is not None:
            last_person = self.person_history[-1]
            dt = current_time - last_person.timestamp
            
            if dt > 0:
                # Calculate pixel velocity
                last_center = last_person.center
                current_center = person.center
                
                vx = (current_center[0] - last_center[0]) / dt
                vy = (current_center[1] - last_center[1]) / dt
                
                # Filter velocity
                self.velocity_filter.append((vx, vy))
                
                # Average velocity
                if self.velocity_filter:
                    vx_avg = sum(v[0] for v in self.velocity_filter) / len(self.velocity_filter)
                    vy_avg = sum(v[1] for v in self.velocity_filter) / len(self.velocity_filter)
                    self.predicted_velocity = (vx_avg, vy_avg)
                    
                    # Predict next position
                    prediction_time = 0.1  # 100ms ahead
                    px = current_center[0] + vx_avg * prediction_time
                    py = current_center[1] + vy_avg * prediction_time
                    self.predicted_position = (int(px), int(py))
        
        # Store in history
        self.person_history.append(person)
        self.last_update_time = current_time
        
        return person
    
    def get_motion_speed(self):
        """Calculate how fast the person is moving (pixels per second)"""
        if len(self.velocity_filter) < 2:
            return 0.0
        
        # Calculate speed magnitude from recent velocities
        speeds = [math.sqrt(vx**2 + vy**2) for vx, vy in self.velocity_filter]
        avg_speed = sum(speeds) / len(speeds)
        
        return avg_speed
    
    def get_current_person(self):
        """Get most recent person detection"""
        if not self.person_history or self.person_history[-1] is None:
            return None
        return self.person_history[-1]
    
    def is_tracking(self):
        """Check if we're actively tracking a person"""
        if not self.person_history:
            return False
        
        # Count recent detections
        recent_detections = sum(1 for p in list(self.person_history)[-5:] if p is not None)
        return recent_detections >= 2  # Need at least 2 detections in last 5 frames


class PidTrackMode:
    """PID-based TRACK mode - FIXED VERSION: Stop at target, no backward when person rests"""
    
    def __init__(self, controller):
        self.controller = controller
        
        # Target following distance (meters)
        self.target_distance = 1.5  # Comfortable following distance
        
        # Detected persons
        self.persons = []
        self.tracked_person = None
        
        # Person tracker
        self.person_tracker = PersonTracker()
        
        # PID Controllers - Tuned for smooth following
        self.distance_pid = PIDController(kp=1.2, ki=0.01, kd=0.2)  # Gentle distance control
        self.angle_pid = PIDController(kp=0.025, ki=0.0005, kd=0.015)  # Smooth rotation
        
        # Movement limits
        self.max_forward_speed = 0.7
        self.max_backward_speed = -0.15  # Very limited backward movement
        self.max_rotation_speed = 60
        self.min_forward_speed = 0.05  # Minimum forward speed
        
        # Distance zones
        self.comfort_zone_min = 1.2  # Start slowing down
        self.comfort_zone_max = 1.8  # Start speeding up
        self.emergency_distance = 0.25  # Only move back in emergency
        
        # Fast movement tracking
        self.fast_movement_threshold = 40  # pixels/sec
        self.speed_boost_factor = 2.5  # Max speed boost
        
        # Subscription state
        self.subscribed = False
        
        # Mode state
        self.active = False
        
        # Camera geometry (calibration)
        self.real_width_of_person = 0.6  # meters (average shoulder width)
        self.focal_length = 615  # Focal length in pixels
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.fps = 0
        self.tracking_start_time = None
        
        # State tracking
        self.last_outputs = (0, 0, 0)
        self.current_state = "SEARCHING"
        
        print("✓ PID Track Mode initialized (inactive)")

    # ==========================================================
    # MODE LIFECYCLE MANAGEMENT
    # ==========================================================
    
    def set_active(self, active):
        """Activate or deactivate PID TRACK mode"""
        if active and not self.active:
            self._activate()
        elif not active and self.active:
            self._deactivate()
    
    def _activate(self):
        """Activate PID TRACK mode"""
        if self.subscribed:
            return
        
        try:
            print("🔄 Activating PID TRACK mode...")
            self.controller.ep_robot.vision.sub_detect_info(
                name="person",
                callback=self._on_detect_person,
            )
            self.subscribed = True
            self.active = True
            self.tracking_start_time = time.time()
            
            # Reset PID controllers
            self.distance_pid.reset()
            self.angle_pid.reset()
            
            print("✅ PID TRACK MODE: Person detection subscribed")
            print("📏 Target Distance: 1.5m")
            print("⏹️  Will STOP at target distance")
            print("⚡ Will speed up when person moves fast")
            
        except Exception as e:
            print(f"❌ PID TRACK MODE: Subscription failed: {e}")
            self._on_detect_person = self._dummy_person_detection
            self.subscribed = True
            self.active = True
            self.tracking_start_time = time.time()
            print("⚠️  PID TRACK MODE: Using dummy detection")
    
    def _deactivate(self):
        """Deactivate PID TRACK mode"""
        if not self.subscribed:
            return
        
        try:
            print("🔄 Deactivating PID TRACK mode...")
            self.controller.ep_robot.vision.unsub_detect_info(name="person")
            print("✅ PID TRACK MODE: Person detection unsubscribed")
        except Exception as e:
            print(f"⚠️  PID TRACK MODE: Unsubscription warning: {e}")
        
        self.persons.clear()
        self.tracked_person = None
        self.subscribed = False
        self.active = False
        self.tracking_start_time = None
    
    def cleanup(self):
        """Clean up resources (called on program exit)"""
        self._deactivate()
        print("✅ PID TRACK MODE: Cleanup complete")

    # ==========================================================
    # PERSON DETECTION CALLBACKS
    # ==========================================================
    
    def _dummy_person_detection(self, person_info):
        """Fallback detection for testing"""
        self.persons = [PersonInfo(0.5, 0.5, 0.2, 0.4)]
    
    def _on_detect_person(self, person_info):
        """Callback for RoboMaster's internal person detection"""
        self.persons.clear()
        
        if not person_info or len(person_info) == 0:
            return
        
        try:
            for i in range(len(person_info)):
                x, y, w, h = person_info[i]
                
                # Create PersonInfo object
                person = PersonInfo(x, y, w, h)
                self.persons.append(person)
                
                # Track the largest person (closest)
                if i == 0:
                    self.tracked_person = self.person_tracker.update(person)
                
        except Exception as e:
            print(f"⚠️  Person detection parsing error: {e}")
            self.persons = []
            self.tracked_person = self.person_tracker.update(None)

    # ==========================================================
    # DISTANCE AND ANGLE CALCULATIONS
    # ==========================================================
    
    def calculate_distance(self, person):
        """Calculate distance to person using camera geometry"""
        try:
            pixel_width_of_person = person.width_pixels
            
            if pixel_width_of_person <= 0:
                return self.target_distance
            
            # Calculate actual distance using similar triangles
            distance_to_person = (self.real_width_of_person * self.focal_length) / pixel_width_of_person
            
            # Clamp to reasonable range
            return max(0.2, min(distance_to_person, 10.0))
            
        except Exception as e:
            print(f"⚠️  Distance calculation error: {e}")
            return self.target_distance
    
    def calculate_angle(self, person):
        """Calculate angle to person relative to robot center"""
        try:
            frame = self.controller.latest_frame
            if frame is None:
                return 0.0
            
            # Get person's horizontal center position
            person_center_x, _ = person.center
            
            # Calculate horizontal offset from image center
            image_center_x = frame.shape[1] // 2
            relative_position_pix = person_center_x - image_center_x
            
            # Normalize to -1 to 1 range
            max_offset = frame.shape[1] // 2
            normalized_offset = relative_position_pix / max_offset
            
            return normalized_offset * 45  # Convert to approximate angle in degrees
            
        except Exception as e:
            print(f"⚠️  Angle calculation error: {e}")
            return 0.0

    # ==========================================================
    # MOVEMENT CONTROL - FIXED: Stop at target, no backward when person rests
    # ==========================================================
    
    def calculate_adaptive_speed(self, distance, person_speed):
        """Calculate adaptive speed based on distance and person speed"""
        
        # Base speed calculation based on distance error
        distance_error = distance - self.target_distance
        
        # If within comfort zone, reduce speed
        if self.comfort_zone_min <= distance <= self.comfort_zone_max:
            # In comfort zone - move very slowly or stop
            base_speed = distance_error * 0.3  # Very gentle
        else:
            # Outside comfort zone - normal speed
            base_speed = distance_error * 0.8
        
        # Apply speed boost based on person movement
        speed_boost = 1.0
        if person_speed > self.fast_movement_threshold:
            # Person is moving fast - boost speed
            speed_factor = min(self.speed_boost_factor, 
                              1.0 + (person_speed - self.fast_movement_threshold) / 50)
            speed_boost = speed_factor
        
        # Combine base speed with boost
        adaptive_speed = base_speed * speed_boost
        
        return adaptive_speed, speed_boost
    
    def update(self):
        """Return (x_speed, y_speed, z_speed) for chassis movement"""
        
        # If PID TRACK mode is not active, do nothing
        if not self.active:
            return 0, 0, 0
        
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = current_time
        
        # EMERGENCY ONLY: IR sensor indicates dangerously close
        if (self.controller.inf_distance is not None
            and self.controller.inf_distance < self.emergency_distance):
            print(f"🚨 EMERGENCY: Too close ({self.controller.inf_distance:.2f} m) - Moving back")
            return -0.15, 0, 0  # Move backward slowly ONLY in emergency
        
        # No person detected or lost tracking: search behavior
        if not self.persons or not self.person_tracker.is_tracking():
            self.current_state = "SEARCHING"
            self.distance_pid.reset()
            self.angle_pid.reset()
            
            # Gentle search pattern
            if self.tracking_start_time:
                search_time = current_time - self.tracking_start_time
            else:
                search_time = current_time
            search_speed = 20 + math.sin(search_time * 0.3) * 15
            
            return 0.1, 0, search_speed
        
        # Get current person and motion information
        person = self.tracked_person
        person_speed = self.person_tracker.get_motion_speed()
        
        # Calculate distance and angle
        distance = self.calculate_distance(person)
        angle_error = self.calculate_angle(person)
        
        # ============================================
        # DISTANCE CONTROL - FIXED: Stop at target
        # ============================================
        
        # Calculate distance error
        distance_error = distance - self.target_distance
        
        # Calculate adaptive speed
        adaptive_speed, speed_boost = self.calculate_adaptive_speed(distance, person_speed)
        
        # Apply PID for fine adjustment
        x_speed_pid = self.distance_pid.update(distance_error, dt)
        
        # Combine adaptive speed with PID
        x_speed = adaptive_speed + x_speed_pid * 0.3
        
        # ============================================
        # CRITICAL FIX: DON'T GO BACKWARD WHEN PERSON IS AT REST
        # ============================================
        
        # If person is moving slowly (at rest) and we're close to target
        if person_speed < 20:  # Person is moving slowly (almost at rest)
            if distance < self.target_distance + 0.2:  # We're close to target
                # If PID wants to go backward, prevent it
                if x_speed < 0:
                    x_speed = 0  # STOP, don't go backward
                    self.distance_pid.reset()  # Reset PID to prevent windup
                    print("⏹️  Person at rest - Stopping at target distance")
        
        # If person is moving fast, allow normal operation
        elif person_speed > 40:  # Person is moving fast
            # Boost speed when chasing
            if distance > self.target_distance:
                x_speed *= min(2.0, 1.0 + person_speed / 100)
                self.current_state = "CHASING"
                print(f"⚡ CHASING: Person moving at {person_speed:.0f} px/s")
        
        # Apply speed limits
        x_speed = max(self.max_backward_speed, min(x_speed, self.max_forward_speed))
        
        # Special case: If very close and person at rest, stop completely
        if distance < self.comfort_zone_min and person_speed < 15:
            x_speed = 0
            self.current_state = "STOPPED"
        
        # ============================================
        # ANGLE CONTROL - Smooth rotation
        # ============================================
        
        # Normalize angle error
        normalized_angle_error = angle_error / 45.0
        normalized_angle_error = max(-1.0, min(normalized_angle_error, 1.0))
        
        # Calculate rotation speed using PID
        z_speed = self.angle_pid.update(normalized_angle_error, dt) * 50
        
        # Boost rotation if person is moving laterally fast
        if person_speed > 30 and abs(angle_error) > 10:
            lateral_boost = min(25, person_speed / 4)
            z_speed += lateral_boost if angle_error > 0 else -lateral_boost
        
        # Limit rotation speed
        z_speed = max(-self.max_rotation_speed, min(z_speed, self.max_rotation_speed))
        
        # ============================================
        # LATERAL MOVEMENT - Minor adjustments
        # ============================================
        # Use small lateral movement for fine positioning
        if abs(normalized_angle_error) > 0.3:  # More than 30% off-center
            y_speed = -normalized_angle_error * 0.1  # Small lateral adjustment
            y_speed = max(-0.2, min(y_speed, 0.2))
        else:
            y_speed = 0
        
        # ============================================
        # SMOOTHING AND STATE UPDATE
        # ============================================
        
        # Smooth outputs
        smoothing = 0.4
        x_speed = smoothing * x_speed + (1 - smoothing) * self.last_outputs[0]
        y_speed = smoothing * y_speed + (1 - smoothing) * self.last_outputs[1]
        z_speed = smoothing * z_speed + (1 - smoothing) * self.last_outputs[2]
        
        self.last_outputs = (x_speed, y_speed, z_speed)
        
        # Update state
        if abs(x_speed) < 0.05 and abs(z_speed) < 5:
            self.current_state = "STOPPED"
        elif person_speed > 40:
            self.current_state = "CHASING"
        else:
            self.current_state = "FOLLOWING"
        
        return x_speed, y_speed, z_speed

    # ==========================================================
    # VISUALIZATION
    # ==========================================================
    
    def draw(self, frame):
        """Draw PID TRACK mode specific elements on the frame"""
        
        if not self.active:
            return
        
        h, w = frame.shape[:2]
        
        # Draw all detected persons
        for i, person in enumerate(self.persons):
            # Color based on tracking status
            if person == self.tracked_person:
                color = (0, 255, 0)  # Green for tracked person
                thickness = 3
            else:
                color = (0, 165, 255)  # Orange for other persons
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, person.pt1, person.pt2, color, thickness)
            
            # Draw center point
            center_x, center_y = person.center
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw predicted position if available
            if (person == self.tracked_person and 
                self.person_tracker.predicted_position is not None):
                pred_x, pred_y = self.person_tracker.predicted_position
                cv2.circle(frame, (pred_x, pred_y), 8, (255, 255, 0), 2)
                cv2.line(frame, (center_x, center_y), (pred_x, pred_y), 
                        (255, 255, 0), 2)
            
            # Calculate and display distance and angle
            distance = self.calculate_distance(person)
            angle = self.calculate_angle(person)
            
            # Calculate motion speed
            motion_speed = self.person_tracker.get_motion_speed()
            
            # Create info text
            info_text = f"Person: {distance:.1f}m, {angle:.0f}°, {motion_speed:.0f}px/s"
            
            # Draw text above bounding box
            cv2.putText(
                frame,
                info_text,
                (person.pt1[0], person.pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness,
            )
            
            # Draw line from person center to image center
            cv2.line(
                frame,
                (center_x, center_y),
                (w // 2, h // 2),
                (0, 255, 255),  # Yellow line
                2,
            )
        
        # Draw mode status
        status_text = f"PID TRACK: {self.current_state}"
        
        # Color based on state
        if self.current_state == "STOPPED":
            status_color = (0, 255, 255)  # Yellow
        elif self.current_state == "CHASING":
            status_color = (0, 100, 255)  # Blue
        elif self.current_state == "FOLLOWING":
            status_color = (0, 255, 0)  # Green
        else:  # SEARCHING
            status_color = (0, 165, 255)  # Orange
        
        cv2.putText(
            frame,
            status_text,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        
        # Draw distance info
        if self.tracked_person:
            distance = self.calculate_distance(self.tracked_person)
            person_speed = self.person_tracker.get_motion_speed()
            
            distance_text = f"Distance: {distance:.1f}m | Target: {self.target_distance}m"
            speed_text = f"Person Speed: {person_speed:.0f} px/s"
            
            cv2.putText(
                frame,
                distance_text,
                (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),  # Yellow
                2,
            )
            
            cv2.putText(
                frame,
                speed_text,
                (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),  # Cyan
                2,
            )
        
        # Draw comfort zone
        comfort_color = (0, 255, 0, 50)  # Semi-transparent green
        comfort_text = f"Comfort Zone: {self.comfort_zone_min}-{self.comfort_zone_max}m"
        cv2.putText(
            frame,
            comfort_text,
            (w - 300, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        
        # Draw speed visualization
        self._draw_speed_visualization(frame, w, h)
    
    def _draw_speed_visualization(self, frame, w, h):
        """Draw speed visualization bars"""
        
        viz_width = 200
        viz_height = 60
        viz_x = w - viz_width - 20
        viz_y = 20
        
        # Draw background
        cv2.rectangle(frame, (viz_x, viz_y), 
                     (viz_x + viz_width, viz_y + viz_height), 
                     (50, 50, 50), -1)
        
        # Get current speeds
        x_speed, y_speed, z_speed = self.last_outputs
        
        # Draw forward/backward speed bar
        bar_width = int(abs(x_speed) * 80)
        if x_speed >= 0:
            bar_color = (0, 255, 0)  # Green for forward
            bar_x = viz_x + 10
        else:
            bar_color = (0, 0, 255)  # Red for backward
            bar_x = viz_x + 10 - bar_width
        
        cv2.rectangle(frame, (bar_x, viz_y + 10),
                     (viz_x + 10 + bar_width if x_speed >= 0 else viz_x + 10, viz_y + 20),
                     bar_color, -1)
        
        # Draw speed text
        speed_text = f"Speed: {x_speed:.2f} m/s"
        cv2.putText(frame, speed_text, (viz_x + 10, viz_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw state indicator
        state_text = f"State: {self.current_state}"
        cv2.putText(frame, state_text, (viz_x + 10, viz_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# Simple test
if __name__ == "__main__":
    print("PID Track Mode - Fixed Version")
    print("Key Features:")
    print("1. ⏹️  STOPS at target distance when person is at rest")
    print("2. 🚫 Does NOT go backward when person is close (unless emergency)")
    print("3. ⚡ Speeds up when person moves fast")
    print("4. 📏 Maintains comfortable following distance")
    print("✅ Ready for integration with main.py")