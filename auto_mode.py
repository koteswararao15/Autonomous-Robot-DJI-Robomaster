import cv2
import time
import threading
import queue
import torch
import numpy as np
import math

class AutoMode:
    def __init__(self, controller):
        self.controller = controller
        self.obstacle_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 
                                'motorcycle', 'chair', 'bench', 'couch', 'backpack',
                                'handbag', 'suitcase', 'bottle', 'cup', 'bowl']
        
        # Thread safety lock
        self.detections_lock = threading.Lock()
        self.detections = []
        
        # YOLO queue - separate from manual mode
        self.yolo_queue = queue.Queue(maxsize=2)
        
        # Navigation parameters
        self.safe_distance_threshold = 0.3  # meters (IR sensor)
        self.yolo_distance_threshold = 10  # pixels (YOLO distance threshold)
        self.last_avoidance_direction = 0  # -1 = left, 0 = center, 1 = right
        
        # OBSTACLE AVOIDANCE STATES
        self.state = "MOVING_FORWARD"  # States: MOVING_FORWARD, AVOIDING_RIGHT, TURNING_LEFT, RESUME
        self.state_start_time = 0
        self.chosen_direction = 0  # -1 = left, 1 = right
        self.avoidance_reason = ""  # Track why avoidance was triggered
        
        # Movement parameters (NORMAL SPEED)
        self.normal_forward_speed = 0.4  # Normal forward speed
        self.sideways_speed = 0.5  # X-axis movement speed
        self.sideways_duration = 1.0  # How long to move sideways (seconds)
        
        # TURNING parameters (for continuous right movement)
        self.turn_speed = 0.5  # Rotation speed for 90-degree turn
        self.turn_duration = 1.5  # Duration for 90-degree turn (1.5 seconds)
        
        # Detection parameters
        self.early_detection_distance = 30  # Detect obstacles 30 pixels away
        self.danger_zone_expansion = 2.0  # Larger detection area
        
        # SIMPLIFIED Wall/Corner detection - ONLY RIGHT SIDE
        self.consecutive_right_count = 0  # Count consecutive right movements
        self.consecutive_threshold = 2  # 2 consecutive right movements triggers LEFT turn
        
        # Initialize DeepSORT tracker
        self._init_deepsort()
        
        # FPS tracking
        self.last_detection_time = 0
        self.detection_fps = 0
        self.frame_count = 0
        
        # Distance estimation parameters
        self.focal_length = 615  # Focal length in pixels (adjust based on your camera)
        self.known_widths = {
            'person': 0.5,      # Average shoulder width in meters
            'car': 1.8,         # Average car width in meters
            'truck': 2.5,       # Truck width
            'bus': 2.5,         # Bus width
            'bicycle': 0.7,     # Bicycle width
            'motorcycle': 0.8,  # Motorcycle width
            'chair': 0.5,       # Chair width
            'bench': 1.5,       # Bench width
            'couch': 2.0,       # Couch width
            'backpack': 0.4,    # Backpack width
            'handbag': 0.3,     # Handbag width
            'suitcase': 0.5,    # Suitcase width
            'bottle': 0.1,      # Bottle width
            'cup': 0.1,         # Cup width
            'bowl': 0.2         # Bowl width
        }
        
        # Start YOLO worker thread
        self.yolo_thread = threading.Thread(
            target=self._yolo_worker,
            daemon=True
        )
        self.yolo_thread.start()
        
        print("✓ Auto Mode initialized with Right-side Avoidance + Turn on 2 Rights")
    
    def _init_deepsort(self):
        """Initialize DeepSORT tracker"""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4,
                nn_budget=None,
                override_track_class=None
            )
            print("✓ Auto Mode: DeepSORT tracker initialized")
        except ImportError:
            print("⚠ Auto Mode: DeepSORT not available")
            self.tracker = None
        except Exception as e:
            print(f"⚠ Auto Mode: DeepSORT init error: {e}")
            self.tracker = None
    
    def update(self):
        """Autonomous navigation with right-side obstacle avoidance"""
        # 1. IR sensor safety check - emergency avoidance if too close
        ir_triggered = False
        if self.controller.inf_distance is not None and self.controller.inf_distance < self.safe_distance_threshold:
            print(f"IR Alert: Too close ({self.controller.inf_distance:.2f} m) - TRIGGERING AVOIDANCE")
            ir_triggered = True
            # Always move RIGHT when IR detects obstacle
            chosen_direction = 1  # Always right
            self._trigger_avoidance("IR_SENSOR", chosen_direction)
            return self._get_state_movement()
        
        # 2. YOLO distance check
        if self.controller.object_detection_enabled:
            yolo_triggered, closest_distance, chosen_direction = self._check_yolo_distances()
            if yolo_triggered:
                print(f"YOLO Alert: Object at {closest_distance:.1f} pixels - TRIGGERING AVOIDANCE")
                # Always move RIGHT when YOLO detects obstacle
                chosen_direction = 1  # Always right
                self._trigger_avoidance("YOLO_DISTANCE", chosen_direction)
                return self._get_state_movement()
        
        # 3. Check for continuous right movement (2 consecutive rights)
        if self.consecutive_right_count >= self.consecutive_threshold:
            print(f"CONTINUOUS RIGHT MOVEMENT DETECTED! {self.consecutive_right_count} consecutive RIGHT movements")
            print("Making 90-degree LEFT turn to break cycle")
            self.state = "TURNING_LEFT"
            self.state_start_time = time.time()
            self.consecutive_right_count = 0  # Reset counter
            return self._get_state_movement()
        
        # 4. State machine navigation (normal operation)
        return self._state_machine_navigation()
    
    def _check_yolo_distances(self):
        """Check if any object is closer than distance threshold"""
        if not self.controller.object_detection_enabled:
            return False, float('inf'), 0
        
        # Get frame for calculations
        frame = self.controller.latest_frame
        if frame is None:
            return False, float('inf'), 0
            
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # Get current detections
        with self.detections_lock:
            current_detections = self.detections.copy()
        
        closest_distance = float('inf')
        closest_obstacle = None
        
        # Check all detections for distance
        for det in current_detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            class_name = det.get("class_name", "")
            
            if class_name not in self.obstacle_classes:
                continue
            
            # Calculate object center
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2
            
            # Calculate distance from center
            distance = math.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
            
            # Update closest obstacle
            if distance < closest_distance:
                closest_distance = distance
                closest_obstacle = det
        
        # Check if closest obstacle is within threshold
        if closest_obstacle and closest_distance < self.yolo_distance_threshold:
            # ALWAYS move RIGHT when obstacle detected
            return True, closest_distance, 1  # Always return 1 (right)
        
        return False, closest_distance, 0
    
    def _calculate_avoidance_direction(self):
        """Calculate which direction to avoid - SIMPLIFIED: Always right"""
        # SIMPLIFIED: Always move right when avoiding obstacles
        print("AVOIDANCE DIRECTION: ALWAYS RIGHT")
        return 1  # Always right
    
    def _trigger_avoidance(self, reason, direction):
        """Trigger the avoidance sequence with wall detection tracking"""
        if self.state == "MOVING_FORWARD":
            # Always use AVOIDING_RIGHT state
            self.state = "AVOIDING_RIGHT"
            self.state_start_time = time.time()
            self.chosen_direction = direction
            self.avoidance_reason = reason
            
            # Increment consecutive right counter for wall detection
            self.consecutive_right_count += 1
            
            print(f"AVOIDANCE TRIGGERED by {reason}. Moving RIGHT")
            print(f"Consecutive right movements: {self.consecutive_right_count}/{self.consecutive_threshold}")
    
    def _get_state_movement(self):
        """Get movement commands based on current state"""
        current_time = time.time()
        
        # STATE 1: AVOIDING_RIGHT (always move right when avoiding)
        if self.state == "AVOIDING_RIGHT":
            elapsed = current_time - self.state_start_time
            
            if elapsed < self.sideways_duration:
                # Continue moving right
                print(f"AVOIDING RIGHT: {elapsed:.1f}/{self.sideways_duration}s")
                # Return values: (forward_speed, sideways_speed, rotation_speed)
                return 0, self.sideways_speed, 0
            else:
                # Finished avoiding - switch to RESUME
                self.state = "RESUME"
                self.state_start_time = current_time
                print("RESUMING forward movement")
                return 0, 0, 0
        
        # STATE 2: TURNING_LEFT (90-degree left turn for continuous right movement)
        elif self.state == "TURNING_LEFT":
            elapsed = current_time - self.state_start_time
            
            if elapsed < self.turn_duration:
                # Continue turning LEFT
                progress = elapsed / self.turn_duration
                print(f"TURNING LEFT: {progress*100:.0f}% complete")
                # Return values: (forward_speed, sideways_speed, rotation_speed)
                # For LEFT turn, use NEGATIVE rotation speed
                return 0, 0, -self.turn_speed
            else:
                # Finished turning - resume forward movement
                self.state = "RESUME"
                self.state_start_time = current_time
                print("LEFT TURN COMPLETE - Resuming forward movement")
                return 0, 0, 0
        
        # STATE 3: RESUME (short pause before returning to normal)
        elif self.state == "RESUME":
            elapsed = current_time - self.state_start_time
            
            if elapsed < 0.5:  # Pause for 0.5 seconds
                return 0, 0, 0
            else:
                # Return to normal forward movement
                self.state = "MOVING_FORWARD"
                self.avoidance_reason = ""
                return self.normal_forward_speed, 0, 0
        
        # Default fallback - MOVING_FORWARD
        return self.normal_forward_speed, 0, 0
    
    def _state_machine_navigation(self):
        """State machine for sideways avoidance"""
        # If we're in avoidance or turning states, use avoidance logic
        if self.state != "MOVING_FORWARD":
            return self._get_state_movement()
        
        # Normal forward movement
        return self.normal_forward_speed, 0, 0
    
    # ... [Rest of the code remains the same - _estimate_distance, _yolo_worker, _run_yolo_detection, _apply_deepsort, draw, etc.] ...
    # The rest of the methods remain exactly the same as in the previous code
    
    def _estimate_distance(self, bbox_width, class_name):
        """Estimate distance to object using known width and focal length"""
        if class_name not in self.known_widths:
            return float('inf')
        
        known_width = self.known_widths[class_name]
        if bbox_width <= 0:
            return float('inf')
        
        # Distance = (Known Width * Focal Length) / Perceived Width
        distance = (known_width * self.focal_length) / bbox_width
        return distance
    
    def _yolo_worker(self):
        """YOLO worker thread"""
        while self.controller.is_running:
            try:
                if not self.controller.object_detection_enabled:
                    time.sleep(0.05)
                    continue
                
                # Get frame from queue
                try:
                    frame = self.yolo_queue.get(timeout=0.05)
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                
                # Skip processing if queue is backing up
                if self.yolo_queue.qsize() > 1:
                    continue
                
                # Run YOLO detection
                detections = self._run_yolo_detection(frame)
                
                # Apply DeepSORT tracking if available
                if self.tracker:
                    tracked_detections = self._apply_deepsort(frame, detections)
                else:
                    tracked_detections = detections
                
                # Update detections safely
                with self.detections_lock:
                    self.detections = tracked_detections
                
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_detection_time >= 1.0:
                    self.detection_fps = self.frame_count
                    self.frame_count = 0
                    self.last_detection_time = current_time
                
            except Exception as e:
                print(f"Auto YOLO worker error: {e}")
                time.sleep(0.01)
    
    def _run_yolo_detection(self, frame):
        """Run YOLO detection for obstacle detection"""
        try:
            height, width = frame.shape[:2]
            
            # Use optimal size
            target_size = 320
            ratio = target_size / max(height, width)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            inference_frame = cv2.resize(frame, (new_width, new_height))
            
            # Normal confidence threshold
            with torch.no_grad():
                results = self.controller.model(
                    inference_frame,
                    imgsz=target_size,
                    conf=0.3,
                    iou=0.5,
                    max_det=20,
                    verbose=False,
                    augment=False
                )
            
            detections = []
            scale_x = width / new_width
            scale_y = height / new_height
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale back to original size
                        x1_orig = x1 * scale_x
                        x2_orig = x2 * scale_x
                        y1_orig = y1 * scale_y
                        y2_orig = y2 * scale_y
                        
                        box_width = x2_orig - x1_orig
                        box_height = y2_orig - y1_orig
                        
                        # Filter small detections
                        if box_width > 20 and box_height > 20:
                            class_id = int(box.cls[0])
                            class_name = self.controller.model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            # Only include obstacle classes
                            if class_name in self.obstacle_classes:
                                # Calculate estimated distance
                                estimated_distance = self._estimate_distance(box_width, class_name)
                                
                                # Calculate center for distance check
                                obj_center_x = (x1_orig + x2_orig) / 2
                                obj_center_y = (y1_orig + y2_orig) / 2
                                center_x = width / 2
                                center_y = height / 2
                                pixel_distance = math.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
                                
                                detections.append({
                                    "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                                    "confidence": confidence,
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "estimated_distance": estimated_distance,
                                    "pixel_distance": pixel_distance,
                                    "box_width": box_width,
                                    "box_height": box_height
                                })
            
            return detections
            
        except Exception as e:
            print(f"Auto YOLO error: {e}")
            return []
    
    def _apply_deepsort(self, frame, detections):
        """Apply DeepSORT tracking to detections"""
        if not detections or not self.tracker:
            return detections
        
        try:
            # Convert detections to DeepSORT format
            deepsort_detections = []
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # [x, y, w, h, confidence, class_id]
                deepsort_detections.append([
                    [x1, y1, width, height],
                    det["confidence"],
                    det["class_name"]
                ])
            
            # Update tracks
            tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
            
            # Convert tracks back to detection format
            tracked_detections = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
                track_id = track.track_id
                
                # Get class name
                if hasattr(track, 'get_det_class'):
                    class_name = track.get_det_class()
                else:
                    # Find original detection with similar bbox
                    class_name = "unknown"
                    for det in detections:
                        det_bbox = det["bbox"]
                        # Check if bboxes are similar
                        if (abs(det_bbox[0] - ltrb[0]) < 20 and 
                            abs(det_bbox[1] - ltrb[1]) < 20):
                            class_name = det["class_name"]
                            break
                
                # Calculate properties for tracked object
                box_width = ltrb[2] - ltrb[0]
                box_height = ltrb[3] - ltrb[1]
                
                tracked_detections.append({
                    "bbox": ltrb,
                    "confidence": track.det_conf if hasattr(track, 'det_conf') else 0.5,
                    "class_id": 0,  # Placeholder
                    "class_name": class_name,
                    "track_id": track_id,
                    "estimated_distance": self._estimate_distance(box_width, class_name),
                    "box_width": box_width,
                    "box_height": box_height
                })
            
            return tracked_detections
            
        except Exception as e:
            print(f"DeepSORT error: {e}")
            return detections  # Return original detections if tracking fails
    
    def draw(self, frame):
        """Draw auto mode specific elements"""
        if not self.controller.object_detection_enabled:
            return
        
        # Get detections safely
        with self.detections_lock:
            detections_to_draw = self.detections.copy()
        
        # Draw navigation elements
        self._draw_state_based_navigation(frame, detections_to_draw)
    
    def _draw_state_based_navigation(self, frame, detections):
        """Draw navigation information based on current state"""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # Draw distance threshold circle (YOLO distance check)
        cv2.circle(frame, (center_x, center_y), 
                  self.yolo_distance_threshold, 
                  (0, 255, 0), 2)  # Green circle for distance threshold
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Draw obstacle detections with distance info
        for det in detections:
            try:
                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
                class_name = det.get("class_name", "")
                
                if class_name not in self.obstacle_classes:
                    continue
                
                # Calculate object center
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                
                # Calculate distance from center
                pixel_distance = math.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
                
                # Get estimated distance in meters
                estimated_distance = det.get("estimated_distance", float('inf'))
                
                # Set color based on proximity
                if pixel_distance < self.yolo_distance_threshold:
                    color = (0, 0, 255)  # Red for too close
                    thickness = 3
                elif pixel_distance < self.yolo_distance_threshold * 2:
                    color = (0, 255, 255)  # Yellow for warning
                    thickness = 2
                else:
                    color = self._get_color(class_name)
                    thickness = 1
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
                # Draw distance label
                if estimated_distance < float('inf'):
                    distance_text = f"{class_name}: {estimated_distance:.1f}m"
                else:
                    distance_text = f"{class_name}: {pixel_distance:.0f}px"
                
                cv2.putText(frame, distance_text, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw line to center if too close
                if pixel_distance < self.yolo_distance_threshold:
                    cv2.line(frame, (int(obj_center_x), int(obj_center_y)), 
                            (center_x, center_y), color, 2)
                    cv2.putText(frame, "TOO CLOSE!", 
                               (int(x1), int(y1) - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            except:
                continue
        
        # Draw state information
        self._draw_state_info(frame)
    
    def _draw_state_info(self, frame):
        """Draw current state information"""
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # State colors
        state_colors = {
            "MOVING_FORWARD": (0, 255, 0),    # Green
            "AVOIDING_RIGHT": (0, 255, 255),  # Yellow
            "TURNING_LEFT": (255, 0, 0),      # Blue for turn
            "RESUME": (255, 255, 0)           # Cyan
        }
        
        # Get current state color
        state_color = state_colors.get(self.state, (255, 255, 255))
        
        # Draw large state indicator
        state_box_y = 50
        
        # Background for state box
        cv2.rectangle(frame, (w // 2 - 200, state_box_y), 
                     (w // 2 + 200, state_box_y + 60), (40, 40, 40), -1)
        cv2.rectangle(frame, (w // 2 - 200, state_box_y), 
                     (w // 2 + 200, state_box_y + 60), state_color, 3)
        
        # State text
        state_display = self.state.replace("_", " ")
        cv2.putText(frame, f"STATE: {state_display}", 
                   (w // 2 - 180, state_box_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
        
        # Draw avoidance reason if applicable
        if self.avoidance_reason:
            reason_text = f"Reason: {self.avoidance_reason}"
            cv2.putText(frame, reason_text, (w // 2 - 180, state_box_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        # Draw action based on state
        if self.state == "AVOIDING_RIGHT":
            elapsed = current_time - self.state_start_time
            progress = min(elapsed / self.sideways_duration, 1.0)
            
            # Draw right arrow
            arrow_text = "→ AVOIDING RIGHT →"
            cv2.putText(frame, arrow_text, (w // 2 - 140, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            # Progress bar
            bar_width = 300
            bar_height = 20
            bar_x = w // 2 - bar_width // 2
            bar_y = 180
            
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + bar_height), 
                         (0, 255, 255), -1)
            
            cv2.putText(frame, f"Avoiding right: {progress*100:.0f}%", 
                       (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show consecutive right count
            counts_text = f"Consecutive Right: {self.consecutive_right_count}/{self.consecutive_threshold}"
            cv2.putText(frame, counts_text, (w // 2 - 180, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        elif self.state == "TURNING_LEFT":
            elapsed = current_time - self.state_start_time
            progress = min(elapsed / self.turn_duration, 1.0)
            
            # Draw left turn arrow
            arrow_text = "↶ TURNING LEFT ↶"
            
            # Blinking arrow
            blink = int(current_time * 5) % 2
            if blink:
                cv2.putText(frame, arrow_text, (w // 2 - 140, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            
            # Progress bar
            bar_width = 300
            bar_height = 20
            bar_x = w // 2 - bar_width // 2
            bar_y = 180
            
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + bar_height), 
                         (255, 0, 0), -1)
            
            cv2.putText(frame, f"Turning left: {progress*100:.0f}%", 
                       (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show turn info
            turn_info = f"Turn Speed: {self.turn_speed}, Duration: {self.turn_duration}s"
            cv2.putText(frame, turn_info, (w // 2 - 180, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        elif self.state == "RESUME":
            # Countdown to resume
            elapsed = current_time - self.state_start_time
            time_left = max(0, 0.5 - elapsed)
            
            resume_text = f"Resuming in: {time_left:.1f}s"
            cv2.putText(frame, resume_text, (w // 2 - 100, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Mode info
        mode_text = "AUTO: Right-side Avoidance + Turn on 2 Consecutive Rights"
        cv2.putText(frame, mode_text, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection FPS
        cv2.putText(frame, f"FPS: {self.detection_fps}", 
                   (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Speed info
        speed_text = f"Speed: {self.controller.speed:.1f}x"
        cv2.putText(frame, speed_text, (20, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # IR sensor info
        if self.controller.inf_distance is not None:
            ir_color = (0, 255, 0) if self.controller.inf_distance >= self.safe_distance_threshold else (0, 0, 255)
            ir_text = f"IR: {self.controller.inf_distance:.2f}m"
            cv2.putText(frame, ir_text, (20, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ir_color, 1)
        
        # Distance threshold info
        distance_text = f"YOLO Threshold: {self.yolo_distance_threshold}px"
        cv2.putText(frame, distance_text, (20, h - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Wall detection info
        wall_detect_text = f"Right Count: {self.consecutive_right_count}/2"
        wall_color = (0, 255, 0)
        if self.consecutive_right_count >= 2:
            wall_color = (0, 0, 255)
        
        cv2.putText(frame, wall_detect_text, (20, h - 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, wall_color, 1)
    
    def _get_color(self, class_name):
        """Get color for a class"""
        color_map = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'truck': (0, 165, 255),     # Orange
            'bus': (0, 165, 255),       # Orange
            'bicycle': (255, 255, 0),   # Cyan
            'motorcycle': (255, 255, 0),# Cyan
            'chair': (255, 0, 255),     # Purple
            'bench': (255, 0, 255),     # Purple
            'couch': (255, 0, 255),     # Purple
        }
        
        return color_map.get(class_name, (255, 255, 255))  # White for unknown