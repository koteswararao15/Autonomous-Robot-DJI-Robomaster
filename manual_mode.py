import cv2
import numpy as np
import threading
import queue
import torch
import time

class ManualMode:
    def __init__(self, controller):
        self.controller = controller
        self.detections = []
        self.detections_lock = threading.Lock()
        
        # Separate thread for YOLO processing
        self.yolo_queue = queue.Queue(maxsize=2)
        self.yolo_thread = threading.Thread(
            target=self._yolo_worker,
            daemon=True
        )
        self.yolo_thread.start()
        
        # FPS tracking
        self.last_detection_time = 0
        self.detection_fps = 0
        self.frame_count = 0
        
    def update(self):
        """Manual control update - returns x, y, z speeds"""
        x, y, z = 0, 0, 0

        if self.controller.movement['forward']:
            x += self.controller.speed
        if self.controller.movement['backward']:
            x -= self.controller.speed
        if self.controller.movement['left']:
            y += self.controller.speed
        if self.controller.movement['right']:
            y -= self.controller.speed
        if self.controller.movement['rotate_left']:
            z += self.controller.rotation_speed
        if self.controller.movement['rotate_right']:
            z -= self.controller.rotation_speed
        
        # Add frame to YOLO queue if detection is enabled
        if (self.controller.object_detection_enabled and 
            hasattr(self.controller, 'latest_frame') and 
            self.controller.latest_frame is not None):
            
            try:
                if not self.yolo_queue.full():
                    self.yolo_queue.put_nowait(self.controller.latest_frame.copy())
            except:
                pass
        
        return x, y, z
    
    def _yolo_worker(self):
        """YOLO worker thread - runs detection separately from main thread"""
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
                
                # Run fast YOLO detection
                detections = self._run_fast_yolo_detection(frame)
                
                # Update detections safely
                with self.detections_lock:
                    self.detections = detections
                
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_detection_time >= 1.0:
                    self.detection_fps = self.frame_count
                    self.frame_count = 0
                    self.last_detection_time = current_time
                
            except Exception as e:
                print(f"Manual YOLO worker error: {e}")
                time.sleep(0.01)
    
    def _run_fast_yolo_detection(self, frame):
        """Optimized YOLO detection for speed"""
        try:
            height, width = frame.shape[:2]
            
            # Fast resize - smaller image = faster inference
            target_size = 320  # Small but still accurate
            ratio = target_size / max(height, width)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            inference_frame = cv2.resize(frame, (new_width, new_height))
            
            # Use torch.no_grad() for speed
            with torch.no_grad():
                # Faster YOLO settings
                results = self.controller.model(
                    inference_frame,
                    imgsz=target_size,
                    conf=0.3,  # Higher confidence = fewer detections = faster
                    iou=0.5,   # Higher IoU = fewer overlapping boxes
                    max_det=15, # Fewer detections = faster
                    verbose=False,
                    augment=False  # No augmentation = faster
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
                        
                        # Filter very small detections
                        if box_width > 15 and box_height > 15:
                            class_id = int(box.cls[0])
                            class_name = self.controller.model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            detections.append({
                                "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                                "confidence": confidence,
                                "class_id": class_id,
                                "class_name": class_name
                            })
            
            return detections
            
        except Exception as e:
            print(f"Fast YOLO error: {e}")
            return []
    
    def draw(self, frame):
        """Draw manual mode specific elements - optimized for speed"""
        if not self.controller.object_detection_enabled:
            return
        
        # Get detections safely
        with self.detections_lock:
            detections_to_draw = self.detections.copy()
        
        # Draw bounding boxes if enabled
        if detections_to_draw and self.controller.show_bounding_boxes:
            self._draw_bounding_boxes_fast(frame, detections_to_draw)
        
        # Draw lines if enabled
        if detections_to_draw and self.controller.show_lines:
            self._draw_distance_lines_fast(frame, detections_to_draw)
        
        # Show detection FPS
        #if self.controller.object_detection_enabled:
        #    cv2.putText(frame, f"YOLO FPS: {self.detection_fps}", 
        #               (frame.shape[1] - 180, frame.shape[0] - 40),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _draw_bounding_boxes_fast(self, frame, detections):
        """Optimized bounding box drawing"""
        h, w = frame.shape[:2]
        
        for det in detections:
            try:
                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
                conf = det.get("confidence", 0)
                class_name = det.get("class_name", "unknown")
                class_id = det.get("class_id", 0)
            except:
                continue
            
            # Fast ignore zone check
            if self.controller.ignore_zone_enabled:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                cx = w // 2
                ix1 = cx - self.controller.ignore_zone_width / 2
                ix2 = cx + self.controller.ignore_zone_width / 2
                iy1 = h - self.controller.ignore_zone_height
                iy2 = h
                
                if ix1 <= center_x <= ix2 and iy1 <= center_y <= iy2:
                    continue
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = self._get_color(class_id)
            
            # Fast rectangle drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Fast label drawing (simple)
            label = f"{class_name}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_distance_lines_fast(self, frame, detections):
        """Optimized distance lines drawing"""
        h, w = frame.shape[:2]
        robot_pos = (w // 2, h)
        
        for det in detections:
            try:
                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
            except:
                continue
            
            # Fast ignore zone check
            if self.controller.ignore_zone_enabled:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                cx = w // 2
                ix1 = cx - self.controller.ignore_zone_width / 2
                ix2 = cx + self.controller.ignore_zone_width / 2
                iy1 = h - self.controller.ignore_zone_height
                iy2 = h
                
                if ix1 <= center_x <= ix2 and iy1 <= center_y <= iy2:
                    continue
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Fast line drawing
            obj_bottom = ((x1 + x2) // 2, y2)
            cv2.line(frame, robot_pos, obj_bottom, (0, 255, 255), 1)
            
            # Fast distance calculation and text
            distance = int(90 - 80 * (obj_bottom[1] / h))
            distance = max(10, min(distance, 90))
            
            mid_x = (robot_pos[0] + obj_bottom[0]) // 2
            mid_y = (robot_pos[1] + obj_bottom[1]) // 2
            cv2.putText(frame, f"{distance}", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 1)
    
    def _get_color(self, cls_id):
        """Fast color lookup"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
        ]
        return colors[cls_id % len(colors)]