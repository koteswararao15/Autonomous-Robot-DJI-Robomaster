import math
import cv2


class PersonInfo:
    """Convert normalized bounding box to pixel coordinates"""
    
    def __init__(self, x, y, w, h, image_width=1280, image_height=720):
        self._x = x  # Normalized center x (0-1)
        self._y = y  # Normalized center y (0-1)
        self._w = w  # Normalized width (0-1)
        self._h = h  # Normalized height (0-1)
        self.image_width = image_width
        self.image_height = image_height

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


class TrackMode:
    """TRACK mode – RoboMaster internal person following"""
    
    def __init__(self, controller):
        self.controller = controller
        
        # Target following distance (meters)
        self.target_distance = 1.5
        
        # Detected persons
        self.persons = []
        
        # Subscription state
        self.subscribed = False
        
        # Mode state
        self.active = False
        
        # Camera geometry (calibration)
        self.real_width_of_person = 0.6  # meters (average shoulder width)
        self.real_width_of_calibration_object = 0.07  # meters
        self.distance_to_calibration_object = 0.2  # meters
        self.pixel_width_of_calibration_object = 210  # pixels
        
        # Calculate focal distance
        self.distance_focale = (
            self.pixel_width_of_calibration_object
            * self.distance_to_calibration_object
        ) / self.real_width_of_calibration_object
        
        print("✓ Track Mode initialized (inactive)")

    # ==========================================================
    # MODE LIFECYCLE MANAGEMENT
    # ==========================================================
    
    def set_active(self, active):
        """Activate or deactivate TRACK mode"""
        if active and not self.active:
            self._activate()
        elif not active and self.active:
            self._deactivate()
    
    def _activate(self):
        """Activate TRACK mode"""
        if self.subscribed:
            return
        
        try:
            print("🔄 Activating TRACK mode...")
            self.controller.ep_robot.vision.sub_detect_info(
                name="person",
                callback=self._on_detect_person,
            )
            self.subscribed = True
            self.active = True
            print("✅ TRACK MODE: Person detection subscribed")
            
        except Exception as e:
            print(f"❌ TRACK MODE: Subscription failed: {e}")
            # Fallback for testing
            self._on_detect_person = self._dummy_person_detection
            self.subscribed = True
            self.active = True
            print("⚠️  TRACK MODE: Using dummy detection (for testing)")
    
    def _deactivate(self):
        """Deactivate TRACK mode"""
        if not self.subscribed:
            return
        
        try:
            print("🔄 Deactivating TRACK mode...")
            self.controller.ep_robot.vision.unsub_detect_info(name="person")
            print("✅ TRACK MODE: Person detection unsubscribed")
        except Exception as e:
            print(f"⚠️  TRACK MODE: Unsubscription warning: {e}")
        
        self.persons.clear()
        self.subscribed = False
        self.active = False
    
    def cleanup(self):
        """Clean up resources (called on program exit)"""
        self._deactivate()
        print("✅ TRACK MODE: Cleanup complete")

    # ==========================================================
    # PERSON DETECTION CALLBACKS
    # ==========================================================
    
    def _dummy_person_detection(self, person_info):
        """Fallback detection for testing when real detection is unavailable"""
        # Create a dummy person in the center of the frame
        self.persons = [PersonInfo(0.5, 0.5, 0.2, 0.4)]
    
    def _on_detect_person(self, person_info):
        """Callback for RoboMaster's internal person detection"""
        self.persons.clear()
        
        if not person_info or len(person_info) == 0:
            return
        
        try:
            for i in range(len(person_info)):
                # RoboMaster returns [x, y, w, h] where:
                # x, y are normalized coordinates (0-1) of bounding box center
                # w, h are normalized width and height (0-1)
                x, y, w, h = person_info[i]
                
                # Create PersonInfo object
                person = PersonInfo(x, y, w, h)
                self.persons.append(person)
                
        except Exception as e:
            print(f"⚠️  Person detection parsing error: {e}")
            self.persons = []

    # ==========================================================
    # DISTANCE AND ANGLE CALCULATIONS
    # ==========================================================
    
    def calculate_distance(self, person):
        """Calculate distance to person using camera geometry"""
        try:
            pixel_width_of_person = person.width_pixels
            
            if pixel_width_of_person <= 0:
                return self.target_distance  # Default distance
            
            # Calculate actual distance using similar triangles
            distance_to_person = (
                self.real_width_of_person * self.distance_focale
            ) / pixel_width_of_person
            
            # Clamp to reasonable range (0.3m to 10m)
            return max(0.3, min(distance_to_person, 10.0))
            
        except Exception as e:
            print(f"⚠️  Distance calculation error: {e}")
            return self.target_distance  # Fallback to target distance
    
    def calculate_angle(self, person, distance):
        """Calculate angle to person relative to robot center"""
        try:
            # Get frame for image dimensions
            frame = self.controller.latest_frame
            if frame is None:
                return 0.0
            
            # Get person's horizontal center position
            person_center_x, _ = person.center
            
            # Calculate horizontal offset from image center
            image_center_x = frame.shape[1] // 2
            relative_position_pix = person_center_x - image_center_x
            
            # Avoid division by zero
            if relative_position_pix == 0:
                return 0.0
            
            # Calculate real-world horizontal offset
            pixel_to_meter_ratio = self.real_width_of_person / person.width_pixels
            relative_position_meter = pixel_to_meter_ratio * relative_position_pix
            
            # Calculate angle (atan2 gives angle in radians)
            angle_radians = math.atan2(relative_position_meter, distance)
            
            # Convert to degrees
            angle_degrees = math.degrees(angle_radians)
            
            return angle_degrees
            
        except Exception as e:
            print(f"⚠️  Angle calculation error: {e}")
            return 0.0

    # ==========================================================
    # MOVEMENT CONTROL
    # ==========================================================
    
    def update(self):
        """Return (x_speed, y_speed, z_speed) for chassis movement"""
        
        # If TRACK mode is not active, do nothing
        if not self.active:
            return 0, 0, 0
        
        # Safety check: IR sensor indicates too close
        if (
            self.controller.inf_distance is not None
            and self.controller.inf_distance < 0.3
        ):
            print(f"🚨 IR Alert: Too close ({self.controller.inf_distance:.2f} m)")
            return -0.2, 0, 0  # Move backward
        
        # No person detected: search behavior
        if not self.persons:
            return 0.1, 0, 15  # Slow forward + gentle turn
        
        # Track the first detected person
        person = self.persons[0]
        
        # Calculate distance and angle
        distance = self.calculate_distance(person)
        angle = self.calculate_angle(person, distance)
        
        # Calculate distance error from target
        distance_error = distance - self.target_distance
        
        # Calculate movement speeds
        # Forward/backward speed based on distance error
        x_speed = 1.4 * distance_error
        x_speed = max(-0.5, min(x_speed, 0.5))  # Limit speed
        
        # Rotation speed based on angle
        z_speed = 2.0 * angle
        z_speed = max(-90, min(z_speed, 90))  # Limit rotation
        
        # Sideways movement for better tracking
        y_speed = 0
        if abs(angle) > 5:  # If person is significantly off-center
            y_speed = -0.1 if angle > 0 else 0.1
        
        return x_speed, y_speed, z_speed

    # ==========================================================
    # VISUALIZATION
    # ==========================================================
    
    def draw(self, frame):
        """Draw TRACK mode specific elements on the frame"""
        
        # Don't draw if TRACK mode is not active
        if not self.active:
            return
        
        # Draw all detected persons
        for i, person in enumerate(self.persons):
            # Color for detected persons
            color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(frame, person.pt1, person.pt2, color, 2)
            
            # Draw center point
            center_x, center_y = person.center
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Calculate and display distance and angle
            distance = self.calculate_distance(person)
            angle = self.calculate_angle(person, distance)
            
            # Create info text
            info_text = f"Person {i+1}: {distance:.1f}m, {angle:.0f}°"
            
            # Draw text above bounding box
            cv2.putText(
                frame,
                info_text,
                (person.pt1[0], person.pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            
            # Draw line from person center to image center
            h, w = frame.shape[:2]
            cv2.line(
                frame,
                (center_x, center_y),
                (w // 2, h // 2),
                (0, 255, 255),  # Yellow line
                2,
            )
        
        # Draw mode status
        status_text = "STATUS: "
        status_color = (0, 255, 0)
        
        if self.persons:
            status_text += "TRACKING PERSON"
        else:
            status_text += "SEARCHING"
            status_color = (0, 165, 255)  # Orange when searching
        
        cv2.putText(
            frame,
            status_text,
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        
        # Draw target distance
        cv2.putText(
            frame,
            f"Target Distance: {self.target_distance}m",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # Yellow
            2,
        )