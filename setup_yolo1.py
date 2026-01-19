import os
from ultralytics import YOLO

def download_yolo_model():
    """Download YOLO model for offline use"""
    print("Setting up YOLO for offline operation...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Model file path
    model_path = 'models/yolov8n.pt'
    
    if os.path.exists(model_path):
        print("✓ YOLO model already exists")
        return True
    
    print("Downloading YOLOv8n model...")
    try:
        # This will automatically download and cache the model
        model = YOLO('yolov8n.pt')
        
        # Test the model to ensure it works
        print("Testing model...")
        results = model('https://ultralytics.com/images/bus.jpg')
        print("✓ Model test completed")
        
        # The model is now cached locally in ultralytics cache
        # We don't need to manually save it as it will be loaded from cache
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download YOLO model: {e}")
        return False

def verify_setup():
    """Verify that everything is set up correctly"""
    print("\nVerifying setup...")
    
    try:
        # Test YOLO import and loading
        from ultralytics import YOLO
        print("✓ YOLO import successful")
        
        # Test loading the model
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loading successful")
        print(f"✓ Model classes: {len(model.names)}")
        
        # Test OpenCV
        import cv2
        print("✓ OpenCV import successful")
        
        # Test other required packages
        import numpy as np
        import keyboard
        print("✓ All package imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Verification error: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("YOLO OFFLINE SETUP")
    print("=" * 50)
    
    # Download model
    if not download_yolo_model():
        print("Failed to download model. Please check internet connection.")
        return
    
    # Verify setup
    if not verify_setup():
        print("\nSetup completed with warnings")
    else:
        print("\n✓ Setup completed successfully!")
    
    print("\nNow you can run the robot offline using:")
    print("python robot_yolo_control.py")

if __name__ == '__main__':
    main()