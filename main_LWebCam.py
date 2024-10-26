import cv2
import numpy as np
import tensorflow as tf  # Or import torch if using a PyTorch model

# Load your pre-trained pose estimation model
model_path = 'posenet_mobilenet.tflite'  # Replace with your model's path
model = tf.lite.Interpreter(model_path=model_path)
model.allocate_tensors()

# After loading the model and allocating tensors
input_details = model.get_input_details()
model_input_height, model_input_width = input_details[0]['shape'][1:3]

def process_frame(frame):
    global model_input_width, model_input_height
    input_frame = cv2.resize(frame, (model_input_width, model_input_height))
    input_frame = input_frame.astype(np.float32)  # Convert to float32
    input_frame = input_frame / 255.0  # Normalize if required
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
    return input_frame

# Load the golden video
golden_video_path = 'public_/GoldenVideo.mp4'  # Use forward slash
golden_cap = cv2.VideoCapture(golden_video_path)

# Set up webcam capture
cam = cv2.VideoCapture(0)  # 0 is usually the default camera

frame_count = 0

while cam.isOpened() and golden_cap.isOpened():
    ret_cam, frame_cam = cam.read()
    ret_golden, frame_golden = golden_cap.read()
    
    if not ret_cam or not ret_golden:
        print("Error reading from camera or golden video.")
        break

    # Resize golden frame to match the height of the camera frame
    h_cam, w_cam = frame_cam.shape[:2]
    h_golden, w_golden = frame_golden.shape[:2]
    aspect_ratio = w_golden / h_golden
    new_w_golden = int(h_cam * aspect_ratio)
    frame_golden_resized = cv2.resize(frame_golden, (new_w_golden, h_cam))

    # Process frames to fit model requirements
    cam_input = process_frame(frame_cam)
    golden_input = process_frame(frame_golden_resized)

    # Run pose estimation model on both frames
    model.set_tensor(input_details[0]['index'], cam_input)
    model.invoke()
    cam_pose = model.get_tensor(model.get_output_details()[0]['index'])

    model.set_tensor(input_details[0]['index'], golden_input)
    model.invoke()
    golden_pose = model.get_tensor(model.get_output_details()[0]['index'])

    # Compare the detected poses
    difference = np.linalg.norm(cam_pose - golden_pose)
    print(f"Pose Difference: {difference}")

    # Stack the frames horizontally
    combined_frame = np.hstack((frame_cam, frame_golden_resized))

    # Save the combined frame
    cv2.imwrite(f'output_frame_{frame_count}.jpg', combined_frame)
    frame_count += 1

    # Break the loop after processing a few frames (e.g., 10 frames)
    if frame_count >= 10:
        break

# Release resources
cam.release()
golden_cap.release()
