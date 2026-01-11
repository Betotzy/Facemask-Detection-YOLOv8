import streamlit as st
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Face Mask Detection YOLOv8",
    layout="wide"
)

st.title("😷 Face Mask Detection (YOLOv8)")
st.caption("Real-time mask detection with image or video")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
class_names = model.names

# ===============================
# SIDEBAR SETTINGS
# ===============================
st.sidebar.header("⚙️ Settings")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

selected_classes = st.sidebar.multiselect(
    "Filter Class",
    options=list(class_names.values()),
    default=list(class_names.values())
)

show_fps = st.sidebar.checkbox("Show FPS", True)
save_video = st.sidebar.checkbox("Save Output Video", False)
skip_frames = st.sidebar.slider("Skip Frames", 1, 5, 2)
inference_size = st.sidebar.selectbox("Inference Size", [320, 416, 640], index=1)

os.makedirs("output", exist_ok=True)

# ===============================
# HELPER FUNCTION
# ===============================
def process_frame(frame, resize=True):
    start = time.time()
    
    h, w = frame.shape[:2]
    
    if resize:
        frame_resized = cv2.resize(frame, (inference_size, inference_size))
    else:
        frame_resized = frame
    
    results = model(frame_resized, conf=conf_thres, verbose=False)
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            if label not in selected_classes:
                box.conf[0] = 0

    annotated = results[0].plot()
    
    if resize:
        annotated = cv2.resize(annotated, (w, h))

    fps = 1 / (time.time() - start)
    if show_fps:
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    return annotated


# ===============================
# INPUT SOURCE
# ===============================
source = st.selectbox(
    "Select Input Source",
    ["Image", "Video"]
)

frame_placeholder = st.empty()
progress_bar = st.empty()
status_text = st.empty()

# ===============================
# IMAGE MODE
# ===============================
if source == "Image":
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file:
        image = np.array(
            cv2.imdecode(
                np.frombuffer(image_file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
        )

        with status_text.container():
            st.info("🔄 Processing image...")

        result_img = process_frame(image, resize=True)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detection Result", use_column_width=True)
        st.success("✅ Done!")

# ===============================
# VIDEO MODE
# ===============================
else:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        with status_text.container():
            st.info(f"📹 {total_frames} frames @ {fps}fps | Processing every {skip_frames} frame(s)")
        
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/video_output.mp4",
                fourcc,
                fps,
                (width, height)
            )

        frame_count = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames != 0:
                if writer:
                    writer.write(frame)
                frame_count += 1
                continue

            frame = process_frame(frame, resize=True)
            processed_count += 1

            if writer:
                writer.write(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            progress = frame_count / total_frames
            progress_bar.progress(min(progress, 1.0))
            
            frame_count += 1

        cap.release()
        if writer:
            writer.release()

        progress_bar.progress(1.0)
        st.success(f"✅ Done! ({processed_count} frames processed)")
        
        if save_video and os.path.exists("output/video_output.mp4"):
            with open("output/video_output.mp4", "rb") as f:
                st.download_button(
                    "⬇️ Download",
                    f.read(),
                    "video_output.mp4",
                    "video/mp4"
                )
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
