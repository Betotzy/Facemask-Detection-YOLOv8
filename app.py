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
st.caption("Real-time mask detection with webcam, image, or video")

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
    value=0.4,
    step=0.05
)

selected_classes = st.sidebar.multiselect(
    "Filter Class",
    options=list(class_names.values()),
    default=list(class_names.values())
)

show_fps = st.sidebar.checkbox("Show FPS", True)
save_video = st.sidebar.checkbox("Save Output Video", False)

os.makedirs("output", exist_ok=True)

# ===============================
# HELPER FUNCTION
# ===============================
def process_frame(frame):
    start = time.time()

    results = model(frame, conf=conf_thres)
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]

            if label not in selected_classes:
                box.conf[0] = 0  # hide

    annotated = results[0].plot()

    fps = 1 / (time.time() - start)
    if show_fps:
        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
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
    ["Webcam", "Image", "Video"]
)

frame_placeholder = st.empty()

# ===============================
# WEBCAM MODE
# ===============================
if source == "Webcam":
    run = st.checkbox("▶ Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        writer = None

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/webcam_output.mp4",
                fourcc,
                20,
                (int(cap.get(3)), int(cap.get(4)))
            )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = process_frame(frame)

            if writer:
                writer.write(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        if writer:
            writer.release()

# ===============================
# IMAGE MODE
# ===============================
elif source == "Image":
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file:
        image = np.array(
            cv2.imdecode(
                np.frombuffer(image_file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
        )

        result_img = process_frame(image)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detection Result", use_column_width=True)

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
        writer = None

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                "output/video_output.mp4",
                fourcc,
                int(cap.get(cv2.CAP_PROP_FPS)),
                (int(cap.get(3)), int(cap.get(4)))
            )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)

            if writer:
                writer.write(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        if writer:
            writer.release()

        st.success("✅ Video processing completed")
