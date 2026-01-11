import streamlit as st
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import shutil

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Face Mask Detection & Trainer YOLOv8",
    layout="wide"
)

st.title("😷 Face Mask Detection & Training (YOLOv8)")
st.caption("Deteksi masker real-time atau latih ulang model dengan dataset baru")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model(model_path="best.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Model load error: {str(e)}")
        return None

model = load_model()

# ===============================
# SIDEBAR SETTINGS
# ===============================
st.sidebar.header("⚙️ Settings")

# Menu Navigasi Utama
mode = st.sidebar.selectbox("Pilih Mode", ["Inference (Deteksi)", "Training (Latih Model)"])

if mode == "Inference (Deteksi)":
    conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    iou_thres = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    if model:
        class_names = model.names
        selected_classes = st.sidebar.multiselect(
            "Filter Class",
            options=list(class_names.values()),
            default=list(class_names.values())
        )
    
    show_fps = st.sidebar.checkbox("Show FPS", True)
    save_video = st.sidebar.checkbox("Save Output Video", False)
    skip_frames = st.sidebar.slider("Process Every N Frames", 1, 5, 1)
    inference_size = st.sidebar.selectbox("Inference Size", [320, 416, 640], index=2)

os.makedirs("output", exist_ok=True)

# ===============================
# HELPER FUNCTION (PROSES FRAME)
# ===============================
def process_frame(frame, resize=True):
    start = time.time()
    h, w = frame.shape[:2]
    
    if resize and inference_size < 640:
        frame_resized = cv2.resize(frame, (inference_size, inference_size))
    else:
        frame_resized = frame
    
    results = model(frame_resized, conf=conf_thres, iou=iou_thres, verbose=False)
    boxes = results[0].boxes

    # Filter class
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] not in selected_classes:
                box.conf[0] = 0 

    annotated = results[0].plot()
    
    if resize and inference_size < 640:
        annotated = cv2.resize(annotated, (w, h))

    fps = 1 / (time.time() - start)
    if show_fps:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return annotated

# ===============================
# LOGIC: INFERENCE MODE
# ===============================
if mode == "Inference (Deteksi)":
    if model is None:
        st.warning("Silakan pastikan file 'best.pt' ada atau latih model terlebih dahulu.")
        st.stop()

    source = st.selectbox("Select Input Source", ["Image", "Video"])
    
    if source == "Image":
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if image_file:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            result_img = process_frame(img, resize=False)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    else: # Video Mode
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if video_file:
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            
            cap = cv2.VideoCapture(temp_path)
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_frame = process_frame(frame)
                frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            cap.release()
            os.remove(temp_path)

# ===============================
# LOGIC: TRAINING MODE
# ===============================
else:
    st.header("🏋️ Train New Model")
    st.info("Pastikan Anda memiliki dataset dalam format YOLO (folder images & labels) dan file data.yaml.")

    col1, col2 = st.columns(2)
    with col1:
        yaml_path = st.text_input("Path ke file data.yaml", value="dataset/data.yaml")
        epochs = st.number_input("Jumlah Epochs", min_value=1, value=10)
        imgsz = st.selectbox("Image Size Training", [320, 416, 640], index=2)
    
    with col2:
        batch_size = st.number_input("Batch Size", min_value=-1, value=16, help="-1 untuk Auto-batch")
        model_variant = st.selectbox("Base Model", ["yolov8n.pt", "yolov8s.pt", "best.pt"])

    if st.button("🚀 Start Training"):
        if not os.path.exists(yaml_path):
            st.error(f"File {yaml_path} tidak ditemukan!")
        else:
            try:
                # Load base model untuk training
                train_model = YOLO(model_variant)
                
                st.write("---")
                with st.status("🏗️ Training sedang berjalan...", expanded=True) as status:
                    st.write("Inisialisasi dataset...")
                    # Menjalankan training
                    results = train_model.train(
                        data=yaml_path,
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch_size,
                        project="runs/detect",
                        name="mask_train",
                        exist_ok=True
                    )
                    status.update(label="✅ Training Selesai!", state="complete", expanded=False)
                
                st.success("Model berhasil dilatih!")
                
                # Path ke model terbaik hasil training
                best_model_path = "runs/detect/mask_train/weights/best.pt"
                
                if os.path.exists(best_model_path):
                    with open(best_model_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Trained Model (best.pt)",
                            data=f,
                            file_name="trained_best.pt",
                            mime="application/octet-stream"
                        )
                    
                    if st.button("🔄 Gunakan Model Baru"):
                        shutil.copy(best_model_path, "best.pt")
                        st.rerun()
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat training: {e}")
