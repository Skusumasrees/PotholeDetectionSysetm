import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 segmentation model
MODEL_PATH = "best_50.pt"  # Update with your trained model path
model = YOLO(MODEL_PATH)

# Custom CSS & Animations for Enhanced UI/UX
def set_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Montserrat:wght@700&display=swap');
            
            body {background: #0F172A; color: #F8FAFC; font-family: 'Inter', sans-serif;}
            .main {background: linear-gradient(to right, #1E293B, #334155); color: white; padding: 20px; border-radius: 10px;}
            h1 {color: #E2E8F0; text-align: center; font-weight: 700; font-family: 'Montserrat', sans-serif; font-size: 2.5em;}
            h2, h3 {color: #E2E8F0; text-align: center; font-weight: 600; font-family: 'Inter', sans-serif;}
            p {text-align: center; font-size: 1.1em; color: #CBD5E1;}
            .stButton>button, .stDownloadButton>button {
                background: linear-gradient(to right, #2563EB, #1E40AF);
                color: white;
                font-size: 16px;
                padding: 12px 24px;
                border-radius: 8px;
                transition: all 0.3s ease-in-out;
                border: none;
                cursor: pointer;
                font-weight: bold;
            }
            .stButton>button:hover, .stDownloadButton>button:hover {
                background: linear-gradient(to right, #1E40AF, #1E3A8A);
            }
            .stSidebar {background: #1E293B; color: white; padding: 20px; border-radius: 10px;}
            
            /* Updated background for Image & Video Segmentation sections */
            .segmentation-section {
                background: #2D3748;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply Custom CSS
set_custom_css()

# Streamlit UI
st.title("Pothole Detection System")
st.markdown("<p style='text-align: center; font-size: 1.2em;'>Detects potholes in images and videos with AI-powered segmentation.</p>", unsafe_allow_html=True)

# Sidebar for options
task = st.sidebar.radio("Choose Task", ["Image Segmentation", "Video Segmentation"], index=0)

if task == "Image Segmentation":
    st.markdown("<div class='segmentation-section'>", unsafe_allow_html=True)
    st.subheader("Upload an Image for Pothole Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", channels="BGR", use_container_width=True)
        
        button_text = st.empty()
        process_button = button_text.button("Start Pothole Detection")
        
        if process_button:
            button_text.empty()
            button_text.button("Processing Image...", disabled=True)
            
            results = model(image)
            for result in results:
                segmented_image = result.plot()
            
            button_text.empty()
            button_text.button("Process Completed!", disabled=True)
            
            st.image(segmented_image, caption="Pothole Detection Output", channels="BGR", use_container_width=True)
            output_image_path = "segmented_pothole.jpg"
            cv2.imwrite(output_image_path, segmented_image)

            with open(output_image_path, "rb") as f:
                st.download_button("Download Processed Image", f, file_name="pothole_output.jpg")
    st.markdown("</div>", unsafe_allow_html=True)

elif task == "Video Segmentation":
    st.markdown("<div class='segmentation-section'>", unsafe_allow_html=True)
    st.subheader("Upload a Video for Pothole Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        st.video(temp_video.name)
        
        button_text = st.empty()
        process_button = button_text.button("Start Pothole Detection")
        
        if process_button:
            button_text.empty()
            button_text.button("Processing Video...", disabled=True)
            
            cap = cv2.VideoCapture(temp_video.name)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out_video_path = "output_pothole.mp4"
            out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

            stframe = st.empty()
                
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                for result in results:
                    segmented_frame = result.plot()
                
                out.write(segmented_frame)
                stframe.image(segmented_frame, channels="BGR", use_container_width=True)
                time.sleep(1 / fps)

            cap.release()
            out.release()
            
            button_text.empty()
            button_text.button("Process Completed!", disabled=True)
            
            st.success("Processing Complete! Download the Processed Video Below")
            with open(out_video_path, "rb") as f:
                st.download_button("Download Processed Video", f, file_name="pothole_output.mp4")
    st.markdown("</div>", unsafe_allow_html=True)
