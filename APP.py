from PIL import Image
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_DIR = 'https://github.com/hr828061/fabric_detection/blob/main/best.pt'

def main():
    # Load the YOLO model
    model = YOLO(MODEL_DIR)

    st.sidebar.header("Fabric Defect Detection using YOLOv8\nMembers:\nk21-3010\n\n        DevOps Project")
    st.title("Real-time Fabric Defect Detection")
    st.write("""
    This app allows you to upload a fabric image or use your webcam for real-time fabric defect detection using the YOLOv8 model.
    """)

    # Sidebar Options
    option = st.sidebar.radio("Choose Input Method", ("Browse Image", "Real-time Video"))

    if option == "Browse Image":
        uploaded_file = st.file_uploader("Upload a Fabric Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                inference_images(uploaded_file, model)

    elif option == "Real-time Video":
        st.warning("Ensure your webcam is enabled!")
        real_time_video(model)

def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
    # Perform inference on the uploaded image
    predict = model.predict(image)
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    st.image(plotted, caption="Detected Image", width=600)

def real_time_video(model):
    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust for external cameras

    stframe = st.empty()  # Create a Streamlit container to hold video frames
    st.button("Stop", key="stop_button")  # Placeholder for stopping (not implemented fully)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break

        # Perform inference on the current frame
        results = model.predict(frame)
        plotted_frame = results[0].plot()

        # Display the result in real-time
        stframe.image(plotted_frame, channels="BGR", use_column_width=True)

    cap.release()

if __name__ == '__main__':
    main()
