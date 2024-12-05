pip install os
import os
pip install ultralytics
from ultralytics import YOLO
import streamlit as st

MODEL_DIR = "best.pt"  # Adjust path as necessary

def main():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model file not found at: {MODEL_DIR}")
        return

    try:
        model = YOLO(MODEL_DIR)
        st.sidebar.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Rest of the Streamlit app logic
    st.title("Fabric Defect Detection")
    st.write("App is running!")

if __name__ == "__main__":
    main()
