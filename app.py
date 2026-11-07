import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import os

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🏍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

st.title("Helmet Detection System")
st.subheader("AI-Based Helmet Detection Using YOLOv11")
st.write("Upload an image to detect helmet-related classes.")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Helmet"):
        with st.spinner("Detecting objects..."):
            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640
            )

            result = results[0]
            plotted = result.plot()[:, :, ::-1]

            st.subheader("Detection Result")
            st.image(plotted, use_container_width=True)

            counts = Counter(
                [model.names[int(box.cls[0])] for box in result.boxes]
            )

            st.subheader("Detection Summary")

            if counts:
                for name, count in counts.items():
                    st.write(f"{name}: {count}")
            else:
                st.info("No objects detected.")
