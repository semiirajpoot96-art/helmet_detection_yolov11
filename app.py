import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖"
)

st.title("Helmet Detection System")
st.subheader("AI-Based Helmet Detection Using YOLOv11")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("YOLOv11 Model Loaded Successfully!")

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    confidence = st.slider(
        "Confidence Threshold",
        0.10, 1.00, 0.25, 0.05
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        if st.button("Detect Helmet"):
            with st.spinner("Detecting..."):
                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640
                )

                result = results[0]
                plotted = result.plot()[:, :, ::-1]

                st.subheader("Detection Result")
                st.image(plotted)

                st.subheader("Detection Summary")

                counts = Counter(
                    model.names[int(box.cls[0])]
                    for box in result.boxes
                )

                if counts:
                    for name, count in counts.items():
                        st.write(f"{name}: {count}")
                else:
                    st.write("No objects detected.")

except Exception as e:
    st.error(f"Error: {e}")
