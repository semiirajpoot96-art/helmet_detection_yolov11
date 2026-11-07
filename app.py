from flask import Flask, request, render_template_string
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from io import BytesIO
import base64
import os

app = Flask(_name_)

model_path = "best.pt"
model = YOLO(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]
        confidence = float(request.form["confidence"])
        image = Image.open(file.stream).convert("RGB")

        predictions = model.predict(
            source=image, conf=confidence, imgsz=640
        )
        result = predictions[0]
        plotted = result.plot()[:, :, ::-1]

        output = BytesIO()
        Image.fromarray(plotted).save(output, format="JPEG")
        encoded_image = base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

        counts = Counter(
            [model.names[int(box.cls[0])] for box in result.boxes]
        )
        summary = "<br>".join(
            [f"{name}: {count}" for name, count in counts.items()]
        ) or "No objects detected."

        return render_template_string("""
        <h1>Helmet Detection System</h1>
        <h2>Detection Result</h2>
        <img src="data:image/jpeg;base64,{{ image }}"
             style="max-width:100%;">
        <h3>Detection Summary</h3>
        <p>{{ summary|safe }}</p>
        <a href="/">Upload Another Image</a>
        """, image=encoded_image, summary=summary)

    return render_template_string("""
    <h1>Helmet Detection System</h1>
    <h3>AI-Based Helmet Detection Using YOLOv11</h3>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        Confidence Threshold:
        <input type="number" name="confidence"
               value="0.25" min="0.1" max="1" step="0.05">
        <br><br>
        <button type="submit">Detect Helmet</button>
    </form>
    """)

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
