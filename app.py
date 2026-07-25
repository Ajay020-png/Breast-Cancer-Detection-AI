from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("saved_models/breast_cancer_model.h5")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get uploaded image
        image = request.files["image"]

        if image.filename == "":
            return "Please select an image."

        # Save image
        upload_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(upload_path)

        # Preprocess image
        img = Image.open(upload_path).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)

        probability = float(prediction[0][0])

        if probability > 0.5:
            result = "Malignant"
            confidence = probability * 100
        else:
            result = "Benign"
            confidence = (1 - probability) * 100

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(confidence, 2),
            image_name=image.filename
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)