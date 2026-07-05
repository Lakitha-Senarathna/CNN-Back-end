import io
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


app = FastAPI(title="Skin Cancer Classification API")

model = load_model("skin_cancer_best_model.h5")

Class_Names = ["Benign", "Malignant"]  

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))                # Resize the image to the input size expected by the model
    image_array = np.array(image)
    image_array = np.array(image_array) / 255.0     # Normalize the image
    return np.expand_dims(image, axis=0)

@app.get("/")
async def root():
    return {"message": "Hello World. Model is loaded and ready."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    image_data = preprocess_image(image)            # Preprocess the image as required by the model
    prediction = model.predict(image_data)

    predicted_class_index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
    diagnosis = Class_Names[predicted_class_index]

    return {
            "filename": file.filename,
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 2)
        }
        