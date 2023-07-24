import uvicorn
from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import cv2
import numpy as np

app = FastAPI()

# Load the trained model
model = keras.models.load_model('F:/4th Smester/work/drowziness model/detection.h5')
img_size = 224

# Preprocess image
def preprocess_image(image):
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    img = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)
    img = preprocess_image(img)

    prediction = model.predict(img)
    class_label = "Open_Eyes" if prediction[0][0] < 0.5 else "Close_Eyes"
    return {"class_label": class_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
