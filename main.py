import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Load the trained model
try:
    model = tf.keras.models.load_model("dog_breed_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model. Ensure 'dog_breed_model.h5' is present and compatible.")

# Initialize the FastAPI app
app = FastAPI()

# Define the class names
class_names = [
    "Beagle", "Boxer", "Bulldog", "Dachshund",
    "German_Shepherd", "Golden_Retriever",
    "Labrador_Retriever", "Poodle", "Rottweiler", "Yorkshire_Terrier"
]

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    try:
        # Convert to RGB if the image is not in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize the image to the model's expected input size
        image = image.resize((150, 150))
        # Normalize image to [0, 1]
        image = np.array(image) / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError(f"Error in image preprocessing: {e}")

@app.get("/")
def index():
    return {"message": "Welcome to the Dog Breed Prediction API"}

@app.post("/Dog_Breed_Predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Check file content type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400, detail="Invalid file type. Only JPEG and PNG images are supported."
            )

        # Read the uploaded image file
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make the prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Return the predicted class name
        return {"predicted_breed": class_names[predicted_class]}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # Log and return a generic error
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
