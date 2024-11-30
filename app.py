import streamlit as st
import requests
from PIL import Image
from streamlit_lottie import st_lottie
import json

# Set the FastAPI endpoint URL
API_URL = "http://localhost:8000/Dog_Breed_Predict/"

# Load Lottie animation function
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load health Lottie animation (replace with your own path)
lottie_health = load_lottiefile("./assets/dog.json")

# Define the Streamlit app
def app():
    # Display the Lottie animation before content
    st_lottie(lottie_health, speed=1, width=800, height=400, key="dog-lottie")

    st.title("Dog Breed Classifier")
    st.write("Upload an image of a dog, and the app will predict its breed!")

    # File uploader for the image
    file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Display the uploaded image
        image = Image.open(file)
        st.image(image, caption="Uploaded Image")

        # Predict button
        if st.button("Predict"):
            # Read the image as bytes
            image_bytes = file.getvalue()

            try:
                with st.spinner("Predicting..."):
                    response = requests.post(API_URL, files={"file": ("image.jpg", image_bytes, file.type)})
                    response.raise_for_status()
                    prediction = response.json()

                # Display the predicted breed with a beautiful quote
                predicted_breed = prediction["predicted_breed"]
                st.markdown(f"""
                #### 🎉 **Prediction Successful!**
                "The **{predicted_breed}** is an amazing dog! 🐕"
                """, unsafe_allow_html=True)
                st.balloons()  # Celebrate with balloons

                

            except requests.exceptions.RequestException as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    app()
