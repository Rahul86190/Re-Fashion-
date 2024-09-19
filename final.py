import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow
from io import BytesIO

feature_list = np.array(pickle.load(open("new_embeddings.pkl", "rb")))  # List converted into array
filenames = pickle.load(open("new_filenames.pkl", "rb"))

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))  # We get weights that have trained on the ImageNet dataset
model.trainable = False
model = Sequential([
    model,     # Pass the ResNet50 model in sequential layer
    GlobalMaxPooling2D()  # We added our layer GlobalMaxPooling2D instead of the top layer. It gives 2048 features
])

st.title(" 🎀 FASHION RECOMMENDATION✨")
st.write("Welcome to our Fashion Recommendation System!")
st.write("This system will recommend you the best fashion items based on your preferences.")
st.write("Please select your preferences below:")

# Sidebar for filters
st.sidebar.title("Filters")
st.sidebar.subheader("Refine your search")
category = st.sidebar.selectbox('Category', ['All', 'Shirts', 'Pants', 'Dresses', 'Shoes'])
color = st.sidebar.multiselect('Color', ['Black', 'White', 'Blue', 'Red', 'Green'])

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("./uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function for feature extraction
def feature_extraction(img, model):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array, axis=0)  # Into batch
    preprocessed_img = preprocess_input(expand_img_array)
    result = model.predict(preprocessed_img).flatten()  # Model gives the features of the given image
    normalized_result = result / norm(result)
    return normalized_result

# Function for recommending based on nearest neighbors
def recommend(features, feature_list):
    neighbours = NearestNeighbors(n_neighbors=7, algorithm='brute', metric="euclidean")
    neighbours.fit(feature_list)
    distances, indices = neighbours.kneighbors([features])
    return distances, indices

# Like/Dislike feature tracking
feedback = {}

# Allow multiple file uploads
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            display_image = Image.open(uploaded_file)
            st.image(display_image, use_column_width=True)
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            distances, indices = recommend(features, feature_list)

            # Sort recommendations by distance (similarity)
            sorted_indices = sorted(zip(distances[0], indices[0]), key=lambda x: x[0])

            # Display recommended images with like, dislike, and download buttons
            cols = st.columns(7)
            for i, (distance, idx) in enumerate(sorted_indices[:7]):
                with cols[i]:
                    st.image(filenames[idx])
                    st.write(f"Similarity: {1 - distance:.2f}")

                    # Like/Dislike buttons
                    if st.button(f"👍 Like {i+1}"):
                        feedback[filenames[idx]] = 'Liked'
                        st.success("You liked this!")

                    # Download button
                    if st.button(f"Download {i+1}"):
                        img = Image.open(filenames[idx])
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(label="Download Image", data=byte_im, file_name=f"recommended_{i+1}.png")

        else:
            st.write("Error in uploading file.")
    else:
        st.write("Please upload an image")
