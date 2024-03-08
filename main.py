import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

# Page navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Single Image Prediction', 'Multiple Images Prediction', 'About'])


def load_and_preprocess_image(image):
    img = Image.open(image).resize((128, 128))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


def predict(model, image):
    processed_image = load_and_preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.round(prediction).astype(int)[0][0]


## Load Model
model_path = 'cherry_leaf_model.h5'
model = load_model(model_path)

if page == 'Single Image Prediction':
    st.title("🍒 Cherry Leaf Disease Detection")

    # Custom styling for the prediction text and center alignment
    st.markdown("""
        <style>
        .big-font {
            font-size:25px !important;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-left: 52px
        }
        .image-display {
            display: flex;
            justify-content: center;
        }
        .stButton>button {
            font-size: 20px;
            width: 200px; 
            height: 50px;  
            border: 2px solid #4CAF50;
            border-radius: 25px;
            margin: 10px 0;  
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-left: 266px;
        }
        </style>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a cherry leaf image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.markdown('<div class="image-display">', unsafe_allow_html=True)
        st.image(uploaded_file, caption='Uploaded Image', width=700)
        st.markdown('</div>', unsafe_allow_html=True)

        # Centrally aligned 'Predict' button
        if st.button('Predict', key='single_predict'):
            prediction = predict(model, uploaded_file)
            prediction_text = "The leaf is healthy." if prediction == 0 else "The leaf has powdery mildew."
            st.markdown(f'<p class="big-font">{prediction_text}</p>', unsafe_allow_html=True)

elif page == 'Multiple Images Prediction':
    st.title("🍒 Cherry Leaf Disease Detection - Multiple Images")

    uploaded_files = st.file_uploader("Upload cherry leaf images", type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True)

    # Custom button styling
    st.markdown("""
        <style>
        .stButton>button {
            font-size: 20px;
            width: 200px;
            height: 50px;
            border: 2px solid #4CAF50;
            border-radius: 25px;
            margin: 10px 0;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .prediction-text {
            font-size: 20px;
            color: #FF4B4B;
            font-weight: bold;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    if uploaded_files:
        # Use st.columns instead of st.beta_columns as beta_columns is deprecated
        cols = st.beta_columns(3)
        predictions = []

        if 'predictions' not in st.session_state:
            st.session_state.predictions = []

        if st.button('Predict All'):
            with st.spinner('Predicting...'):
                predictions = [predict(model, file) for file in uploaded_files]
                st.session_state.predictions = predictions  # Store predictions in session state

        col_index = 0
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[col_index % 3]
            with col:
                st.image(uploaded_file, use_column_width=True)
                if st.session_state.predictions:
                    # Display prediction result under each image
                    prediction_text = "Healthy" if st.session_state.predictions[idx] == 0 else "Powdery Mildew"
                    st.markdown(f'<div class="prediction-text">{prediction_text}</div>', unsafe_allow_html=True)
            col_index += 1

        if predictions:
            healthy_count = predictions.count(0)
            mildew_count = predictions.count(1)
            st.markdown(f"""
                            <div style="font-size: 20px; font-weight: bold; color: #4CAF50; text-align: center;">
                                Healthy: {healthy_count}, Powdery Mildew: {mildew_count}
                            </div>
                            """, unsafe_allow_html=True)

            data = pd.DataFrame({'Condition': ['Healthy', 'Powdery Mildew'], 'Count': [healthy_count, mildew_count]})
            chart = alt.Chart(data).mark_bar().encode(x='Condition', y='Count')
            st.altair_chart(chart, use_container_width=True)

elif page == 'About':
    st.title("About Powdery Mildew")

    st.markdown("""
        ## 🌿 Powdery Mildew in Cherry Plants
        Powdery mildew is a common fungal disease that affects a wide variety of plants. It appears as white or gray powdery spots on the leaves and stems of plants. 🍃
 """)

    st.image('mildew.png', caption='Powdery Mildew on Cherry Leaves', use_column_width=True)

    st.markdown("""
     Effective treatment involves the use of fungicides and proper gardening practices to prevent the spread of the disease. 🌱

        The model is designed to detect powdery mildew in cherry plants using images of the leaves. By leveraging deep learning techniques, it can accurately distinguish between healthy leaves 🍒 and those affected by the disease. 🍂

    """)

    st.image('diagram.png', caption='Model Architecture', use_column_width=True)

