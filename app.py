import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from groq import Groq

st.set_page_config(
    page_title="Agri AI Disease Detection",
    page_icon="🌱",
    layout="wide"
)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

MODEL_PATH = "models/mobilenetv2_finetuned_best.keras"

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "groq_information" not in st.session_state:
    st.session_state.groq_information = None


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def format_name(name):
    name = name.replace("___", " - ")
    name = name.replace("_", " ")
    name = name.replace(",", "")
    return name


def get_plant_and_condition(class_name):
    parts = class_name.split("___")

    plant = parts[0].replace("_", " ").replace(",", "")

    if len(parts) > 1:
        condition = parts[1].replace("_", " ").replace(",", "")
    else:
        condition = "Unknown"

    return plant, condition


def get_groq_information(plant, disease):

    client = get_groq_client()

    if client is None:
        return "GROQ_API_KEY was not found in the environment."

    prompt = f"""
You are an agricultural information assistant.

A deep learning image classifier predicted:

Plant: {plant}
Condition: {disease}

Provide practical agricultural information about this prediction.

Use exactly these headings:

## About the Disease

Explain what the disease or condition is.

## Symptoms

Give 4 to 6 important visible symptoms.

## Causes

Explain the known cause, pathogen, pest, virus, fungus, bacteria, or environmental factor when established.

## Management / Solution

Give 5 to 7 practical management steps.
Do not invent pesticide names, chemical doses, concentrations, or application rates.
Mention appropriate agricultural products only in general terms when relevant.

## Prevention

Give 4 to 6 practical prevention measures.

## When to Seek Expert Help

Explain when the farmer should contact a local agricultural expert.

Important:
- The image classifier can make mistakes.
- Do not present this prediction as a confirmed diagnosis.
- Do not invent facts.
- Do not provide unsafe chemical instructions.
- Keep the information practical and easy to understand.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful agricultural information assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1600
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API error: {str(e)}"


def show_model_performance():

    st.header("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "97.20%")
    col2.metric("Precision", "97.37%")
    col3.metric("Recall", "97.20%")
    col4.metric("F1 Score", "97.16%")

    st.divider()

    st.subheader("📈 Training Accuracy")

    if os.path.exists("models/accuracy_curve.png"):
        st.image(
            "models/accuracy_curve.png",
            use_container_width=True
        )

    st.subheader("📉 Training Loss")

    if os.path.exists("models/loss_curve.png"):
        st.image(
            "models/loss_curve.png",
            use_container_width=True
        )

    st.subheader("🔲 Confusion Matrix")

    if os.path.exists("models/confusion_matrix.png"):
        st.image(
            "models/confusion_matrix.png",
            use_container_width=True
        )

    st.subheader("🔬 Frozen vs Fine-Tuned")

    st.table({
        "Model": [
            "MobileNetV2 Frozen",
            "MobileNetV2 Fine-Tuned"
        ],
        "Test Accuracy": [
            "94.33%",
            "97.20%"
        ]
    })

    st.info(
        "Fine-tuning improved test accuracy by 2.87 percentage points."
    )


def show_model_information():

    st.header("🧠 Model Information")

    information = {
        "Architecture": "MobileNetV2",
        "Approach": "Transfer Learning",
        "Fine-Tuning": "Yes",
        "Input Size": "224 × 224 × 3",
        "Number of Classes": "38",
        "Optimizer": "Adam",
        "Fine-Tuning Learning Rate": "1e-5",
        "Test Accuracy": "97.20%",
        "Precision": "97.37%",
        "Recall": "97.20%",
        "F1 Score": "97.16%"
    }

    for key, value in information.items():
        col1, col2 = st.columns([1, 2])
        col1.write(f"**{key}**")
        col2.write(value)

    st.divider()

    st.subheader("Architecture")

    st.write(
        "MobileNetV2 pretrained on ImageNet was used as the feature "
        "extractor. A custom classification head containing Global "
        "Average Pooling, Dropout and a 38-class Softmax layer was added. "
        "The later layers were then fine-tuned using a low learning rate."
    )


def show_dataset():

    st.header("📁 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Images", "54,305")
    col2.metric("Classes", "38")
    col3.metric("Image Size", "224 × 224")

    st.divider()

    st.subheader("Dataset Split")

    st.table({
        "Dataset": [
            "Training",
            "Validation",
            "Testing"
        ],
        "Images": [
            "39,236",
            "4,360",
            "10,709"
        ]
    })

    st.subheader("🌿 Plant Types")

    plants = [
        "Apple",
        "Blueberry",
        "Cherry",
        "Corn",
        "Grape",
        "Orange",
        "Peach",
        "Pepper",
        "Potato",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry",
        "Tomato"
    ]

    cols = st.columns(4)

    for i, plant in enumerate(plants):
        cols[i % 4].write(f"🌿 {plant}")

    st.subheader("🦠 Disease / Health Classes")

    for class_name in CLASS_NAMES:
        st.write(f"• {format_name(class_name)}")


st.title("🌱 AI-Powered Agriculture Disease Detection")

st.write(
    "Deep learning based plant disease classification with "
    "AI-powered agricultural information."
)

st.divider()

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("🌿 Plant Types", "14")
metric2.metric("🦠 Classes", "38")
metric3.metric("🖼️ Images", "54,305")
metric4.metric("🎯 Test Accuracy", "97.20%")

st.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Disease Detection",
        "📊 Model Performance",
        "🧠 Model Information",
        "📁 Dataset Information"
    ]
)

if page == "🔍 Disease Detection":

    st.header("📷 Upload Plant Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("🔍 Disease Detection")

            predict_button = st.button(
                "🔍 Predict Disease",
                use_container_width=True
            )

            if predict_button:

                with st.spinner("Analyzing leaf image..."):

                    model = load_model()

                    image_resized = image.resize((224, 224))

                    image_array = np.array(image_resized)

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    top_indices = np.argsort(
                        predictions
                    )[-3:][::-1]

                    predicted_index = top_indices[0]

                    predicted_class = CLASS_NAMES[
                        predicted_index
                    ]

                    confidence = (
                        predictions[predicted_index] * 100
                    )

                    plant, condition = get_plant_and_condition(
                        predicted_class
                    )

                    st.session_state.prediction_data = {
                        "predictions": predictions,
                        "top_indices": top_indices,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "plant": plant,
                        "condition": condition
                    }

                    st.session_state.groq_information = None

    if st.session_state.prediction_data is not None:

        data = st.session_state.prediction_data

        predictions = data["predictions"]
        top_indices = data["top_indices"]
        predicted_class = data["predicted_class"]
        confidence = data["confidence"]
        plant = data["plant"]
        condition = data["condition"]

        st.divider()

        st.success("Prediction completed!")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.subheader("🌿 Plant")

            st.markdown(
                f"### {plant}"
            )

        with result_col2:

            st.subheader("🦠 Detected Condition")

            st.markdown(
                f"### {condition}"
            )

        st.write(
            f"**Confidence: {confidence:.2f}%**"
        )

        st.progress(
            float(predictions[top_indices[0]])
        )

        st.divider()

        st.subheader("📊 Top 3 Predictions")

        for rank, index in enumerate(top_indices):

            class_name = CLASS_NAMES[index]

            probability = predictions[index] * 100

            st.write(
                f"**{rank + 1}. {format_name(class_name)}**"
            )

            st.progress(
                float(predictions[index])
            )

            st.caption(
                f"Confidence: {probability:.2f}%"
            )

        st.divider()

        if "healthy" in condition.lower():

            st.subheader("✅ Plant Health Information")

            st.success(
                "The model classified this leaf as healthy."
            )

            st.write(
                "Continue regular monitoring, maintain proper "
                "irrigation and nutrition, and inspect nearby "
                "plants periodically for changes."
            )

        else:

            st.subheader("🌿 Disease Information")

            st.info(
                "The information below is generated by Groq based "
                "on the predicted disease. The image classification "
                "is not a confirmed agricultural diagnosis."
            )

            get_info_button = st.button(
                "🤖 Get Disease Information & Solution",
                use_container_width=True
            )

            if get_info_button:

                with st.spinner(
                    "Generating disease information..."
                ):

                    information = get_groq_information(
                        plant,
                        condition
                    )

                    st.session_state.groq_information = information

        if st.session_state.groq_information:

            st.divider()

            st.subheader(
                "📚 Disease Information & Management"
            )

            st.markdown(
                st.session_state.groq_information
            )

        st.divider()

        st.subheader("⚠️ Important")

        st.caption(
            "This application is an educational and research "
            "prototype. Predictions may be incorrect. For important "
            "crop-management decisions, consult a qualified "
            "agricultural professional."
        )


elif page == "📊 Model Performance":

    show_model_performance()


elif page == "🧠 Model Information":

    show_model_information()


elif page == "📁 Dataset Information":

    show_dataset()


st.divider()

st.caption(
    "AI-Powered Agriculture Disease Detection | "
    "Deep Learning CA-1 Project"
)