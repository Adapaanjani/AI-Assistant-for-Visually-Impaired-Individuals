import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os
import cv2
import numpy as np
from dotenv import load_dotenv
import base64
import pytesseract
from gtts import gTTS
import torch

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Load pre-trained object detection model
def load_object_detection_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Perform object detection
def detect_objects(image):
    model = load_object_detection_model()
    results = model(image)
    
    # Process and extract detected objects
    objects = []
    for *xyxy, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        confidence = conf.item()
        if confidence > 0.5:
            objects.append({
                'label': label,
                'confidence': round(confidence, 2)
            })
    
    return objects

# Generate detailed scene description
def get_image_description(image):
    response = model.generate_content([
        "Describe this image in detail for a visually impaired person. Include spatial relationships, colors, objects, and potential hazards.",
        image
    ])
    return response.text

# Extract text from image
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Convert text to speech
def text_to_speech(text):
    audio_bytes = io.BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return base64.b64encode(audio_bytes.read()).decode()

# Create audio playback element
def create_audio_element(text):
    return f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{text_to_speech(text)}" type="audio/mp3">
        </audio>
        """

# Provide personalized task assistance
def get_task_guidance(image, detected_objects):
    # Generate context-specific guidance based on detected objects
    prompt = f"Provide helpful guidance for a visually impaired person based on these objects: {', '.join([obj['label'] for obj in detected_objects])}"
    response = model.generate_content([
        prompt,
        image
    ])
    return response.text

# Main Streamlit application
def main():
    st.title("Comprehensive AI Assistant for Visually Impaired")
    st.write("Upload an image for detailed analysis and assistance")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Open and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL Image to OpenCV format for object detection
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Scene Understanding
        description = get_image_description(image)
        st.subheader("Scene Description:")
        st.write(description)
        st.markdown(create_audio_element(description), unsafe_allow_html=True)
        
        # 2. Text Extraction and Text-to-Speech
        extracted_text = extract_text_from_image(image)
        if extracted_text.strip():
            st.subheader("Extracted Text:")
            st.write(extracted_text)
            st.markdown(create_audio_element(extracted_text), unsafe_allow_html=True)
        
        # 3. Object and Obstacle Detection
        detected_objects = detect_objects(cv_image)
        st.subheader("Detected Objects:")
        for obj in detected_objects:
            st.write(f"{obj['label']} (Confidence: {obj['confidence']})")
        
        # 4. Personalized Task Assistance
        if detected_objects:
            st.subheader("Personalized Guidance:")
            task_guidance = get_task_guidance(image, detected_objects)
            st.write(task_guidance)
            st.markdown(create_audio_element(task_guidance), unsafe_allow_html=True)

if __name__ == "__main__":
    main()