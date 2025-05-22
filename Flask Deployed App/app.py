import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import google.generativeai as genai
import json
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Gemini API
def setup_gemini():
    # Get API key from environment variable
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Configure the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1024,
    }
    
    # Use Gemini-1.5-pro for advanced capabilities
    model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                                 generation_config=generation_config)
    return model

# Plant health prompts and context
PLANT_CONTEXT = """
You are an AI-powered plant health assistant integrated into a plant disease detection website. 
Your job is to provide helpful, accurate, and easy-to-understand responses to users who upload leaf images 
and want to understand plant health and diseases. Respond in Hindi or Hinglish based on how the user asks.

Website users are typically farmers, gardeners, students, or agriculture researchers. 
Your tone should be friendly, supportive, and informative.
"""

# Initialize Gemini model
gemini_model = setup_gemini()


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# New route for the chat interface
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

# API endpoint for the chat functionality
@app.route('/api/chat', methods=['POST'])
def chat_api():
    try:
        # Check if the request has form data (multipart/form-data with image)
        if request.files and 'image' in request.files:
            # Get the message from form data
            user_message = request.form.get('message', '')
            image_file = request.files['image']
            
            # Save the image temporarily
            image_path = None
            if image_file and image_file.filename != '':
                # Ensure filename is secure
                filename = secure_filename(image_file.filename)
                # Create a timestamped filename to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                new_filename = f"{timestamp}_{filename}"
                # Save to uploads folder
                image_path = os.path.join('static/chat_uploads', new_filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                
                # Save the file
                image_file.save(image_path)
            
            # Prepare different prompt if there's an image
            if image_path:
                # Create a prompt with context about the image
                prompt = f"{PLANT_CONTEXT}\n\nThe user has uploaded a plant image and asked: '{user_message}'. Please analyze the plant condition based on their question and provide helpful advice.\n\nAnswer:"
            else:
                prompt = f"{PLANT_CONTEXT}\n\nUser Question: {user_message}\n\nAnswer:"
            
        else:
            # Handle regular JSON data without image
            data = request.get_json()
            user_message = data.get('message', '')
            
            if not user_message:
                return jsonify({"error": "No message provided"}), 400
            
            # Prepare the prompt with context
            prompt = f"{PLANT_CONTEXT}\n\nUser Question: {user_message}\n\nAnswer:"
        
        # Generate response from Gemini
        response = gemini_model.generate_content(prompt)
        
        # Extract the text response
        ai_response = response.text
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
