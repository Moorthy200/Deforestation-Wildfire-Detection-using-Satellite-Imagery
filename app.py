import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import ee
import datetime
import numpy as np

app = Flask(__name__)

# Initialize Earth Engine
try:
    ee.Initialize(project='detection-452708')
except Exception as e:
    print("Earth Engine initialization failed:", str(e))

# Model loading functions
def load_deforestation_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("deforestation_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def load_wildfire_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("wildfire_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Load models
deforestation_model = load_deforestation_model()
wildfire_model = load_wildfire_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_deforestation(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = deforestation_model(image)
        prediction = output.item()
    
    return prediction

def predict_wildfire(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = wildfire_model(image)
        prediction = output.item()
    
    return prediction

def get_sample_satellite_image():
    try:
        # Default coordinates for a sample area
        region = ee.Geometry.Rectangle([-62.2, -11.5, -61.8, -11.2])
        
        # Get Sentinel-2 imagery for deforestation
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(region)
            .filterDate('2020-01-01', '2023-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        # Get FIRMS data for wildfire
        firms_collection = (ee.ImageCollection('FIRMS')
            .filterBounds(region)
            .filterDate('2020-01-01', '2023-12-31'))
        
        # Get median image
        s2_image = s2_collection.median()
        firms_image = firms_collection.mosaic()
        
        # Get URL for visualization
        s2_vis_params = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
        
        firms_vis_params = {
            'bands': ['T21'],
            'min': 300,
            'max': 400,
            'palette': ['blue', 'yellow', 'red']
        }
        
        # Combine both images
        combined_image = s2_image.visualize(s2_vis_params).blend(
            firms_image.visualize(firms_vis_params)
        )
        
        url = combined_image.getThumbURL({
            'region': region,
            'dimensions': 512,
            'format': 'png'
        })
        
        return url
    except Exception as e:
        print("Earth Engine error:", str(e))
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_deforestation', methods=['POST'])
def predict_deforestation_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    img_bytes = file.read()
    try:
        prediction = predict_deforestation(img_bytes)
        result = "Deforestation Detected" if prediction > 0.5 else "No Deforestation Detected"
        confidence = f"{prediction*100:.2f}%" if prediction > 0.5 else f"{(1-prediction)*100:.2f}%"
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'prediction': float(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_wildfire', methods=['POST'])
def predict_wildfire_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    img_bytes = file.read()
    try:
        prediction = predict_wildfire(img_bytes)
        result = "Wildfire Detected" if prediction > 0.5 else "No Wildfire Detected"
        confidence = f"{prediction*100:.2f}%" if prediction > 0.5 else f"{(1-prediction)*100:.2f}%"
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'prediction': float(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_satellite_image', methods=['POST'])
def get_satellite_image_route():
    try:
        image_url = get_sample_satellite_image()
        if not image_url:
            return jsonify({'error': 'Failed to get satellite image'}), 500
            
        return jsonify({'image_url': image_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
