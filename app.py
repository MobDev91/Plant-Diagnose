#!/usr/bin/env python3
"""
PlantDiagnose - Advanced Plant Disease Detection Web Application
Author: AI Assistant
Description: Modern Flask web application for plant disease detection using deep learning
"""

import os
import io
import base64
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from simple_model_loader import load_model as load_trained_model, get_model_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'plant-disease-detection-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MODEL_PATH = 'models/best_model.pth'
CLASS_NAMES_PATH = 'models/class_names.json'
CLASS_DISTRIBUTION_PATH = 'models/class_distribution.csv'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease information database
DISEASE_INFO = {
    "Apple_scab": {
        "description": "A fungal disease that affects apple trees, causing dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides preventively, ensure good air circulation, and remove infected debris.",
        "severity": "Moderate",
        "prevention": "Regular pruning, proper spacing, and resistant varieties."
    },
    "Black_rot": {
        "description": "A serious fungal disease causing black, circular lesions on leaves and fruit rot.",
        "treatment": "Remove infected parts, apply copper-based fungicides, improve drainage.",
        "severity": "High",
        "prevention": "Avoid overhead watering, ensure good air circulation."
    },
    "Cedar_apple_rust": {
        "description": "A fungal disease requiring both apple and cedar trees to complete its life cycle.",
        "treatment": "Apply fungicides, remove nearby cedar trees if possible.",
        "severity": "Moderate",
        "prevention": "Plant resistant varieties, maintain distance from cedar trees."
    },
    "Powdery_mildew": {
        "description": "A fungal disease creating white, powdery coating on leaves and stems.",
        "treatment": "Apply sulfur or potassium bicarbonate sprays, improve air circulation.",
        "severity": "Moderate",
        "prevention": "Avoid overhead watering, ensure proper spacing."
    },
    "Cercospora_leaf_spot": {
        "description": "Fungal disease causing gray-brown spots with dark borders on corn leaves.",
        "treatment": "Apply fungicides, rotate crops, remove infected debris.",
        "severity": "Moderate",
        "prevention": "Crop rotation, resistant varieties, proper field sanitation."
    },
    "Common_rust": {
        "description": "Fungal disease causing orange-brown pustules on corn leaves.",
        "treatment": "Apply fungicides if severe, plant resistant varieties.",
        "severity": "Low to Moderate",
        "prevention": "Resistant varieties, proper field management."
    },
    "Northern_Leaf_Blight": {
        "description": "Fungal disease causing long, elliptical lesions on corn leaves.",
        "treatment": "Apply fungicides, practice crop rotation, remove debris.",
        "severity": "High",
        "prevention": "Resistant varieties, crop rotation, field sanitation."
    },
    "Esca": {
        "description": "Complex disease affecting grape vines, causing leaf yellowing and wood decay.",
        "treatment": "Prune infected wood, apply protective fungicides.",
        "severity": "High",
        "prevention": "Proper pruning practices, wound protection."
    },
    "Leaf_blight": {
        "description": "Fungal disease causing brown lesions on grape leaves.",
        "treatment": "Apply copper-based fungicides, improve air circulation.",
        "severity": "Moderate",
        "prevention": "Proper canopy management, resistant varieties."
    },
    "Haunglongbing": {
        "description": "Bacterial disease spread by insects, causing citrus greening.",
        "treatment": "No cure available, remove infected trees, control vectors.",
        "severity": "Very High",
        "prevention": "Vector control, certified clean planting material."
    },
    "Bacterial_spot": {
        "description": "Bacterial disease causing dark spots on leaves and fruit.",
        "treatment": "Apply copper sprays, improve air circulation, avoid overhead watering.",
        "severity": "Moderate to High",
        "prevention": "Resistant varieties, proper sanitation, avoid wet conditions."
    },
    "Early_blight": {
        "description": "Fungal disease causing dark spots with concentric rings on leaves.",
        "treatment": "Apply fungicides, remove infected debris, improve air circulation.",
        "severity": "Moderate",
        "prevention": "Crop rotation, resistant varieties, proper spacing."
    },
    "Late_blight": {
        "description": "Destructive fungal disease causing rapid leaf and fruit decay.",
        "treatment": "Apply fungicides immediately, remove infected plants.",
        "severity": "Very High",
        "prevention": "Resistant varieties, avoid wet conditions, proper sanitation."
    },
    "Leaf_scorch": {
        "description": "Fungal disease causing purple-red spots on strawberry leaves.",
        "treatment": "Apply fungicides, remove infected leaves, improve air circulation.",
        "severity": "Moderate",
        "prevention": "Resistant varieties, proper spacing, avoid overhead watering."
    },
    "Leaf_Mold": {
        "description": "Fungal disease causing yellowing and moldy growth on tomato leaves.",
        "treatment": "Improve ventilation, apply fungicides, reduce humidity.",
        "severity": "Moderate",
        "prevention": "Good air circulation, avoid overhead watering."
    },
    "Septoria_leaf_spot": {
        "description": "Fungal disease causing small, dark spots with light centers on tomato leaves.",
        "treatment": "Apply fungicides, remove infected debris, improve air circulation.",
        "severity": "Moderate",
        "prevention": "Crop rotation, proper spacing, avoid wet foliage."
    },
    "Spider_mites": {
        "description": "Tiny pests causing stippling and webbing on tomato leaves.",
        "treatment": "Apply miticides, increase humidity, use beneficial insects.",
        "severity": "Moderate",
        "prevention": "Maintain proper humidity, regular monitoring."
    },
    "Target_Spot": {
        "description": "Fungal disease causing circular spots with concentric rings on tomato leaves.",
        "treatment": "Apply fungicides, remove infected debris, improve air circulation.",
        "severity": "Moderate",
        "prevention": "Crop rotation, resistant varieties, proper sanitation."
    },
    "Tomato_mosaic_virus": {
        "description": "Viral disease causing mottled leaves and reduced fruit quality.",
        "treatment": "No cure available, remove infected plants, control vectors.",
        "severity": "High",
        "prevention": "Use certified seeds, control aphids, proper sanitation."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease causing yellowing and curling of tomato leaves.",
        "treatment": "No cure available, remove infected plants, control whiteflies.",
        "severity": "High",
        "prevention": "Control whiteflies, use resistant varieties, proper sanitation."
    }
}

# Global model variable and class data
model = None
PLANT_CLASSES = {}
CLASS_DISTRIBUTION = {}

def load_class_names():
    """Load class names from JSON file"""
    global PLANT_CLASSES
    try:
        import json
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_list = json.load(f)
        
        # Convert list to dictionary with indices
        PLANT_CLASSES = {i: class_name for i, class_name in enumerate(class_list)}
        logger.info(f"📋 Loaded {len(PLANT_CLASSES)} class names from {CLASS_NAMES_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading class names: {str(e)}")
        return False

def load_class_distribution():
    """Load class distribution data from CSV file"""
    global CLASS_DISTRIBUTION
    try:
        import pandas as pd
        df = pd.read_csv(CLASS_DISTRIBUTION_PATH, index_col=0)
        CLASS_DISTRIBUTION = df.to_dict('index')
        logger.info(f"📊 Loaded class distribution data for {len(CLASS_DISTRIBUTION)} classes")
        return True
    except ImportError:
        logger.warning("⚠️ pandas not available, loading class distribution with basic CSV reader")
        try:
            import csv
            with open(CLASS_DISTRIBUTION_PATH, 'r') as f:
                reader = csv.DictReader(f)
                CLASS_DISTRIBUTION = {row[''].replace('"', ''): {
                    'train': int(row['train']), 
                    'val': int(row['val']), 
                    'test': int(row['test']), 
                    'total': int(row['total'])
                } for row in reader}
            logger.info(f"📊 Loaded class distribution data for {len(CLASS_DISTRIBUTION)} classes")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading class distribution: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading class distribution: {str(e)}")
        return False

def load_model():
    """Load the trained PyTorch model with enhanced error handling"""
    global model
    try:
        # Load class names and distribution data first
        logger.info("📚 Loading class data...")
        if not load_class_names():
            logger.error("❌ Failed to load class names")
            return False
        
        if not load_class_distribution():
            logger.warning("⚠️ Could not load class distribution data (optional)")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            logger.error(f"Current working directory: {os.getcwd()}")
            available_models = [f for f in os.listdir('.') if f.endswith('.pth')]
            logger.error(f"Available .pth files: {available_models}")
            return False
        
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        logger.info(f"📁 Found model file: {MODEL_PATH} ({file_size:.1f} MB)")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Loading model on device: {device}")
        
        # Get model info first
        logger.info("🔍 Analyzing model structure...")
        model_info = get_model_info(MODEL_PATH)
        
        if 'error' in model_info:
            logger.error(f"❌ Model analysis failed: {model_info['error']}")
            return False
        
        logger.info(f"✅ Model analysis successful:")
        for key, value in model_info.items():
            if key not in ['sample_layers', 'layers']:
                logger.info(f"   {key}: {value}")
        
        # Load the model using our integrated loader
        logger.info("🧠 Loading model...")
        model = load_trained_model(MODEL_PATH, device=device)
        model.eval()
        
        # Test the model with dummy input
        logger.info("🎯 Testing model with dummy input...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            test_output = model(test_input)
            probabilities = torch.nn.functional.softmax(test_output, dim=1)
            max_prob = probabilities.max().item()
            logger.info(f"✅ Model test successful:")
            logger.info(f"   Output shape: {test_output.shape}")
            logger.info(f"   Max probability: {max_prob:.4f}")
            logger.info(f"   Classes: {test_output.shape[1]}")
            
            # Verify class count matches
            if test_output.shape[1] != len(PLANT_CLASSES):
                logger.warning(f"⚠️ Model output classes ({test_output.shape[1]}) != loaded classes ({len(PLANT_CLASSES)})")
            else:
                logger.info(f"✅ Class count matches: {len(PLANT_CLASSES)} classes")
        
        logger.info("🎉 Model loaded and tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Define transforms (adjust based on your model's training preprocessing)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(image):
    """Predict plant disease from image"""
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        if image_tensor is None:
            return None, "Error preprocessing image"
        
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            top3_predictions = []
            
            for i in range(3):
                class_idx = top3_indices[0][i].item()
                prob = top3_prob[0][i].item()
                class_name = PLANT_CLASSES[class_idx]
                top3_predictions.append({
                    'class': class_name,
                    'confidence': prob * 100
                })
            
            return {
                'predicted_class': PLANT_CLASSES[predicted_class],
                'confidence': confidence_score * 100,
                'top3_predictions': top3_predictions
            }, None
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None, f"Prediction error: {str(e)}"

def get_class_statistics(class_name):
    """Get statistics for a specific class"""
    clean_name = class_name.replace('___', '___').replace('_', ' ')
    
    # Try different name variations to match CSV data
    for key in CLASS_DISTRIBUTION.keys():
        if key.replace('_', ' ') == clean_name or key == class_name:
            return CLASS_DISTRIBUTION[key]
    
    return {
        'train': 'N/A',
        'val': 'N/A', 
        'test': 'N/A',
        'total': 'N/A'
    }

def parse_prediction(prediction_result):
    """Parse prediction result and extract plant and disease information"""
    if not prediction_result:
        return None
    
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    
    # Parse plant name and disease
    if '___' in predicted_class:
        plant_name, disease_name = predicted_class.split('___', 1)
        plant_name = plant_name.replace('_', ' ').replace('(', '').replace(')', '')
        disease_name = disease_name.replace('_', ' ')
    else:
        plant_name = predicted_class.replace('_', ' ')
        disease_name = "Unknown"
    
    # Get disease information
    disease_key = disease_name.replace(' ', '_').replace('(', '').replace(')', '')
    disease_info = DISEASE_INFO.get(disease_key, {
        'description': 'No information available for this condition.',
        'treatment': 'Consult with agricultural experts.',
        'severity': 'Unknown',
        'prevention': 'Follow general plant care practices.'
    })
    
    is_healthy = 'healthy' in disease_name.lower()
    
    return {
        'plant_name': plant_name,
        'disease_name': disease_name,
        'confidence': confidence,
        'is_healthy': is_healthy,
        'disease_info': disease_info,
        'top3_predictions': prediction_result['top3_predictions'],
        'class_statistics': get_class_statistics(predicted_class)
    }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Read image
                image = Image.open(file.stream)
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                image.save(filepath)
                
                # Make prediction
                prediction, error = predict_disease(image)
                
                if error:
                    flash(f'Prediction error: {error}')
                    return redirect(url_for('upload_file'))
                
                # Parse results
                result = parse_prediction(prediction)
                
                return render_template('result.html', 
                                     result=result, 
                                     image_path=f'uploads/{filename}')
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(url_for('upload_file'))
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/camera')
def camera():
    """Camera capture page"""
    return render_template('camera.html')

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    """Handle camera image prediction"""
    try:
        # Get image data from request
        image_data = request.json.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save captured image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        # Make prediction
        prediction, error = predict_disease(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Parse results
        result = parse_prediction(prediction)
        result['image_path'] = f'uploads/{filename}'
        result['filename'] = filename
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Error in camera prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/camera-result')
def camera_result():
    """Display camera capture results"""
    # For camera results, we'll use a simple HTML page that handles the sessionStorage data
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Camera Analysis Result - PlantDiagnose</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; text-align: center; }
        .loading { color: #059669; }
    </style>
</head>
<body>
    <div class="loading">
        <h3>Loading analysis results...</h3>
        <p>Please wait while we display your plant analysis.</p>
    </div>
    
    <script>
        // Get result from sessionStorage
        const result = JSON.parse(sessionStorage.getItem('cameraResult') || '{}');
        
        if (result && result.plant_name) {
            // Create a form to POST to results page
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = '/display-camera-result';
            
            const resultInput = document.createElement('input');
            resultInput.type = 'hidden';
            resultInput.name = 'result_data';
            resultInput.value = JSON.stringify(result);
            
            form.appendChild(resultInput);
            document.body.appendChild(form);
            form.submit();
        } else {
            // No result data, redirect to camera page
            setTimeout(() => {
                window.location.href = '/camera';
            }, 2000);
            document.querySelector('.loading').innerHTML = '<h3>No result data found</h3><p>Redirecting to camera page...</p>';
        }
    </script>
</body>
</html>
    '''

@app.route('/display-camera-result', methods=['POST'])
def display_camera_result():
    """Display camera analysis results"""
    try:
        result_data = request.form.get('result_data')
        if not result_data:
            flash('No result data provided')
            return redirect(url_for('camera'))
        
        import json
        result = json.loads(result_data)
        
        return render_template('result.html', 
                             result=result, 
                             image_path=result.get('image_path', ''))
        
    except Exception as e:
        logger.error(f"Error displaying camera results: {str(e)}")
        flash('Error displaying results')
        return redirect(url_for('camera'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Process image
        image = Image.open(file.stream)
        prediction, error = predict_disease(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        result = parse_prediction(prediction)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500













@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting PlantDiagnose application...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("Failed to load model. Exiting...")
