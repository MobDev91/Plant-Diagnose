"""
Plant Disease Model Loader - Based on Working Architecture
Uses the exact same architecture as your working model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import json
import logging

logger = logging.getLogger(__name__)

class PlantDiseaseModel(nn.Module):
    """Plant Disease Model - Exact match to your working model"""
    
    def __init__(self, model_name='efficientnet_b4', num_classes=39, pretrained=False):
        super(PlantDiseaseModel, self).__init__()
        
        # Backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool='avg'
        )
        
        # Custom classifier matching your working model
        self.num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def load_model(model_path, device='cpu'):
    """
    Load the trained model using the exact working architecture
    """
    logger.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Checkpoint loaded successfully")
        
        # Extract model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            model_name = config.get('MODEL_NAME', 'efficientnet_b4')
            num_classes = config.get('NUM_CLASSES', 39)
            logger.info(f"Model: {model_name}, Classes: {num_classes}")
        else:
            # Fallback defaults
            model_name = 'efficientnet_b4'
            num_classes = 39
            logger.warning("No config found, using defaults")
        
        # Create the model with exact architecture
        logger.info(f"Creating {model_name} model with {num_classes} classes")
        model = PlantDiseaseModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load the trained weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Model weights loaded successfully (strict=True)")
        except Exception as e:
            logger.warning(f"Strict loading failed: {str(e)[:100]}...")
            logger.info("Trying flexible loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Flexible loading: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
            
            if len(missing_keys) > 10:  # Too many missing keys
                logger.error("Too many missing keys, model architecture mismatch")
                logger.error(f"Missing keys (first 5): {missing_keys[:5]}")
                raise ValueError("Model architecture mismatch - too many missing parameters")
        
        # Move to device and set eval mode
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        
        # Print model info if available from checkpoint
        if 'epoch' in checkpoint:
            logger.info(f"Model epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            logger.info(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

def get_model_info(model_path):
    """
    Get information about the model checkpoint
    """
    if not os.path.exists(model_path):
        return {'error': f'Model file not found: {model_path}'}
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'file_exists': True,
            'file_path': model_path,
            'file_size_mb': f"{os.path.getsize(model_path) / (1024*1024):.2f}",
            'type': type(checkpoint).__name__,
            'architecture': 'timm EfficientNet B4 with custom classifier'
        }
        
        if isinstance(checkpoint, dict):
            info['keys'] = list(checkpoint.keys())
            
            # Add specific checkpoint info
            for key in ['epoch', 'val_acc', 'config', 'class_names']:
                if key in checkpoint:
                    if key == 'class_names' and isinstance(checkpoint[key], list):
                        info[f'{key}_count'] = len(checkpoint[key])
                        info['sample_classes'] = checkpoint[key][:5]  # First 5 classes
                    else:
                        info[key] = checkpoint[key]
            
            # Count parameters if state_dict exists
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                total_params = sum(p.numel() for p in state_dict.values())
                info['total_parameters'] = total_params
                
                # Analyze classifier architecture
                classifier_layers = []
                for key in state_dict.keys():
                    if 'classifier' in key and 'weight' in key:
                        shape = state_dict[key].shape
                        classifier_layers.append(f"{key}: {shape}")
                
                info['classifier_architecture'] = classifier_layers
        
        return info
        
    except Exception as e:
        return {'error': str(e)}

def test_model_loading():
    """Test function to verify model loading works"""
    model_path = "models/best_model.pth"
    
    print("🔍 Testing Plant Disease Model Loading")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Get model info
        info = get_model_info(model_path)
        print("📊 Model Information:")
        for key, value in info.items():
            if key not in ['sample_classes', 'classifier_architecture']:
                print(f"  {key}: {value}")
        
        if 'classifier_architecture' in info:
            print("  Classifier layers:")
            for layer in info['classifier_architecture']:
                print(f"    {layer}")
        
        print("\n🧠 Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(model_path, device=device)
        
        print(f"✅ Model loaded successfully!")
        print(f"📱 Device: {device}")
        print(f"🎯 Model type: {type(model).__name__}")
        
        # Test inference
        print("\n🧪 Testing inference...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            output = model(test_input)
            probabilities = F.softmax(output, dim=1)
            
            print(f"✅ Inference successful!")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Max probability: {probabilities.max().item():.4f}")
            print(f"  Predicted class: {probabilities.argmax().item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if timm is available
    try:
        import timm
        print(f"✅ timm version: {timm.__version__}")
    except ImportError:
        print("❌ timm not installed. Install with: pip install timm")
        exit(1)
    
    # Run test
    success = test_model_loading()
    
    if success:
        print("\n🎉 Model loading test PASSED!")
        print("✅ Ready to integrate with Flask app")
    else:
        print("\n💥 Model loading test FAILED!")
        print("❌ Check the errors above")
