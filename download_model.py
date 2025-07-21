#!/usr/bin/env python3
"""
Model Download Script for PlantDiagnose

This script helps users download the trained model file required for the application.
The model file is too large to be stored in Git (224MB) so it needs to be downloaded separately.
"""

import os
import sys
import argparse
from pathlib import Path

def check_model_exists():
    """Check if the model file already exists"""
    model_path = Path("models/best_model.pth")
    return model_path.exists()

def get_model_info():
    """Display information about the required model"""
    print("🌿 PlantDiagnose Model Information")
    print("=" * 50)
    print("Model: EfficientNet B4 + Custom Classifier")
    print("File: models/best_model.pth")
    print("Size: ~224MB")
    print("Classes: 39 plant disease categories")
    print("Accuracy: 99.45% on validation set")
    print("Framework: PyTorch + timm")
    print()

def create_model_directory():
    """Create the models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Created/verified models directory: {models_dir}")

def download_instructions():
    """Provide instructions for obtaining the model"""
    print("📥 Model Download Instructions")
    print("=" * 50)
    print()
    print("The trained model file is not included in this repository due to its size.")
    print("Please choose one of the following options:")
    print()
    print("Option 1: Contact Repository Owner")
    print("- Contact the repository maintainer for the trained model file")
    print("- Request access to the pre-trained model")
    print("- Download and place in: models/best_model.pth")
    print()
    print("Option 2: Train Your Own Model")
    print("- Use the model architecture in simple_model_loader.py")
    print("- Train on plant disease datasets (PlantVillage, etc.)")
    print("- Save the trained model as: models/best_model.pth")
    print()
    print("Option 3: Use Placeholder Model (for testing)")
    print("- Run: python download_model.py --create-placeholder")
    print("- Creates a dummy model for testing the interface")
    print("- Note: This won't provide real disease detection")
    print()

def create_placeholder_model():
    """Create a placeholder model for testing purposes"""
    try:
        import torch
        import torch.nn as nn
        from simple_model_loader import PlantDiseaseModel
        
        print("🔧 Creating placeholder model...")
        
        # Create model with same architecture
        model = PlantDiseaseModel(
            model_name='efficientnet_b4',
            num_classes=39,
            pretrained=False
        )
        
        # Create a dummy checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'MODEL_NAME': 'efficientnet_b4',
                'NUM_CLASSES': 39
            },
            'epoch': 0,
            'val_acc': 0.0,
            'note': 'This is a placeholder model for testing the interface only. It will not provide accurate disease detection.'
        }
        
        # Save the placeholder model
        model_path = Path("models/best_model.pth")
        torch.save(checkpoint, model_path)
        
        print(f"✅ Placeholder model created: {model_path}")
        print("⚠️  WARNING: This is a placeholder model for testing only!")
        print("   It will not provide accurate disease detection results.")
        print("   Please obtain a properly trained model for production use.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for creating placeholder model")
        print(f"   {e}")
        print("   Please install requirements first: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error creating placeholder model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download or setup model for PlantDiagnose')
    parser.add_argument('--info', action='store_true', help='Show model information')
    parser.add_argument('--create-placeholder', action='store_true', help='Create placeholder model for testing')
    parser.add_argument('--check', action='store_true', help='Check if model exists')
    
    args = parser.parse_args()
    
    print("🌿 PlantDiagnose Model Setup")
    print("=" * 30)
    print()
    
    # Create models directory
    create_model_directory()
    
    # Check if model already exists
    if check_model_exists():
        print("✅ Model file found: models/best_model.pth")
        if args.check:
            sys.exit(0)
        print("   The application should work correctly.")
        print("   Run: python app.py")
        return
    
    if args.info:
        get_model_info()
        return
    
    if args.check:
        print("❌ Model file not found: models/best_model.pth")
        sys.exit(1)
    
    if args.create_placeholder:
        success = create_placeholder_model()
        if success:
            print()
            print("🚀 You can now test the application interface:")
            print("   python app.py")
        return
    
    # Default: show download instructions
    get_model_info()
    download_instructions()

if __name__ == "__main__":
    main()
