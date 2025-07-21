#!/usr/bin/env python3
"""
Setup Script for PlantDiagnose
Automated setup for the plant disease detection application
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("🌿" + "=" * 48 + "🌿")
    print("   PlantDiagnose - AI Plant Disease Detection")
    print("   Automated Setup Script")
    print("🌿" + "=" * 48 + "🌿")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_pip():
    """Check if pip is available"""
    print("📦 Checking pip availability...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("   ✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ pip is not available")
        return False

def create_virtual_environment():
    """Create a virtual environment"""
    print("🏗️  Creating virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("   ✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("   ✅ Virtual environment created: venv/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get the appropriate pip command for the platform"""
    system = platform.system()
    if system == "Windows":
        return ["venv\\Scripts\\python", "-m", "pip"]
    else:
        return ["venv/bin/python", "-m", "pip"]

def install_requirements():
    """Install Python requirements"""
    print("📚 Installing requirements...")
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("   ❌ requirements.txt not found")
        return False
    
    try:
        pip_cmd = get_pip_command()
        subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], 
                      check=True)
        print("   ✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install requirements: {e}")
        return False

def check_model_file():
    """Check if the model file exists"""
    print("🤖 Checking model file...")
    model_path = Path("models/best_model.pth")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model file found: {size_mb:.1f}MB")
        return True
    else:
        print("   ⚠️  Model file not found: models/best_model.pth")
        print("   📥 Run: python download_model.py")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "models",
        "static/uploads",
        "static/css", 
        "static/js",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("   ✅ Directories created/verified")

def run_tests():
    """Run basic application tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        pip_cmd = get_pip_command()
        test_script = '''
import flask
import torch
import torchvision
import PIL
print("✅ All imports successful")
'''
        
        result = subprocess.run(
            pip_cmd[:-2] + ["-c", test_script],
            capture_output=True,
            text=True,
            check=True
        )
        print("   ✅ Dependencies test passed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Dependencies test failed: {e.stderr}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print()
    print("🎉 Setup Complete!")
    print("=" * 20)
    print()
    print("Next steps:")
    print("1. Activate virtual environment:")
    
    system = platform.system()
    if system == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print()
    print("2. Download the model (if not already done):")
    print("   python download_model.py")
    print()
    print("3. Start the application:")
    print("   python app.py")
    print()
    print("4. Open in browser:")
    print("   http://localhost:5001")
    print()
    print("🌿 Happy plant disease detection! 🌿")

def main():
    """Main setup function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing requirements", install_requirements),
        ("Checking model file", check_model_file),
        ("Running tests", run_tests),
    ]
    
    success = True
    for step_name, step_func in steps:
        if not step_func():
            success = False
            break
    
    if success:
        print_next_steps()
    else:
        print()
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
