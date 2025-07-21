# 🌿 PlantDiagnose - AI Plant Disease Detection

A sophisticated web application for plant disease detection using deep learning and computer vision. Built with Flask and powered by EfficientNet B4 neural network.

![PlantDiagnose Banner](https://img.shields.io/badge/AI-Plant%20Disease%20Detection-green?style=for-the-badge&logo=leaf)

## ✨ Features

- 🤖 **AI-Powered Analysis**: Advanced EfficientNet B4 neural network
- 📷 **Multiple Input Methods**: File upload + live camera capture  
- 🎯 **High Accuracy**: 99.45% accuracy across 39 disease classes
- 🌱 **15+ Plant Species**: Apple, Tomato, Corn, Grape, Potato, and more
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- ⚡ **Real-time Results**: Analysis completed in under 2 seconds
- 🎨 **Modern UI**: Glassmorphism design with smooth animations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model**
   
   ⚠️ **Important**: The trained model file (`best_model.pth`) is not included in this repository due to its large size (~224MB).
   
   **Option 1: Train your own model**
   - Use the provided model architecture in `simple_model_loader.py`
   - Train on plant disease datasets (PlantVillage, etc.)
   - Save as `models/best_model.pth`
   
   **Option 2: Download pre-trained model**
   - Contact the repository owner for the trained model
   - Or check the releases page for model downloads
   - Place the `best_model.pth` file in the `models/` directory

5. **Verify project structure**
   ```
   plant_disease_app/
   ├── app.py                    # Main Flask application
   ├── simple_model_loader.py    # Model loading utilities
   ├── requirements.txt          # Dependencies
   ├── models/
   │   ├── best_model.pth       # Trained model (download required)
   │   ├── class_names.json     # Disease class names
   │   └── class_distribution.csv # Training statistics
   ├── static/
   │   ├── css/style.css        # Custom styles
   │   ├── js/main.js          # JavaScript utilities
   │   └── uploads/            # Uploaded images (auto-created)
   └── templates/              # HTML templates
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open in browser**
   ```
   http://localhost:5001
   ```

## 📱 Usage

### Upload Image Analysis
1. Navigate to the Upload page
2. Drag & drop or select a plant image
3. Click "Analyze Plant" 
4. View detailed results with confidence scores

### Live Camera Analysis  
1. Navigate to the Camera page
2. Grant camera permissions
3. Position plant within the focus frame
4. Capture and analyze instantly

### Supported Plants & Diseases

| Plant Species | Supported Diseases |
|---------------|-------------------|
| **Apple** | Scab, Black rot, Cedar rust, Healthy |
| **Tomato** | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Mosaic virus, Yellow Leaf Curl Virus, Healthy |
| **Corn** | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| **Grape** | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| **Potato** | Early blight, Late blight, Healthy |
| **Pepper** | Bacterial spot, Healthy |
| **Others** | Orange, Peach, Raspberry, Soybean, Squash, Strawberry, Cherry, Blueberry |

## 🏗️ Architecture

### Model Details
- **Architecture**: EfficientNet B4 + Custom Classifier
- **Framework**: PyTorch + timm
- **Input Size**: 224x224 RGB images
- **Classes**: 39 disease categories
- **Accuracy**: 99.45% on validation set

### Web Framework
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Styling**: Modern glassmorphism design
- **Icons**: Font Awesome 6

### Key Components
```
app.py                  # Main Flask application with all routes
simple_model_loader.py  # Model loading and inference utilities  
templates/              # Jinja2 HTML templates
static/css/             # Custom CSS with modern styling
static/js/              # JavaScript for interactions
models/                 # Model files and metadata
```

## 🛠️ Development

### Project Structure
```bash
# Core files (required)
app.py                    # Main Flask app
simple_model_loader.py    # Model utilities
requirements.txt          # Python dependencies

# Model files (download required)
models/best_model.pth     # Trained model (~224MB)
models/class_names.json   # Class definitions
models/class_distribution.csv # Training stats

# Frontend
templates/                # HTML templates  
static/css/style.css     # Custom styles
static/js/main.js        # JavaScript

# Auto-generated
static/uploads/          # User uploaded images
__pycache__/            # Python cache files
```

### Adding New Features

1. **New Disease Classes**
   - Update `models/class_names.json`
   - Retrain model with new data
   - Update disease info in `app.py`

2. **UI Modifications**
   - Edit templates in `templates/`
   - Modify styles in `static/css/style.css`
   - Add JavaScript in `static/js/main.js`

3. **New Routes**
   - Add routes in `app.py`
   - Create corresponding templates
   - Update navigation in `base.html`

## 🚀 Deployment

### Local Development
```bash
python app.py
# Runs on http://localhost:5001
```

### Production Deployment

**Using Gunicorn:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Using Docker:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**Environment Variables:**
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export MAX_CONTENT_LENGTH=16777216  # 16MB
```

## 📊 Performance

- **Accuracy**: 99.45% on validation dataset
- **Inference Time**: < 2 seconds per image
- **Model Size**: ~224MB (EfficientNet B4)
- **Supported Formats**: JPG, PNG, GIF, BMP, WEBP
- **Max Image Size**: 16MB

## 🔧 Configuration

### Model Configuration
Edit `simple_model_loader.py` to modify:
- Model architecture
- Input preprocessing
- Class mappings
- Inference parameters

### App Configuration  
Edit `app.py` to modify:
- Upload limits
- Supported file types
- Disease information
- Treatment recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is designed for educational and research purposes. While our model achieves high accuracy, results should be used as a guide only. For critical plant health decisions, commercial applications, or when in doubt, please consult with qualified agricultural professionals, plant pathologists, or local extension services.

## 🙏 Acknowledgments

- Plant disease datasets from agricultural research institutions
- PyTorch and timm communities for model architectures
- Bootstrap and Font Awesome for UI components
- Flask community for web framework

## 📞 Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [Link to docs]

---

**Made with ❤️ for sustainable agriculture and plant health**
