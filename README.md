Deforestation & Wildfire Detection using Satellite Imagery
This project is a Flask-based web application that predicts deforestation and wildfire incidents from satellite images using deep learning models and integrates with Google Earth Engine for real-time satellite imagery.

✨ Key Features
🛰️ Satellite Image Integration: Fetches Sentinel-2 and FIRMS data using Google Earth Engine.

🌲 Deforestation Detection: Uses a fine-tuned MobileNetV2 model.

🔥 Wildfire Detection: Uses a fine-tuned ResNet18 model.

🖼️ Sample Satellite Image Viewer: Visualizes combined deforestation and wildfire data.

🌐 Flask Web Interface: Upload images and get predictions instantly with confidence scores.

🛠️ Tech Stack
Framework: Flask (Python 3.9+)

Deep Learning: PyTorch, MobileNetV2, ResNet18

Satellite Data: Google Earth Engine (Sentinel-2 & FIRMS)

Image Processing: Pillow (PIL), Torchvision

Frontend: HTML/CSS (Jinja2 templates)

📂 Project Structure
csharp
Copy
Edit
Deforestation-Wildfire-Detection/
├── app.py                     # Main Flask app
├── templates/
│   └── index.html             # Web UI
├── static/                    # (Optional) CSS/JS/Images
├── deforestation_model.pth    # Trained MobileNetV2 model
├── wildfire_model.pth         # Trained ResNet18 model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
⚡ Installation & Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/deforestation-wildfire-detection.git
cd deforestation-wildfire-detection
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv environment
source environment/bin/activate   # For Linux/Mac
environment\Scripts\activate      # For Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download Models From Github Using python scripts
Setup Google Earth Engine:

Create a Google Cloud project and enable the Earth Engine API.

Authenticate using:

bash
Copy
Edit
earthengine authenticate
Update your project name in app.py:

python
Copy
Edit
ee.Initialize(project='your-project-id')
Add Trained Models:

Place your trained deforestation_model.pth and wildfire_model.pth in the project root.

▶️ Running the App
bash
Copy
Edit
python app.py
Go to http://127.0.0.1:5000/ in your browser.

Upload satellite images to detect deforestation or wildfire.

Click on "Get Satellite Image" to view a sample area from Earth Engine.

🌍 Example API Routes
Home (UI):

sql
Copy
Edit
GET /
Deforestation Prediction:

css
Copy
Edit
POST /predict_deforestation
form-data: { file: <image_file> }
Wildfire Prediction:

css
Copy
Edit
POST /predict_wildfire
form-data: { file: <image_file> }
Get Satellite Image:

bash
Copy
Edit
POST /get_satellite_image
🚀 Future Enhancements
Add real-time map visualization (Leaflet.js).

Train models with more datasets for higher accuracy.

Integrate cloud-based storage for image history.

Expand detection to include floods, landslides, and urbanization.




📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Moorthy M – B.Tech AI & Data Science










