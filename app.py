import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Define the same ResNet as training
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False) 
# If you changed the final layer to match 2 classes:
model.fc = nn.Linear(512, 2)

# 2. Load your saved weights
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()

# 3. Transforms must match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def predict_image(image):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()
    
    # Suppose class 0 = cat, class 1 = dog
    label = "cat" if pred_idx == 0 else "dog"
    return label, confidence

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Cat/Dog Classifier!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
    
    try:
        image = Image.open(file.stream).convert("RGB")
        label, conf = predict_image(image)
        logger.info(f"Predicted: {label} (confidence: {conf:.2f})")
        
        return jsonify({"label": label, "confidence": round(conf, 4)})
    except Exception as e:
        logger.exception("Error processing image.")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
