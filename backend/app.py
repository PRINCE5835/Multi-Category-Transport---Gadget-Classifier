from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import json
import os

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = json.load(open("../model/class_names.json"))
n_classes = len(class_names)

my_model = models.mobilenet_v2(weights=None)
feat_count = my_model.classifier[1].in_features
my_model.classifier[1] = nn.Linear(feat_count, n_classes)
my_model.load_state_dict(torch.load("../model/classifier.pth", map_location=device, weights_only=True))
my_model = my_model.to(device)
my_model.eval()

my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    f = request.files["file"]
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    tensor_img = my_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = my_model(tensor_img)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, idx = torch.max(probs, 0)
    label = class_names[idx.item()]
    conf = round(conf.item() * 100, 2)
    return jsonify({"label": label, "confidence": conf})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
