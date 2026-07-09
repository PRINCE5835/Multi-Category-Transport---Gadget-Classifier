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
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": ["GET", "POST", "OPTIONS"]}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

device = torch.device("cpu")
base_dir = os.path.dirname(os.path.abspath(__file__))
class_names = json.load(open(os.path.join(base_dir, "class_names.json")))

my_model = None
my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    global my_model
    if my_model is not None:
        return
    m = models.mobilenet_v2(weights=None)
    feat_count = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(feat_count, len(class_names))
    state = torch.load(os.path.join(base_dir, "classifier.pth"), map_location=device, weights_only=True)
    m.load_state_dict(state)
    m = m.to(device)
    m.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        m(dummy)
    my_model = m

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    try:
        load_model()
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded": my_model is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
