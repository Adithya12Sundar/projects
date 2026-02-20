#cifar 10 model over api building and accessing project for torch saved model
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from model import Net
# Load model
model = Net()
model.load_state_dict(torch.load("cifar.pth", map_location="cpu"))
model.eval()

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("L").resize((28,28))
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1).item()

    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)        
