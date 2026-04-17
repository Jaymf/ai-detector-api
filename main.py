from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import torch.nn as nn
from torchvision import models
import json
import os

app = FastAPI()

# 🔓 CORS (IMPORTANTE PARA FLUTTER WEB / MOBILE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 rutas seguras
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "ai_detector.pth")
classes_path = os.path.join(BASE_DIR, "classes.json")

# 🧠 cargar modelo
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 2)

model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)
model.eval()

# 📁 cargar clases
with open(classes_path, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# 📸 transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_idx = int(torch.argmax(probs))
    label = idx_to_class[predicted_idx]

    return {
        "ai_probability": float(probs[class_to_idx["ai"]]),
        "real_probability": float(probs[class_to_idx["real"]]),
        "label": label
    }