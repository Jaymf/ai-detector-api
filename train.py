import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json

# 📸 transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 📁 dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 🧠 modelo base
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ⚙️ loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔥 entrenamiento
epochs = 3

for epoch in range(epochs):
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# 💾 guardar modelo
torch.save(model.state_dict(), "ai_detector.pth")

# 💾 guardar clases (ESTO ARREGLA TODO)
with open("classes.json", "w") as f:
    json.dump(dataset.class_to_idx, f)

print("✅ Modelo + clases guardados correctamente")