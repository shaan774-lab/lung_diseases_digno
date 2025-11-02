import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to predict Pneumonia")

# Load Model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_model_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

CLASSES = ["NORMAL", "PNEUMONIA"]

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = CLASSES[predicted.item()]

    st.subheader(f"Prediction: **{label}**")
