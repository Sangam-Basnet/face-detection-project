import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# Dummy model (no .pth needed)
@st.cache_resource
def get_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.classifier[1] = torch.nn.Linear(1280, 4)
    model.eval()
    return model

model = get_model()
classes = ['no_acne', 'mild', 'moderate', 'severe']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406])
])

st.title("Acne Detector")
uploaded = st.file_uploader("Upload face photo", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input")
    
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    
    st.success(f"**{classes[pred].upper()} ACNE**")
    st.info("Tip: Wash face 2x daily.")
