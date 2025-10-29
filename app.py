import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_mobilenet
from utils import SUGGESTIONS
import json

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = get_mobilenet(num_classes=4).to(device)
    ckpt = torch.load("face_pimple_detector_final.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with open("label_map.json", "r") as f:
        id2label = json.load(f)
    return model, id2label

model, id2label = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🧴 AI Acne Detector by Sangam")
st.write("Upload a face photo for instant acne grade + safe tips!")

img_file = st.camera_input("📸 Take a live photo")
uploaded = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
elif uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
else:
    st.stop()

st.image(img, caption="Your Photo", use_column_width=True)

with torch.no_grad():
    tensor = transform(img).unsqueeze(0)
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_id = output.argmax().item()
    confidence = probs[pred_id].item()

label = id2label[str(pred_id)]
st.success(f"**Result: {label.upper()} Acne** ({confidence:.1%} confidence)")

st.info(f"💡 **Safe Suggestions:**\n{SUGGESTIONS[label]}")

with st.sidebar:
    st.write("Built with PyTorch + MobileNetV2")
    st.write("Accuracy: ~82%")
    st.markdown("[GitHub](https://github.com/Sangam-Basnet/face-detection-project)")
