import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model (pretrained dummy)
import torch
import torchvision.models as models
import streamlit as st

@st.cache_resource
def get_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(1280, 4)
    model.eval()
    return model


model = get_model()
classes = ['No Acne', 'Mild', 'Moderate', 'Severe']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Acne Detector")

# CAMERA + UPLOAD
col1, col2 = st.columns(2)
with col1:
    camera_img = st.camera_input("Take Photo")
with col2:
    uploaded = st.file_uploader("Or Upload", type=["jpg", "png", "jpeg"])

img = camera_img if camera_img else uploaded

if img:
    image = Image.open(img).convert("RGB")
    st.image(image, caption="Your Face", use_column_width=True)
    
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    
    st.success(f"**{classes[pred]}**")
    if pred == 0:
        st.info("Keep up the good skincare!")
    else:
        st.info("Use gentle cleanser 2x daily.")
