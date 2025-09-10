# simple_main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io
import json

# Simple global model
model = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    global model
    try:
        # Add safe globals first
        import torchvision.models.mobilenetv3
        torch.serialization.add_safe_globals([torchvision.models.mobilenetv3.MobileNetV3])
        
        # Load model directly 
        model = torch.load("models/pepper_model.pth", map_location='cpu', weights_only=False)
        model.eval()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

# Load model on startup
model_loaded = load_model()

@app.get("/")
def root():
    return {"message": "Pepper API Running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": 9
    }

@app.post("/predict") 
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Simple prediction
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Basic preprocessing
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1).max().item()
    
    classes = ["bell_pepper", "cabe_keriting", "cherry_pepper", "chiltepin", 
               "hungarian_wax", "jalapeno", "marconi", "pequin", "thai_chili"]
    
    return {
        "prediction": classes[predicted.item()],
        "confidence": float(confidence * 100)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)