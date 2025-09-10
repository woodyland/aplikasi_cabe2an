# test_model.py
import torch
import torchvision.models.mobilenetv3

# Add safe globals
torch.serialization.add_safe_globals([torchvision.models.mobilenetv3.MobileNetV3])

try:
    print("Loading model...")
    model = torch.load("models/pepper_model.pth", map_location='cpu', weights_only=False)
    print(f"Model type: {type(model)}")
    
    if hasattr(model, 'state_dict'):
        print("Model has state_dict")
    
    if isinstance(model, dict):
        print(f"Dict keys: {model.keys()}")
    
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")