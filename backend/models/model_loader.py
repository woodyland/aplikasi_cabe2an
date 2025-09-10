# models/model_loader.py - update function
def load_pepper_model(model_path: str, num_classes: int = 9):
    """Load model with enhanced error handling"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Add safe globals
        import torchvision.models.mobilenetv3
        torch.serialization.add_safe_globals([torchvision.models.mobilenetv3.MobileNetV3])
        
        # Try loading complete model first
        print(f"Attempting to load model from: {model_path}")
        loaded_model = torch.load(model_path, map_location=device, weights_only=False)
        
        # If it's already a complete model
        if hasattr(loaded_model, 'eval'):
            model = loaded_model
            print("Loaded complete model directly")
        else:
            # Create architecture and load state dict
            model = models.mobilenet_v3_large(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(960, 1280),
                nn.Hardswish(),
                nn.Dropout(0.3),
                nn.Linear(1280, num_classes)
            )
            
            if isinstance(loaded_model, dict):
                model.load_state_dict(loaded_model)
            else:
                raise ValueError(f"Unexpected model format: {type(loaded_model)}")
        
        model.eval()
        model.to(device)
        print(f"✅ Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")