import os
import requests

def download_model_if_not_exists():
    model_path = "models/pepper_model.pth"
    if not os.path.exists(model_path):
        print("Model not found locally. Add model via Railway volume or external storage.")
        # Implementasi download dari external source
        return False
    return True