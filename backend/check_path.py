# check_path.py
import os

print("Current directory:", os.getcwd())
print("Model file exists:", os.path.exists("models/pepper_model.pth"))
print("Model file size:", os.path.getsize("models/pepper_model.pth") if os.path.exists("models/pepper_model.pth") else "File not found")