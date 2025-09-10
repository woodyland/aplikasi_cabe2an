# backend/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import base64
import io
from PIL import Image
import torch
import uvicorn
from typing import List
import asyncio

from models.model_loader import load_pepper_model
from utils.image_processing import preprocess_image
from utils.prediction import predict_pepper_class

# Initialize FastAPI app
app = FastAPI(
    title="Pepper Classification API",
    description="Real-time pepper genus classification using deep learning",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
pepper_model = None
class_names = [
    "bell_pepper_dataset", "cabe_keriting", "cherry_pepper",
    "chiltepin", "hungarian_wax", "jalapeno_dataset", 
    "marconi", "pequin", "thai_chili"
]

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Pepper Classification API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pepper_model is not None,
        "classes": len(class_names)
    }

@app.post("/predict")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """Predict pepper class from uploaded image"""
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess for model
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction_result = predict_pepper_class(pepper_model, processed_image, class_names)
        
        return {
            "success": True,
            "prediction": prediction_result["class"],
            "confidence": prediction_result["confidence"],
            "all_probabilities": prediction_result["probabilities"]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time camera predictions"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive image data from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                # Decode base64 image
                image_data = base64.b64decode(message["image"].split(",")[1])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Preprocess and predict
                processed_image = preprocess_image(image)
                prediction_result = predict_pepper_class(pepper_model, processed_image, class_names)
                
                # Send result back
                response = {
                    "type": "prediction",
                    "success": True,
                    "prediction": prediction_result["class"],
                    "confidence": prediction_result["confidence"],
                    "all_probabilities": prediction_result["probabilities"],
                    "timestamp": message.get("timestamp")
                }
                
                await manager.send_personal_message(json.dumps(response), websocket)
            
            elif message["type"] == "capture":
                # Handle high-quality capture
                image_data = base64.b64decode(message["image"].split(",")[1])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Multiple predictions for stability
                predictions = []
                for _ in range(3):  # 3 predictions for averaging
                    processed_image = preprocess_image(image)
                    result = predict_pepper_class(pepper_model, processed_image, class_names)
                    predictions.append(result)
                
                # Average predictions
                avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)
                most_common_class = max(set(p["class"] for p in predictions), 
                                      key=lambda x: sum(1 for p in predictions if p["class"] == x))
                
                response = {
                    "type": "capture_result",
                    "success": True,
                    "prediction": most_common_class,
                    "confidence": avg_confidence,
                    "individual_predictions": predictions,
                    "timestamp": message.get("timestamp")
                }
                
                await manager.send_personal_message(json.dumps(response), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from WebSocket")
    except Exception as e:
        error_response = {
            "type": "error",
            "success": False,
            "error": str(e)
        }
        await manager.send_personal_message(json.dumps(error_response), websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )