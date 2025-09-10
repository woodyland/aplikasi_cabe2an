# backend/utils/prediction.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

def predict_pepper_class(model: torch.nn.Module, image_tensor: torch.Tensor, 
                        class_names: List[str]) -> Dict:
    """
    Predict pepper class from preprocessed image tensor
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
    
    Returns:
        Dictionary with prediction results
    """
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction and confidence
        confidence, predicted = torch.max(probabilities, 1)
        
        # Convert to numpy for easier handling
        prob_array = probabilities.cpu().numpy()[0]
        predicted_class = predicted.cpu().item()
        confidence_score = confidence.cpu().item()
        
        # Create probability dictionary
        class_probabilities = {
            class_names[i]: float(prob_array[i]) * 100 
            for i in range(len(class_names))
        }
        
        # Sort probabilities in descending order
        sorted_probabilities = dict(
            sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "class": class_names[predicted_class],
            "confidence": float(confidence_score * 100),
            "class_index": predicted_class,
            "probabilities": sorted_probabilities,
            "raw_outputs": outputs.cpu().numpy()[0].tolist()
        }

def get_top_predictions(model: torch.nn.Module, image_tensor: torch.Tensor, 
                       class_names: List[str], top_k: int = 3) -> List[Dict]:
    """
    Get top-k predictions with confidence scores
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        List of dictionaries with top predictions
    """
    
    prediction_result = predict_pepper_class(model, image_tensor, class_names)
    probabilities = prediction_result["probabilities"]
    
    # Get top-k predictions
    top_predictions = []
    for i, (class_name, prob) in enumerate(list(probabilities.items())[:top_k]):
        top_predictions.append({
            "rank": i + 1,
            "class": class_name,
            "confidence": prob,
            "is_predicted": i == 0
        })
    
    return top_predictions

def calculate_prediction_confidence_level(confidence: float) -> str:
    """
    Categorize confidence level for user display
    
    Args:
        confidence: Confidence score (0-100)
    
    Returns:
        Confidence level string
    """
    
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Medium"
    elif confidence >= 40:
        return "Low"
    else:
        return "Very Low"

def get_prediction_advice(confidence: float, top_predictions: List[Dict]) -> str:
    """
    Generate advice based on prediction confidence
    
    Args:
        confidence: Main prediction confidence
        top_predictions: List of top predictions
    
    Returns:
        Advice string for user
    """
    
    if confidence >= 85:
        return "High confidence prediction. This appears to be a reliable classification."
    
    elif confidence >= 70:
        return "Good confidence prediction. Consider the image quality and lighting for best results."
    
    elif confidence >= 50:
        # Check if second prediction is close
        if len(top_predictions) > 1:
            second_confidence = top_predictions[1]["confidence"]
            if abs(confidence - second_confidence) < 15:
                return f"Moderate confidence. The model also considers this might be {top_predictions[1]['class']}. Try a clearer image or different angle."
        
        return "Moderate confidence prediction. Consider taking another photo with better lighting or a different angle."
    
    else:
        return "Low confidence prediction. Please ensure the pepper is clearly visible, well-lit, and takes up most of the image frame."

def batch_predict(model: torch.nn.Module, image_tensors: List[torch.Tensor], 
                 class_names: List[str]) -> List[Dict]:
    """
    Predict multiple images in batch for efficiency
    
    Args:
        model: Trained PyTorch model
        image_tensors: List of preprocessed image tensors
        class_names: List of class names
    
    Returns:
        List of prediction results
    """
    
    if not image_tensors:
        return []
    
    device = next(model.parameters()).device
    
    # Stack tensors into batch
    batch_tensor = torch.cat(image_tensors, dim=0).to(device)
    
    with torch.no_grad():
        # Forward pass for entire batch
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Process each prediction in batch
        results = []
        for i in range(len(image_tensors)):
            prob_array = probabilities[i].cpu().numpy()
            predicted_class = np.argmax(prob_array)
            confidence = prob_array[predicted_class]
            
            class_probabilities = {
                class_names[j]: float(prob_array[j]) * 100 
                for j in range(len(class_names))
            }
            
            sorted_probabilities = dict(
                sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            
            results.append({
                "class": class_names[predicted_class],
                "confidence": float(confidence * 100),
                "class_index": predicted_class,
                "probabilities": sorted_probabilities
            })
    
    return results