# backend/utils/image_processing.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for pepper classification model
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed tensor ready for model inference
    """
    
    # Define preprocessing transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms and add batch dimension
    processed_image = transform(image).unsqueeze(0)
    
    return processed_image

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better prediction accuracy
    
    Args:
        image: PIL Image object
    
    Returns:
        Enhanced PIL Image
    """
    
    # Convert to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply image enhancements
    # 1. Noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
    
    # 2. Contrast enhancement using CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpening
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9, -1], 
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to PIL
    final_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    return final_image

def crop_pepper_region(image: Image.Image, confidence_threshold: float = 0.5) -> Image.Image:
    """
    Automatically crop the pepper region from the image
    (Simple center cropping for now, can be enhanced with object detection)
    
    Args:
        image: PIL Image object
        confidence_threshold: Threshold for automatic cropping
    
    Returns:
        Cropped PIL Image focused on the pepper
    """
    
    width, height = image.size
    
    # Simple center crop (can be enhanced with YOLO/RCNN for automatic pepper detection)
    crop_size = min(width, height) * 0.8  # Take 80% of the smaller dimension
    
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    cropped_image = image.crop((int(left), int(top), int(right), int(bottom)))
    
    return cropped_image

def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validate if the image is suitable for pepper classification
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    # Check image size
    width, height = image.size
    if width < 50 or height < 50:
        return False, "Image too small (minimum 50x50 pixels)"
    
    if width > 4000 or height > 4000:
        return False, "Image too large (maximum 4000x4000 pixels)"
    
    # Check image format
    if image.mode not in ['RGB', 'RGBA']:
        return False, "Image must be in RGB or RGBA format"
    
    # Check for completely black or white images
    np_image = np.array(image.convert('RGB'))
    if np.mean(np_image) < 10:
        return False, "Image appears to be too dark"
    
    if np.mean(np_image) > 245:
        return False, "Image appears to be too bright"
    
    # Check for sufficient contrast
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    if np.std(gray) < 20:
        return False, "Image lacks sufficient contrast"
    
    return True, "Image is valid"

def resize_for_web(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize image for web display while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension size
    
    Returns:
        Resized PIL Image
    """
    
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image