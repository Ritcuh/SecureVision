import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import cv2
from PIL import Image

class VisionModel(nn.Module):
    """Enhanced CNN model for web application security analysis."""
    
    def __init__(self, num_classes=3):
        super(VisionModel, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)  # safe, warning, risky
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)
    
    def extract_features(self, x):
        """Extract feature representations from the model."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)

def build_vision_model(pretrained_path: Optional[str] = None) -> VisionModel:
    """
    Build and optionally load a pretrained vision model.
    
    Args:
        pretrained_path: Path to pretrained model weights
    
    Returns:
        VisionModel instance
    """
    model = VisionModel()
    
    if pretrained_path:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"Loaded pretrained model from {pretrained_path}")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Using randomly initialized model")
    
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Preprocessed image tensor
    """
    # Ensure image is in RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB if needed
        if image.max() > 1.0:
            image = image / 255.0
    
    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize (using ImageNet stats as baseline)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor

def analyze_screenshot(model: VisionModel, image: np.ndarray) -> Dict:
    """
    Analyze screenshot for security vulnerabilities.
    
    Args:
        model: Trained vision model
        image: Screenshot image as numpy array
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            predictions = model(image_tensor)
            probabilities = predictions[0].numpy()
        
        # Map predictions to labels
        labels = ['safe', 'warning', 'risky']
        predicted_idx = np.argmax(probabilities)
        predicted_label = labels[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Analyze visual elements
        visual_analysis = analyze_visual_elements(image)
        
        # Combine model prediction with visual analysis
        risk_factors = detect_ui_risk_factors(image)
        
        # Adjust confidence based on visual analysis
        if risk_factors:
            confidence = min(confidence + len(risk_factors) * 0.1, 1.0)
            if predicted_label == 'safe' and len(risk_factors) > 2:
                predicted_label = 'warning'
        
        return {
            'risk_level': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'safe': float(probabilities[0]),
                'warning': float(probabilities[1]),
                'risky': float(probabilities[2])
            },
            'visual_analysis': visual_analysis,
            'risk_factors': risk_factors,
            'ui_elements': detect_ui_elements(image)
        }
    
    except Exception as e:
        print(f"Error in screenshot analysis: {e}")
        return {
            'risk_level': 'warning',
            'confidence': 0.5,
            'probabilities': {'safe': 0.33, 'warning': 0.34, 'risky': 0.33},
            'visual_analysis': {},
            'risk_factors': [],
            'ui_elements': {},
            'error': str(e)
        }

def analyze_visual_elements(image: np.ndarray) -> Dict:
    """
    Analyze visual elements in the screenshot.
    
    Args:
        image: Screenshot image
    
    Returns:
        Dictionary with visual analysis results
    """
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges (potential UI boundaries)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Analyze color distribution
        if len(image.shape) == 3:
            color_variance = np.var(image, axis=(0, 1))
            dominant_colors = analyze_dominant_colors(image)
        else:
            color_variance = np.array([0, 0, 0])
            dominant_colors = []
        
        # Detect text regions (approximation)
        text_regions = detect_text_regions(gray)
        
        # Calculate layout complexity
        layout_complexity = calculate_layout_complexity(edges)
        
        return {
            'edge_density': float(edge_density),
            'color_variance': color_variance.tolist(),
            'dominant_colors': dominant_colors,
            'text_regions': len(text_regions),
            'layout_complexity': layout_complexity,
            'image_size': image.shape[:2]
        }
    
    except Exception as e:
        return {'error': str(e)}

def detect_ui_risk_factors(image: np.ndarray) -> list:
    """
    Detect potential security risk factors in UI elements.
    
    Args:
        image: Screenshot image
    
    Returns:
        List of detected risk factors
    """
    risk_factors = []
    
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect form-like structures (rectangles that might be input fields)
        contours, _ = cv2.findContours(
            cv2.Canny(gray, 50, 150), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        input_fields = 0
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular and of reasonable size
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                if 50 < w < 300 and 20 < h < 50:  # Typical input field dimensions
                    input_fields += 1
        
        if input_fields > 5:
            risk_factors.append({
                'type': 'multiple_input_fields',
                'description': 'Multiple input fields detected - potential for injection attacks',
                'severity': 'medium'
            })
        
        # Check for potential password fields (dark rectangles)
        # This is a heuristic approach
        dark_regions = np.sum(gray < 50)
        if dark_regions > (gray.size * 0.1):
            risk_factors.append({
                'type': 'dark_input_regions',
                'description': 'Dark input regions detected - potential password fields',
                'severity': 'low'
            })
        
        # Detect potential alert/popup patterns
        bright_regions = np.sum(gray > 200)
        if bright_regions > (gray.size * 0.3):
            risk_factors.append({
                'type': 'bright_overlay',
                'description': 'Bright overlay detected - potential popup or alert',
                'severity': 'medium'
            })
        
        # Check for red color dominance (often indicates errors/warnings)
        if len(image.shape) == 3:
            red_channel = image[:, :, 0]
            red_dominance = np.mean(red_channel) / (np.mean(image) + 1e-6)
            if red_dominance > 1.5:
                risk_factors.append({
                    'type': 'red_warning_indicators',
                    'description': 'Red warning indicators detected',
                    'severity': 'low'
                })
    
    except Exception as e:
        risk_factors.append({
            'type': 'analysis_error',
            'description': f'Error in risk factor detection: {str(e)}',
            'severity': 'low'
        })
    
    return risk_factors

def detect_ui_elements(image: np.ndarray) -> Dict:
    """
    Detect common UI elements in the screenshot.
    
    Args:
        image: Screenshot image
    
    Returns:
        Dictionary with detected UI elements
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect button-like elements (rounded rectangles)
        contours, _ = cv2.findContours(
            cv2.Canny(gray, 50, 150), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        buttons = 0
        input_fields = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Button detection heuristics
            if 2 < aspect_ratio < 6 and 30 < w < 200 and 20 < h < 60:
                buttons += 1
            
            # Input field detection heuristics
            elif 3 < aspect_ratio < 10 and 100 < w < 400 and 20 < h < 50:
                input_fields += 1
        
        return {
            'buttons': buttons,
            'input_fields': input_fields,
            'total_elements': len(contours),
            'form_like_structure': input_fields > 2 and buttons > 0
        }
    
    except Exception as e:
        return {'error': str(e)}

def analyze_dominant_colors(image: np.ndarray, k: int = 5) -> list:
    """
    Analyze dominant colors in the image using K-means clustering.
    
    Args:
        image: Input image
        k: Number of dominant colors to find
    
    Returns:
        List of dominant colors as RGB tuples
    """
    try:
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use a simple approximation instead of full K-means for efficiency
        # Sample a subset of pixels
        sample_size = min(1000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[indices]
        
        # Find unique colors and their frequencies
        unique_colors, counts = np.unique(sample_pixels, axis=0, return_counts=True)
        
        # Sort by frequency and return top k
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = unique_colors[sorted_indices[:k]]
        
        return [tuple(map(int, color)) for color in dominant_colors]
    
    except Exception:
        return []

def detect_text_regions(gray_image: np.ndarray) -> list:
    """
    Detect potential text regions in the image.
    
    Args:
        gray_image: Grayscale image
    
    Returns:
        List of text region bounding boxes
    """
    try:
        # Apply morphological operations to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            cv2.Canny(morph, 50, 150), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Text region heuristics
            if 2 < aspect_ratio < 20 and w > 30 and 10 < h < 50:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    except Exception:
        return []

def calculate_layout_complexity(edge_image: np.ndarray) -> float:
    """
    Calculate layout complexity based on edge distribution.
    
    Args:
        edge_image: Edge-detected image
    
    Returns:
        Complexity score
    """
    try:
        # Find contours in edge image
        contours, _ = cv2.findContours(
            edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Calculate complexity based on number and size variance of contours
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return 0.0
        
        area_variance = np.var(areas) if len(areas) > 1 else 0
        contour_density = len(contours) / edge_image.size
        
        complexity = (contour_density * 1000) + (area_variance / 10000)
        return min(complexity, 1.0)  # Normalize to 0-1 range
    
    except Exception:
        return 0.5