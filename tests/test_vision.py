import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_model import (
    VisionModel, build_vision_model, preprocess_image,
    analyze_screenshot, detect_ui_risk_factors, analyze_visual_elements
)

class TestVisionModel:
    """Test cases for vision model functionality."""
    
    def test_model_initialization(self):
        """Test vision model initialization."""
        model = VisionModel()
        
        # Check model structure
        assert hasattr(model, 'conv1'), "Model should have conv1 layer"
        assert hasattr(model, 'conv2'), "Model should have conv2 layer"
        assert hasattr(model, 'conv3'), "Model should have conv3 layer"
        assert hasattr(model, 'fc1'), "Model should have fc1 layer"
        assert hasattr(model, 'fc2'), "Model should have fc2 layer"
        assert hasattr(model, 'fc3'), "Model should have fc3 layer"
        
        # Check output dimensions
        assert model.fc3.out_features == 3, "Output should have 3 classes"
    
    def test_model_forward_pass(self):
        """Test forward pass through the model."""
        model = VisionModel()
        model.eval()
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 3), f"Expected shape (2, 3), got {output.shape}"
        
        # Check output is probability distribution
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size)), "Output should sum to 1"
        assert torch.all(output >= 0), "All probabilities should be non-negative"
        assert torch.all(output <= 1), "All probabilities should be <= 1"
    
    def test_feature_extraction(self):
        """Test feature extraction capability."""
        model = VisionModel()
        model.eval()
        
        input_tensor = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            features = model.extract_features(input_tensor)
        
        # Features should be 1D for classification
        assert len(features.shape) == 2, "Features should be 2D (batch_size, features)"
        assert features.shape[0] == 1, "Batch size should be 1"
        assert features.shape[1] > 0, "Should have feature dimensions"
    
    def test_build_vision_model(self):
        """Test model building function."""
        model = build_vision_model()
        
        assert isinstance(model, VisionModel), "Should return VisionModel instance"
        assert model.training == False, "Model should be in eval mode"
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Test with different input formats
        
        # RGB image (0-255)
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(rgb_image)
        
        assert isinstance(processed, torch.Tensor), "Should return torch tensor"
        assert processed.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {processed.shape}"
        
        # Normalized image (0-1)
        norm_image = np.random.rand(150, 150, 3).astype(np.float32)
        processed = preprocess_image(norm_image)
        
        assert isinstance(processed, torch.Tensor), "Should return torch tensor"
        assert processed.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {processed.shape}"
    
    def test_analyze_screenshot_basic(self):
        """Test basic screenshot analysis."""
        model = build_vision_model()
        
        # Create dummy screenshot
        screenshot = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
        
        result = analyze_screenshot(model, screenshot)
        
        # Check result structure
        assert isinstance(result, dict), "Should return dictionary"
        assert 'risk_level' in result, "Should have risk_level"
        assert 'confidence' in result, "Should have confidence"
        assert 'probabilities' in result, "Should have probabilities"
        assert 'visual_analysis' in result, "Should have visual_analysis"
        assert 'risk_factors' in result, "Should have risk_factors"
        assert 'ui_elements' in result, "Should have ui_elements"
        
        # Check data types and values
        assert result['risk_level'] in ['safe', 'warning', 'risky'], "Invalid risk level"
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence out of range"
        assert isinstance(result['probabilities'], dict), "Probabilities should be dict"
        assert isinstance(result['visual_analysis'], dict), "Visual analysis should be dict"
        assert isinstance(result['risk_factors'], list), "Risk factors should be list"
        assert isinstance(result['ui_elements'], dict), "UI elements should be dict"
    
    def test_detect_ui_risk_factors(self):
        """Test UI risk factor detection."""
        # Create test image with potential risk factors
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        
        risk_factors = detect_ui_risk_factors(test_image)
        
        assert isinstance(risk_factors, list), "Should return list of risk factors"
        
        # If risk factors are found, check their structure
        for factor in risk_factors:
            assert isinstance(factor, dict), "Risk factor should be dict"
            assert 'type' in factor, "Risk factor should have type"
            assert 'description' in factor, "Risk factor should have description"
            assert 'severity' in factor, "Risk factor should have severity"
            assert factor['severity'] in ['low', 'medium', 'high', 'critical'], "Invalid severity"
    
    def test_analyze_visual_elements(self):
        """Test visual element analysis."""
        # Create test image
        test_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        
        analysis = analyze_visual_elements(test_image)
        
        assert isinstance(analysis, dict), "Should return dictionary"
        
        # Check for expected keys (if no error)
        if 'error' not in analysis:
            expected_keys = ['edge_density', 'color_variance', 'dominant_colors', 
                           'text_regions', 'layout_complexity', 'image_size']
            
            for key in expected_keys:
                assert key in analysis, f"Should have {key} in analysis"
            
            # Check data types
            assert isinstance(analysis['edge_density'], float), "Edge density should be float"
            assert isinstance(analysis['color_variance'], list), "Color variance should be list"
            assert isinstance(analysis['dominant_colors'], list), "Dominant colors should be list"
            assert isinstance(analysis['text_regions'], int), "Text regions should be int"
            assert isinstance(analysis['layout_complexity'], float), "Layout complexity should be float"
            assert isinstance(analysis['image_size'], list), "Image size should be list"
            
            # Check value ranges
            assert 0.0 <= analysis['edge_density'] <= 1.0, "Edge density out of range"
            assert 0.0 <= analysis['layout_complexity'] <= 1.0, "Layout complexity out of range"
            assert analysis['text_regions'] >= 0, "Text regions should be non-negative"
    
    def test_model_with_different_input_sizes(self):
        """Test model with various input sizes."""
        model = VisionModel()
        model.eval()
        
        # Test different input sizes
        input_sizes = [(224, 224), (128, 128), (512, 512), (300, 400)]
        
        for height, width in input_sizes:
            # Create test image
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Preprocess and run through model
            processed = preprocess_image(test_image)
            
            with torch.no_grad():
                output = model(processed)
            
            # Should always output 3 classes
            assert output.shape == (1, 3), f"Expected (1, 3), got {output.shape} for size {height}x{width}"
            assert torch.allclose(output.sum(dim=1), torch.ones(1)), "Output should sum to 1"
    
    def test_grayscale_image_handling(self):
        """Test handling of grayscale images."""
        # Create grayscale image
        gray_image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        
        # Test visual element analysis with grayscale
        analysis = analyze_visual_elements(gray_image)
        assert isinstance(analysis, dict), "Should handle grayscale images"
        
        # Test risk factor detection with grayscale
        risk_factors = detect_ui_risk_factors(gray_image)
        assert isinstance(risk_factors, list), "Should handle grayscale images"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        model = build_vision_model()
        
        # Test with very small image
        tiny_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        result = analyze_screenshot(model, tiny_image)
        
        # Should not crash and return valid result
        assert isinstance(result, dict), "Should handle tiny images"
        assert 'risk_level' in result, "Should return risk level even for tiny images"
        
        # Test with single color image
        solid_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = analyze_screenshot(model, solid_image)
        
        assert isinstance(result, dict), "Should handle solid color images"
        assert 'risk_level' in result, "Should return risk level for solid images"
    
    def test_batch_processing(self):
        """Test batch processing capability."""
        model = VisionModel()
        model.eval()
        
        # Create batch of images
        batch_size = 4
        batch_input = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(batch_input)
        
        assert output.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {output.shape}"
        
        # Each sample should be a valid probability distribution
        for i in range(batch_size):
            assert torch.allclose(output[i].sum(), torch.tensor(1.0)), f"Sample {i} doesn't sum to 1"
            assert torch.all(output[i] >= 0), f"Sample {i} has negative probabilities"
    
    def test_model_gradient_flow(self):
        """Test that gradients can flow through the model (for training)."""
        model = VisionModel()
        model.train()  # Set to training mode
        
        # Create dummy input and target
        input_tensor = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.tensor([0, 2])  # Class indices
        
        # Forward pass
        output = model(input_tensor)
        
        # Compute loss (cross-entropy)
        loss = torch.nn.functional.cross_entropy(torch.log(output + 1e-8), target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert input_tensor.grad is not None, "Input should have gradients"
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Model parameters should have gradients"
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent."""
        model = VisionModel()
        model.eval()
        
        # Same input should produce same features
        input_tensor = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            features1 = model.extract_features(input_tensor)
            features2 = model.extract_features(input_tensor)
        
        assert torch.allclose(features1, features2), "Features should be consistent"
    
    def test_ui_element_detection(self):
        """Test UI element detection functionality."""
        # Create synthetic image with button-like rectangles
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Add some rectangular regions that might be detected as buttons
        test_image[50:80, 50:150] = 200  # Button-like rectangle
        test_image[100:130, 50:250] = 150  # Input field-like rectangle
        
        # Test UI element detection
        from models.vision_model import detect_ui_elements
        ui_elements = detect_ui_elements(test_image)
        
        assert isinstance(ui_elements, dict), "Should return dictionary"
        
        if 'error' not in ui_elements:
            assert 'buttons' in ui_elements, "Should detect buttons count"
            assert 'input_fields' in ui_elements, "Should detect input fields count"
            assert 'total_elements' in ui_elements, "Should count total elements"
            assert 'form_like_structure' in ui_elements, "Should detect form-like structures"
            
            # Check data types
            assert isinstance(ui_elements['buttons'], int), "Button count should be int"
            assert isinstance(ui_elements['input_fields'], int), "Input field count should be int"
            assert isinstance(ui_elements['total_elements'], int), "Total elements should be int"
            assert isinstance(ui_elements['form_like_structure'], bool), "Form detection should be bool"

def test_model_shape():
    """Standalone test for model output shape."""
    model = build_vision_model()
    
    # Test with single input
    test_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(test_input)
    
    # Check output shape (batch_size, num_classes)
    assert output.shape == (1, 3), f"Expected (1, 3), got {output.shape}"

def test_basic_functionality():
    """Test basic model functionality."""
    model = build_vision_model()
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    result = analyze_screenshot(model, test_image)
    
    assert isinstance(result, dict), "Should return a dictionary"
    assert 'risk_level' in result, "Should have risk_level key"

if __name__ == "__main__":
    # Run basic tests
    test_model_shape()
    test_basic_functionality()
    
    # Run full test suite if pytest is available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Pytest not available, running basic tests only")
        print("All basic tests passed!")