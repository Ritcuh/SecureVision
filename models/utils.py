import os
import io
import numpy as np
from PIL import Image
from typing import Optional, Union
import streamlit as st

def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load and preprocess image from Streamlit file upload.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Image as numpy array or None if error
    """
    if uploaded_file is None:
        return None
    
    try:
        # Read image data
        image_data = uploaded_file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize to 0-1 range
        if image_array.max() > 1:
            image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def load_image(path: Union[str, io.BytesIO]) -> Optional[np.ndarray]:
    """
    Load and preprocess image from file path or BytesIO object.
    
    Args:
        path: File path or BytesIO object
    
    Returns:
        Image as numpy array or None if error
    """
    try:
        # Handle different input types
        if isinstance(path, str):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            image = Image.open(path)
        else:
            image = Image.open(path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size for consistency
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        return image_array
        
    except Exception as e:
        print(f"Error loading image from {path}: {str(e)}")
        return None

def load_text_file_from_upload(uploaded_file) -> Optional[str]:
    """
    Load text content from Streamlit file upload.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        File content as string or None if error
    """
    if uploaded_file is None:
        return None
    
    try:
        # Read file content
        content = uploaded_file.read()
        
        # Decode based on file type
        if isinstance(content, bytes):
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, return with error replacement
            return content.decode('utf-8', errors='replace')
        else:
            return str(content)
            
    except Exception as e:
        st.error(f"Error loading text file: {str(e)}")
        return None

def load_text_file(path: str) -> Optional[str]:
    """
    Load text content from file path.
    
    Args:
        path: File path
    
    Returns:
        File content as string or None if error
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with error replacement
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
            
    except Exception as e:
        print(f"Error loading text file from {path}: {str(e)}")
        return None

def save_image(image_array: np.ndarray, path: str) -> bool:
    """
    Save numpy array as image file.
    
    Args:
        image_array: Image as numpy array
        path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure image is in correct format
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save image
        image.save(path)
        return True
        
    except Exception as e:
        print(f"Error saving image to {path}: {str(e)}")
        return False

def save_text_file(content: str, path: str) -> bool:
    """
    Save text content to file.
    
    Args:
        content: Text content to save
        path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save text file
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error saving text file to {path}: {str(e)}")
        return False

def validate_image_format(uploaded_file) -> bool:
    """
    Validate if uploaded file is a supported image format.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        True if valid image format, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file extension
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension not in valid_extensions:
        return False
    
    # Try to open as image
    try:
        image_data = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        Image.open(io.BytesIO(image_data))
        return True
    except Exception:
        return False

def validate_code_file(uploaded_file) -> bool:
    """
    Validate if uploaded file contains code.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        True if valid code file, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file extension
    code_extensions = ['.txt', '.html', '.htm', '.js', '.css', '.php', '.py', 
                      '.java', '.cpp', '.c', '.cs', '.rb', '.go', '.rs', '.jsx', '.tsx']
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    return file_extension in code_extensions

def get_file_size(uploaded_file) -> int:
    """
    Get file size in bytes.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        File size in bytes
    """
    if uploaded_file is None:
        return 0
    
    try:
        # Get current position
        current_pos = uploaded_file.tell()
        
        # Seek to end to get size
        uploaded_file.seek(0, 2)
        size = uploaded_file.tell()
        
        # Reset position
        uploaded_file.seek(current_pos)
        
        return size
    except Exception:
        return 0

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def create_thumbnail(image_array: np.ndarray, size: tuple = (150, 150)) -> np.ndarray:
    """
    Create thumbnail from image array.
    
    Args:
        image_array: Source image as numpy array
        size: Thumbnail size as (width, height)
    
    Returns:
        Thumbnail as numpy array
    """
    try:
        # Convert to PIL Image
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        # Create thumbnail
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert back to numpy array
        thumbnail = np.array(image)
        
        return thumbnail
        
    except Exception as e:
        print(f"Error creating thumbnail: {str(e)}")
        return image_array

def extract_metadata(uploaded_file) -> dict:
    """
    Extract metadata from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Dictionary with file metadata
    """
    if uploaded_file is None:
        return {}
    
    metadata = {
        'filename': uploaded_file.name,
        'size': get_file_size(uploaded_file),
        'size_formatted': format_file_size(get_file_size(uploaded_file)),
        'type': uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
    }
    
    # Add file extension
    _, ext = os.path.splitext(uploaded_file.name)
    metadata['extension'] = ext.lower()
    
    # Try to extract image-specific metadata
    if validate_image_format(uploaded_file):
        try:
            image_data = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            image = Image.open(io.BytesIO(image_data))
            metadata.update({
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format
            })
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                metadata['has_exif'] = True
            else:
                metadata['has_exif'] = False
                
        except Exception as e:
            metadata['image_error'] = str(e)
    
    return metadata

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def get_supported_formats() -> dict:
    """
    Get dictionary of supported file formats.
    
    Returns:
        Dictionary with format categories and extensions
    """
    return {
        'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'],
        'code': ['.txt', '.html', '.htm', '.js', '.css', '.php', '.py', 
                '.java', '.cpp', '.c', '.cs', '.rb', '.go', '.rs', '.jsx', '.tsx'],
        'web': ['.html', '.htm', '.css', '.js', '.jsx', '.tsx', '.php'],
        'text': ['.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml']
    }

def is_safe_file(filename: str, max_size_mb: int = 10) -> tuple:
    """
    Check if file is safe for processing.
    
    Args:
        filename: Name of the file
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        Tuple of (is_safe: bool, reason: str)
    """
    # Check filename
    if not filename or len(filename.strip()) == 0:
        return False, "Invalid filename"
    
    # Check for potentially dangerous extensions
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js']
    _, ext = os.path.splitext(filename)
    
    if ext.lower() in dangerous_extensions:
        return False, f"Potentially dangerous file extension: {ext}"
    
    # Check if extension is supported
    supported = get_supported_formats()
    all_supported = []
    for category in supported.values():
        all_supported.extend(category)
    
    if ext.lower() not in all_supported:
        return False, f"Unsupported file extension: {ext}"
    
    return True, "File appears safe"