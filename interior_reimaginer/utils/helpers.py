import os
import torch
import logging
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

def setup_logging(debug=False):
    """Configure logging for the application"""
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("interior_reimaginer.log"),
            logging.StreamHandler()
        ]
    )
    
def get_device(cpu_only=False):
    """Determine the appropriate device for computation"""
    if cpu_only:
        logger.warning("Forcing CPU usage as requested. This will be significantly slower.")
        return "cpu"
    
    # First try CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    
    # Then try MPS (Apple Silicon GPUs)
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for Apple Silicon compatibility")
        logger.warning("Using Apple Silicon GPU with MPS. Some operations may fall back to CPU.")
        return "cpu"  # We use CPU with MPS fallback due to compatibility issues
    
    # Fall back to CPU
    logger.warning("No GPU detected, using CPU. Processing will be significantly slower.")
    return "cpu"

def print_system_info():
    """Print system information for debugging"""
    logger.info("System information:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")

def resize_for_model(image, multiple=32):
    """Resize image to dimensions suitable for models (multiple of 32)"""
    width, height = image.size
    # Ensure dimensions are appropriate for models
    new_width = (width // multiple) * multiple
    new_height = (height // multiple) * multiple
    
    if new_width != width or new_height != height:
        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height))
    return image

def save_temp_image(image, prefix="temp", suffix=".jpg"):
    """Save image to a temporary file and return the path"""
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}{suffix}"
    path = os.path.join(temp_dir, filename)
    image.save(path)
    return path

def create_visualization(original_image, depth_map=None, segmentation_map=None, edge_map=None, masks=None):
    """
    Create a visualization image showing the original and processing results
    
    Args:
        original_image: PIL Image of the original input
        depth_map: Depth map as numpy array
        segmentation_map: Segmentation map as numpy array
        edge_map: Edge map as numpy array
        masks: Dictionary of object masks
        
    Returns:
        PIL Image with visualization
    """
    # Convert original to numpy
    orig_np = np.array(original_image.convert('RGB'))
    
    # Create base visualization
    rows = []
    rows.append(orig_np)
    
    # Add depth map if available
    if depth_map is not None:
        # Convert to colormap for visualization
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        rows.append(depth_color)
    
    # Add edge map if available
    if edge_map is not None:
        # Convert to RGB for visualization
        edge_color = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        rows.append(edge_color)
    
    # Add segmentation map if available
    if segmentation_map is not None:
        # Create a colormap for visualization
        seg_color = cv2.applyColorMap((segmentation_map * 25) % 255, cv2.COLORMAP_JET)
        seg_color = cv2.cvtColor(seg_color, cv2.COLOR_BGR2RGB)
        rows.append(seg_color)
    
    # Stack the rows and convert back to PIL
    if len(rows) > 1:
        # Ensure all images have the same width
        max_width = max(img.shape[1] for img in rows)
        for i in range(len(rows)):
            if rows[i].shape[1] != max_width:
                rows[i] = cv2.resize(rows[i], (max_width, int(rows[i].shape[0] * max_width / rows[i].shape[1])))
        
        # Create the final visualization
        viz = np.vstack(rows)
        return Image.fromarray(viz)
    else:
        return original_image