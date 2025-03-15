"""
Depth Map Comparison Module

This module provides tools for comparing different depth map enhancement methods,
including original depth maps, GAN-enhanced depth maps, and diffusion-enhanced depth maps.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import os

class DepthMapComparison:
    """
    A class for comparing and visualizing different depth map enhancement methods.
    """
    
    def __init__(self, depth_reconstructor=None):
        """
        Initialize the depth map comparison tool.
        
        Args:
            depth_reconstructor: A DepthReconstructor instance (optional)
        """
        self.depth_reconstructor = depth_reconstructor
    
    def normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize a depth map to the range 0-255.
        
        Args:
            depth_map: Raw depth map array
            
        Returns:
            Normalized depth map as uint8
        """
        if depth_map.dtype == np.float32 or depth_map.dtype == np.float64:
            # Handle float depth maps (typically 0-1 range)
            normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        else:
            # Handle integer depth maps
            normalized = (depth_map.astype(np.float32) - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        
        return normalized.astype(np.uint8)
    
    def apply_colormap(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Apply a colormap to a depth map for better visualization.
        
        Args:
            depth_map: Depth map array
            colormap: OpenCV colormap constant (default: COLORMAP_INFERNO)
            
        Returns:
            Colorized depth map
        """
        # Ensure depth map is normalized to 0-255 range
        normalized_depth = self.normalize_depth_map(depth_map)
        
        # Apply colormap
        colored_depth = cv2.applyColorMap(normalized_depth, colormap)
        
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        colored_depth_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
        
        return colored_depth_rgb
        
    def create_comparison_grid(
        self, 
        original_depth: np.ndarray, 
        rgb_image: Union[np.ndarray, Image.Image], 
        width: int = 1200, 
        height: int = 400, 
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Create a side-by-side comparison grid of original, GAN-enhanced, and diffusion-enhanced depth maps.
        
        Args:
            original_depth: Original depth map array
            rgb_image: RGB image corresponding to the depth map (PIL Image or numpy array)
            width: Width of the output comparison grid
            height: Height of the output comparison grid
            colormap: OpenCV colormap constant to use for depth visualization
            
        Returns:
            Comparison grid as a numpy array
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
        
        # Generate enhanced depth maps using the reconstructor
        # Fallback to original if enhancement fails or reconstructor not available
        if self.depth_reconstructor is not None:
            try:
                gan_depth = self.depth_reconstructor.enhance_depth_with_gan(original_depth, rgb_image)
            except Exception as e:
                print(f"Warning: GAN enhancement failed ({str(e)}). Using original depth map instead.")
                gan_depth = original_depth
                
            try:
                diffusion_depth = self.depth_reconstructor.enhance_depth_with_diffusion(original_depth, rgb_image)
            except Exception as e:
                print(f"Warning: Diffusion enhancement failed ({str(e)}). Using original depth map instead.")
                diffusion_depth = original_depth
        else:
            print("Warning: No depth reconstructor provided. Using original depth map for all visualizations.")
            gan_depth = original_depth
            diffusion_depth = original_depth
        
        # Apply colormap to depth maps
        original_colored = self.apply_colormap(original_depth, colormap)
        gan_colored = self.apply_colormap(gan_depth, colormap)
        diffusion_colored = self.apply_colormap(diffusion_depth, colormap)
        
        # Resize RGB image to match depth visualizations
        single_width = width // 3
        single_height = height
        rgb_resized = cv2.resize(rgb_array, (single_width, single_height))
        
        # Resize depth visualizations
        original_resized = cv2.resize(original_colored, (single_width, single_height))
        gan_resized = cv2.resize(gan_colored, (single_width, single_height))
        diffusion_resized = cv2.resize(diffusion_colored, (single_width, single_height))
        
        # Create comparison grid
        comparison = np.zeros((single_height, width, 3), dtype=np.uint8)
        
        # Add title text to each section
        title_height = 30
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.8
        title_thickness = 2
        title_color = (255, 255, 255)
        
        # Create blank space for titles
        comparison = np.zeros((single_height + title_height, width, 3), dtype=np.uint8)
        
        # Add titles
        cv2.putText(comparison, "Original Depth Map", (single_width//2 - 100, 25), 
                   title_font, title_scale, title_color, title_thickness)
        cv2.putText(comparison, "GAN-Enhanced Depth Map", (single_width + single_width//2 - 120, 25), 
                   title_font, title_scale, title_color, title_thickness)
        cv2.putText(comparison, "Diffusion-Enhanced Depth Map", (2*single_width + single_width//2 - 140, 25), 
                   title_font, title_scale, title_color, title_thickness)
        
        # Place images in grid
        comparison[title_height:, 0:single_width] = original_resized
        comparison[title_height:, single_width:2*single_width] = gan_resized
        comparison[title_height:, 2*single_width:] = diffusion_resized
        
        return comparison
    
    def compute_depth_difference(self, original_depth: np.ndarray, enhanced_depth: np.ndarray) -> np.ndarray:
        """
        Compute the absolute difference between original and enhanced depth maps.
        
        Args:
            original_depth: Original depth map
            enhanced_depth: Enhanced depth map
            
        Returns:
            Normalized difference map
        """
        # Normalize both depth maps to 0-1 range for fair comparison
        if original_depth.dtype != np.float32 and original_depth.dtype != np.float64:
            orig_norm = original_depth.astype(np.float32) / 255.0
        else:
            orig_norm = original_depth.copy()
            # Ensure range 0-1
            if np.max(orig_norm) > 1.0:
                orig_norm = orig_norm / 255.0
        
        if enhanced_depth.dtype != np.float32 and enhanced_depth.dtype != np.float64:
            enh_norm = enhanced_depth.astype(np.float32) / 255.0
        else:
            enh_norm = enhanced_depth.copy()
            # Ensure range 0-1
            if np.max(enh_norm) > 1.0:
                enh_norm = enh_norm / 255.0
                
        # Compute absolute difference
        diff = np.abs(orig_norm - enh_norm)
        
        # Normalize to 0-1 for visualization
        if np.max(diff) > 0:
            diff = diff / np.max(diff)
        
        return diff
    
    def create_detailed_comparison(
        self, 
        original_depth: np.ndarray, 
        rgb_image: Union[np.ndarray, Image.Image], 
        width: int = 1200, 
        height: int = 800, 
        add_difference_maps: bool = True,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Create a detailed comparison with original image, depth maps, and difference maps.
        
        Args:
            original_depth: Original depth map array
            rgb_image: RGB image corresponding to the depth map
            width: Width of the output comparison grid
            height: Height of the output comparison grid
            add_difference_maps: Whether to include difference maps in the visualization
            colormap: OpenCV colormap constant to use for depth visualization
            
        Returns:
            Detailed comparison grid as a numpy array
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
        
        # Generate enhanced depth maps
        if self.depth_reconstructor is not None:
            try:
                gan_depth = self.depth_reconstructor.enhance_depth_with_gan(original_depth, rgb_image)
            except Exception as e:
                print(f"Warning: GAN enhancement failed ({str(e)}). Using original depth map instead.")
                gan_depth = original_depth
                
            try:
                diffusion_depth = self.depth_reconstructor.enhance_depth_with_diffusion(original_depth, rgb_image)
            except Exception as e:
                print(f"Warning: Diffusion enhancement failed ({str(e)}). Using original depth map instead.")
                diffusion_depth = original_depth
        else:
            print("Warning: No depth reconstructor provided. Using original depth map for all visualizations.")
            gan_depth = original_depth
            diffusion_depth = original_depth
        
        # Compute difference maps if requested
        if add_difference_maps:
            gan_diff = self.compute_depth_difference(original_depth, gan_depth)
            diffusion_diff = self.compute_depth_difference(original_depth, diffusion_depth)
            
            # Apply heat colormap to difference maps (red = more difference)
            gan_diff_colored = self.apply_colormap(gan_diff * 255, cv2.COLORMAP_HOT)
            diffusion_diff_colored = self.apply_colormap(diffusion_diff * 255, cv2.COLORMAP_HOT)
            
            # Create 2-row grid
            rows = 2
            cols = 3
            cell_width = width // cols
            cell_height = height // rows
            
            # Create grid
            comparison = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Apply colormap to depth maps
            original_colored = self.apply_colormap(original_depth, colormap)
            gan_colored = self.apply_colormap(gan_depth, colormap)
            diffusion_colored = self.apply_colormap(diffusion_depth, colormap)
            
            # Resize all images
            rgb_resized = cv2.resize(rgb_array, (cell_width, cell_height))
            original_resized = cv2.resize(original_colored, (cell_width, cell_height))
            gan_resized = cv2.resize(gan_colored, (cell_width, cell_height))
            diffusion_resized = cv2.resize(diffusion_colored, (cell_width, cell_height))
            gan_diff_resized = cv2.resize(gan_diff_colored, (cell_width, cell_height))
            diffusion_diff_resized = cv2.resize(diffusion_diff_colored, (cell_width, cell_height))
            
            # Add title text to each section
            title_height = 25
            title_font = cv2.FONT_HERSHEY_SIMPLEX
            title_scale = 0.7
            title_thickness = 2
            title_color = (255, 255, 255)
            
            # Add top row (RGB and depth maps)
            comparison[0:cell_height, 0:cell_width] = rgb_resized
            comparison[0:cell_height, cell_width:2*cell_width] = original_resized
            comparison[0:cell_height, 2*cell_width:] = gan_resized
            
            # Add bottom row (diffusion depth and difference maps)
            comparison[cell_height:, 0:cell_width] = diffusion_resized
            comparison[cell_height:, cell_width:2*cell_width] = gan_diff_resized
            comparison[cell_height:, 2*cell_width:] = diffusion_diff_resized
            
            # Add titles
            cv2.putText(comparison, "Original RGB Image", (cell_width//2 - 120, 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "Original Depth Map", (cell_width + cell_width//2 - 120, 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "GAN-Enhanced Depth Map", (2*cell_width + cell_width//2 - 130, 20), 
                       title_font, title_scale, title_color, title_thickness)
            
            cv2.putText(comparison, "Diffusion-Enhanced Depth Map", (cell_width//2 - 160, cell_height + 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "GAN Enhancement Difference", (cell_width + cell_width//2 - 140, cell_height + 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "Diffusion Enhancement Difference", (2*cell_width + cell_width//2 - 160, cell_height + 20), 
                       title_font, title_scale, title_color, title_thickness)
        else:
            # Create simple 1-row grid with RGB, original depth, and enhanced depths
            cell_width = width // 4
            cell_height = height
            
            # Create grid
            comparison = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Apply colormap to depth maps
            original_colored = self.apply_colormap(original_depth, colormap)
            gan_colored = self.apply_colormap(gan_depth, colormap)
            diffusion_colored = self.apply_colormap(diffusion_depth, colormap)
            
            # Resize all images
            rgb_resized = cv2.resize(rgb_array, (cell_width, cell_height))
            original_resized = cv2.resize(original_colored, (cell_width, cell_height))
            gan_resized = cv2.resize(gan_colored, (cell_width, cell_height))
            diffusion_resized = cv2.resize(diffusion_colored, (cell_width, cell_height))
            
            # Place images in grid
            comparison[0:cell_height, 0:cell_width] = rgb_resized
            comparison[0:cell_height, cell_width:2*cell_width] = original_resized
            comparison[0:cell_height, 2*cell_width:3*cell_width] = gan_resized
            comparison[0:cell_height, 3*cell_width:] = diffusion_resized
            
            # Add titles
            title_height = 25
            title_font = cv2.FONT_HERSHEY_SIMPLEX
            title_scale = 0.7
            title_thickness = 2
            title_color = (255, 255, 255)
            
            cv2.putText(comparison, "Original RGB Image", (cell_width//2 - 100, 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "Original Depth Map", (cell_width + cell_width//2 - 100, 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "GAN-Enhanced Depth", (2*cell_width + cell_width//2 - 100, 20), 
                       title_font, title_scale, title_color, title_thickness)
            cv2.putText(comparison, "Diffusion-Enhanced Depth", (3*cell_width + cell_width//2 - 120, 20), 
                       title_font, title_scale, title_color, title_thickness)
            
        return comparison
    
    def save_comparison(self, comparison_grid: np.ndarray, filepath: str) -> str:
        """
        Save a comparison grid to a file.
        
        Args:
            comparison_grid: The comparison grid to save
            filepath: Path to save the file to
            
        Returns:
            Path to the saved file
        """
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save the comparison grid
        cv2.imwrite(filepath, cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR))
        
        return filepath

if __name__ == "__main__":
    # Simple test
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load a sample image and create a synthetic depth map
    img = Image.open("sample_image.jpg")
    img_array = np.array(img)
    
    # Create a synthetic depth map (gradient from top to bottom)
    h, w = img_array.shape[:2]
    depth_map = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        depth_map[i, :] = int(255 * (i / h))
    
    # Initialize comparison tool
    comparator = DepthMapComparison()
    
    # Create a simple comparison grid
    comparison = comparator.create_comparison_grid(depth_map, img)
    
    # Display the comparison
    plt.figure(figsize=(15, 5))
    plt.imshow(comparison)
    plt.title("Depth Map Comparison")
    plt.axis('off')
    plt.show()
    
    # Save the comparison
    comparator.save_comparison(comparison, "depth_comparison.png")
