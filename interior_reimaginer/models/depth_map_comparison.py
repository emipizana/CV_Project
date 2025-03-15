import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Union
import torch

logger = logging.getLogger(__name__)

class DepthMapComparison:
    """
    Class for creating grid visualizations to compare different depth map enhancement methods.
    
    This class provides functionality to:
    1. Generate side-by-side comparisons of original, GAN-enhanced, and diffusion-enhanced depth maps
    2. Visualize the differences between enhancement methods
    3. Create publication-ready grid visualizations
    """
    
    def __init__(self, depth_reconstructor=None):
        """
        Initialize the DepthMapComparison module.
        
        Args:
            depth_reconstructor: Optional DepthReconstructor instance. If provided, 
                                will use this instance for depth enhancement methods.
        """
        self.depth_reconstructor = depth_reconstructor
        logger.info("Initializing Depth Map Comparison module")
    
    def create_comparison_grid(self, 
                             original_depth: np.ndarray,
                             rgb_image: Image.Image,
                             width: int = 1200,
                             height: int = 400,
                             colormap: int = cv2.COLORMAP_INFERNO,
                             titles: List[str] = None) -> np.ndarray:
        """
        Create a comparison grid showing original, GAN-enhanced, and diffusion-enhanced depth maps.
        
        Args:
            original_depth: Original depth map as numpy array
            rgb_image: Original RGB image corresponding to the depth map
            width: Total width of the output grid
            height: Height of the output grid
            colormap: OpenCV colormap to use for depth visualization
            titles: List of titles for each column (default: ["Original", "GAN-Enhanced", "Diffusion-Enhanced"])
            
        Returns:
            Visualization grid as numpy array (RGB)
        """
        logger.info("Creating depth map comparison grid")
        
        # Set default titles if not provided
        if titles is None:
            titles = ["Original", "GAN-Enhanced", "Diffusion-Enhanced"]
        
        # Check if original depth is valid
        if original_depth is None or original_depth.size == 0:
            logger.warning("Invalid original depth map")
            # Create an error image
            error_img = np.ones((height, width, 3), dtype=np.uint8) * 200
            cv2.putText(
                error_img, 
                "Invalid depth map provided", 
                (width // 3, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 0), 
                2
            )
            return error_img
        
        if rgb_image is None:
            logger.warning("No RGB image provided, creating grayscale placeholder")
            # Create a grayscale placeholder image with the same dimensions as the depth map
            rgb_array = np.zeros((original_depth.shape[0], original_depth.shape[1], 3), dtype=np.uint8)
            rgb_array[..., 0] = 128
            rgb_array[..., 1] = 128
            rgb_array[..., 2] = 128
            rgb_image = Image.fromarray(rgb_array)
        
        # Convert RGB image to numpy array if it's a PIL image
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
            
        # Resize RGB if dimensions don't match
        if rgb_array.shape[:2] != original_depth.shape[:2]:
            rgb_array = cv2.resize(rgb_array, (original_depth.shape[1], original_depth.shape[0]))
            
        # Calculate individual cell size
        cell_width = width // 3
        cell_height = height
        
        # Create output image
        output_grid = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Step 1: Create original depth visualization
        try:
            original_viz = self._visualize_depth(
                depth_map=original_depth,
                width=cell_width,
                height=cell_height,
                colormap=colormap
            )
            # Place in first column
            output_grid[:cell_height, 0:cell_width] = original_viz
        except Exception as e:
            logger.error(f"Error visualizing original depth: {e}")
            # Create error message in the cell
            cv2.putText(
                output_grid[:cell_height, 0:cell_width], 
                "Error visualizing", 
                (10, cell_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                1
            )
        
        # Step 2: Generate and visualize GAN-enhanced depth
        try:
            if self.depth_reconstructor is not None:
                logger.info("Generating GAN-enhanced depth map")
                gan_depth = self.depth_reconstructor.enhance_depth_with_gan(
                    depth_map=original_depth,
                    image=rgb_image
                )
                
                # Visualize GAN-enhanced depth
                gan_viz = self._visualize_depth(
                    depth_map=gan_depth,
                    width=cell_width,
                    height=cell_height,
                    colormap=colormap
                )
                # Place in second column
                output_grid[:cell_height, cell_width:cell_width*2] = gan_viz
            else:
                logger.warning("No depth reconstructor provided, cannot generate GAN-enhanced depth")
                # Create a message in the cell
                cv2.putText(
                    output_grid[:cell_height, cell_width:cell_width*2], 
                    "No GAN model available", 
                    (10, cell_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 0), 
                    1
                )
        except Exception as e:
            logger.error(f"Error generating GAN-enhanced depth: {e}")
            # Create error message in the cell
            cv2.putText(
                output_grid[:cell_height, cell_width:cell_width*2], 
                f"GAN error: {type(e).__name__}", 
                (10, cell_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                1
            )
        
        # Step 3: Generate and visualize Diffusion-enhanced depth
        try:
            if self.depth_reconstructor is not None:
                logger.info("Generating Diffusion-enhanced depth map")
                diffusion_depth = self.depth_reconstructor.enhance_depth_with_diffusion(
                    depth_map=original_depth,
                    image=rgb_image
                )
                
                # Visualize Diffusion-enhanced depth
                diffusion_viz = self._visualize_depth(
                    depth_map=diffusion_depth,
                    width=cell_width,
                    height=cell_height,
                    colormap=colormap
                )
                # Place in third column
                output_grid[:cell_height, cell_width*2:width] = diffusion_viz
            else:
                logger.warning("No depth reconstructor provided, cannot generate Diffusion-enhanced depth")
                # Create a message in the cell
                cv2.putText(
                    output_grid[:cell_height, cell_width*2:width], 
                    "No Diffusion model available", 
                    (10, cell_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 0), 
                    1
                )
        except Exception as e:
            logger.error(f"Error generating Diffusion-enhanced depth: {e}")
            # Create error message in the cell
            cv2.putText(
                output_grid[:cell_height, cell_width*2:width], 
                f"Diffusion error: {type(e).__name__}", 
                (10, cell_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                1
            )
        
        # Add titles
        title_y_pos = 30  # Y position for the title text
        font_size = 0.8
        font_thickness = 2
        
        for i, title in enumerate(titles[:3]):  # Ensure we only use up to 3 titles
            x_pos = cell_width * i + (cell_width // 2 - 70)  # Center the title
            cv2.putText(
                output_grid, 
                title, 
                (x_pos, title_y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_size, 
                (0, 0, 0), 
                font_thickness
            )
        
        return output_grid
    
    def _visualize_depth(self, 
                         depth_map: np.ndarray, 
                         width: int = 400, 
                         height: int = 400, 
                         colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Visualize a depth map using OpenCV's colormaps.
        
        Args:
            depth_map: Depth map as numpy array
            width: Desired output width
            height: Desired output height
            colormap: OpenCV colormap to use
            
        Returns:
            Rendered colorized depth map as numpy array (RGB)
        """
        # Ensure depth map is valid
        if depth_map is None or depth_map.size == 0:
            logger.warning("Invalid depth map provided to visualizer")
            return np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray
            
        # Normalize depth map to 0-255 range for visualization
        if depth_map.max() > 0:
            normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            logger.warning("Depth map has no valid depth values")
            return np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        
        # Resize to desired dimensions
        if colored.shape[0] != height or colored.shape[1] != width:
            colored = cv2.resize(colored, (width, height))
        
        # Convert to RGB (from BGR)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored_rgb
    
    def create_detailed_comparison(self,
                                  original_depth: np.ndarray,
                                  rgb_image: Image.Image,
                                  width: int = 1200,
                                  height: int = 800,
                                  add_difference_maps: bool = True,
                                  colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Create a detailed comparison grid showing original depth, enhanced versions,
        and optionally difference maps between the methods.
        
        Args:
            original_depth: Original depth map as numpy array
            rgb_image: Original RGB image corresponding to the depth map
            width: Total width of the output grid
            height: Height of the output grid
            add_difference_maps: Whether to include difference maps
            colormap: OpenCV colormap to use for depth visualization
            
        Returns:
            Detailed visualization grid as numpy array (RGB)
        """
        logger.info("Creating detailed depth map comparison")
        
        # Determine grid layout based on whether to include difference maps
        if add_difference_maps:
            # 2x3 grid: top row is depth maps, bottom row is difference maps
            rows, cols = 2, 3
        else:
            # 1x3 grid: just depth maps
            rows, cols = 1, 3
        
        # Calculate individual cell size
        cell_width = width // cols
        cell_height = height // rows
        
        # Create output image
        output_grid = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Check if original depth is valid
        if original_depth is None or original_depth.size == 0:
            logger.warning("Invalid original depth map")
            # Create an error image
            cv2.putText(
                output_grid, 
                "Invalid depth map provided", 
                (width // 3, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 0), 
                2
            )
            return output_grid
        
        if rgb_image is None:
            logger.warning("No RGB image provided, creating grayscale placeholder")
            # Create a grayscale placeholder image
            rgb_array = np.zeros((original_depth.shape[0], original_depth.shape[1], 3), dtype=np.uint8)
            rgb_array[..., 0] = 128
            rgb_array[..., 1] = 128
            rgb_array[..., 2] = 128
            rgb_image = Image.fromarray(rgb_array)
        
        # Generate enhanced depth maps
        try:
            if self.depth_reconstructor is not None:
                gan_depth = self.depth_reconstructor.enhance_depth_with_gan(
                    depth_map=original_depth,
                    image=rgb_image
                )
                
                diffusion_depth = self.depth_reconstructor.enhance_depth_with_diffusion(
                    depth_map=original_depth,
                    image=rgb_image
                )
            else:
                # If no depth reconstructor provided, use original depth as fallback
                logger.warning("No depth reconstructor provided, using original depth as fallback")
                gan_depth = original_depth.copy()
                diffusion_depth = original_depth.copy()
        except Exception as e:
            logger.error(f"Error generating enhanced depth maps: {e}")
            # Create an error message
            cv2.putText(
                output_grid, 
                f"Error generating enhanced depth maps: {type(e).__name__}", 
                (width // 4, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 0), 
                1
            )
            return output_grid
        
        # Create visualizations for each depth map
        original_viz = self._visualize_depth(
            depth_map=original_depth,
            width=cell_width,
            height=cell_height,
            colormap=colormap
        )
        
        gan_viz = self._visualize_depth(
            depth_map=gan_depth,
            width=cell_width,
            height=cell_height,
            colormap=colormap
        )
        
        diffusion_viz = self._visualize_depth(
            depth_map=diffusion_depth,
            width=cell_width,
            height=cell_height,
            colormap=colormap
        )
        
        # Place depth map visualizations in top row
        output_grid[:cell_height, 0:cell_width] = original_viz
        output_grid[:cell_height, cell_width:cell_width*2] = gan_viz
        output_grid[:cell_height, cell_width*2:width] = diffusion_viz
        
        # Add difference maps if requested
        if add_difference_maps and rows > 1:
            # Compute difference maps - first normalize to 0-1 range
            if original_depth.max() > 0:
                orig_norm = original_depth.astype(np.float32) / original_depth.max()
            else:
                orig_norm = original_depth.astype(np.float32)
                
            if gan_depth.max() > 0:
                gan_norm = gan_depth.astype(np.float32) / gan_depth.max()
            else:
                gan_norm = gan_depth.astype(np.float32)
                
            if diffusion_depth.max() > 0:
                diff_norm = diffusion_depth.astype(np.float32) / diffusion_depth.max()
            else:
                diff_norm = diffusion_depth.astype(np.float32)
            
            # Calculate absolute differences between enhanced maps and original
            gan_diff = np.abs(gan_norm - orig_norm)
            diffusion_diff = np.abs(diff_norm - orig_norm)
            
            # Calculate difference between GAN and Diffusion
            gan_diffusion_diff = np.abs(gan_norm - diff_norm)
            
            # Create visualizations of the difference maps
            # Use a different colormap (e.g., jet) for difference maps
            gan_diff_viz = self._visualize_depth(
                depth_map=gan_diff,
                width=cell_width,
                height=cell_height,
                colormap=cv2.COLORMAP_JET
            )
            
            diffusion_diff_viz = self._visualize_depth(
                depth_map=diffusion_diff,
                width=cell_width,
                height=cell_height,
                colormap=cv2.COLORMAP_JET
            )
            
            gan_diffusion_diff_viz = self._visualize_depth(
                depth_map=gan_diffusion_diff,
                width=cell_width,
                height=cell_height,
                colormap=cv2.COLORMAP_JET
            )
            
            # Place difference map visualizations in bottom row
            output_grid[cell_height:height, 0:cell_width] = gan_diff_viz
            output_grid[cell_height:height, cell_width:cell_width*2] = diffusion_diff_viz
            output_grid[cell_height:height, cell_width*2:width] = gan_diffusion_diff_viz
        
        # Add titles
        titles_top = ["Original", "GAN-Enhanced", "Diffusion-Enhanced"]
        titles_bottom = ["GAN Difference", "Diffusion Difference", "GAN vs Diffusion"]
        
        font_size = 0.8
        font_thickness = 2
        
        # Add top row titles
        for i, title in enumerate(titles_top):
            x_pos = cell_width * i + (cell_width // 2 - 70)  # Center the title
            cv2.putText(
                output_grid, 
                title, 
                (x_pos, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_size, 
                (0, 0, 0), 
                font_thickness
            )
        
        # Add bottom row titles if showing difference maps
        if add_difference_maps and rows > 1:
            for i, title in enumerate(titles_bottom):
                x_pos = cell_width * i + (cell_width // 2 - 70)  # Center the title
                cv2.putText(
                    output_grid, 
                    title, 
                    (x_pos, cell_height + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size, 
                    (0, 0, 0), 
                    font_thickness
                )
        
        return output_grid
    
    def save_comparison(self, 
                       comparison_grid: np.ndarray, 
                       filepath: str = "depth_comparison.png") -> str:
        """
        Save a comparison grid to disk.
        
        Args:
            comparison_grid: Comparison visualization as numpy array
            filepath: Path to save the visualization
            
        Returns:
            Path to the saved file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the image
        try:
            cv2.imwrite(filepath, cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR))
            logger.info(f"Comparison grid saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving comparison grid: {e}")
            return None
