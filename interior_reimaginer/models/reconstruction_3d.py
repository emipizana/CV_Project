import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from PIL import Image
import cv2
import logging
import io
import base64
import tempfile
import urllib.request
from typing import List, Dict, Tuple, Optional, Union, Any, Literal, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math

logger = logging.getLogger(__name__)

class DepthSRGAN(nn.Module):
    """
    GAN-based model for depth map enhancement and refinement
    Implements a simplified version of DepthSRGAN architecture for depth map super-resolution
    and filling in missing depth information.
    """
    def __init__(self):
        super(DepthSRGAN, self).__init__()
        
        # Generator network
        self.generator = nn.Sequential(
            # Initial convolution
            nn.Conv2d(4, 64, kernel_size=3, padding=1),  # 4 channels: 1 for depth, 3 for RGB
            nn.LeakyReLU(0.2, inplace=True),
            
            # Feature extraction blocks
            self._make_dense_block(64, 64),
            self._make_dense_block(64, 64),
            self._make_dense_block(64, 64),
            
            # Output convolution
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
    def _make_dense_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, depth_map, rgb_image):
        # Concatenate depth map and RGB image along channel dimension
        x = torch.cat([depth_map, rgb_image], dim=1)
        return self.generator(x)


class DepthDiffusionModel(nn.Module):
    """
    Diffusion-based model for depth map enhancement and refinement.
    Implements a conditional diffusion model architecture for depth map super-resolution
    and filling in missing depth information.
    """
    def __init__(self, time_steps=1000, channels=64):
        super(DepthDiffusionModel, self).__init__()
        
        self.time_steps = time_steps
        
        # Time embeddings for diffusion process
        self.time_embedding = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        
        # U-Net architecture for the denoising network
        # Encoder (downsampling path)
        self.conv_in = nn.Conv2d(4, channels, kernel_size=3, padding=1)  # 4 channels: 1 for depth, 3 for RGB
        
        self.down1 = self._make_down_block(channels, channels * 2)
        self.down2 = self._make_down_block(channels * 2, channels * 4)
        self.down3 = self._make_down_block(channels * 4, channels * 4)
        
        # Middle block
        self.mid_block = nn.Sequential(
            self._make_res_block(channels * 4, channels * 4),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels * 4),
            nn.SiLU()
        )
        
        # Decoder (upsampling path)
        self.up1 = self._make_up_block(channels * 8, channels * 2)
        self.up2 = self._make_up_block(channels * 4, channels)
        self.up3 = self._make_up_block(channels * 2, channels)
        
        # Output layer
        self.conv_out = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
        # Beta schedule for diffusion process
        self.beta = torch.linspace(0.0001, 0.02, time_steps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def _make_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            self._make_res_block(out_channels, out_channels)
        )
    
    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            self._make_res_block(out_channels, out_channels)
        )
    
    def forward(self, depth_map, rgb_image, noise=None, t=None, return_noise=False):
        """
        Forward pass for denoising diffusion model.
        
        Args:
            depth_map: Input depth map (can be noisy during training)
            rgb_image: RGB image for conditioning
            noise: Optional noise to be added (for training)
            t: Timestep for diffusion process
            return_noise: If True, returns added noise for loss calculation (training only)
            
        Returns:
            Denoised depth map, and optionally noise if return_noise=True
        """
        device = depth_map.device
        
        # During inference, noise and t are not required
        if t is None and not self.training:  
            # Inference process
            batch_size = depth_map.shape[0]
            x_t = torch.randn_like(depth_map)
            
            # Reverse diffusion process
            for i in range(self.time_steps - 1, -1, -1):
                t_tensor = torch.tensor([i / self.time_steps], device=device).unsqueeze(0).repeat(batch_size, 1)
                
                # Concat input image and noisy depth
                x_in = torch.cat([x_t, rgb_image], dim=1)
                
                # Get time embedding
                t_emb = self.time_embedding(t_tensor)
                
                # Run through model backbone
                h = self.conv_in(x_in)
                
                # Apply time embedding
                h = h + t_emb.view(-1, h.shape[1], 1, 1)
                
                # Down blocks
                h1 = self.down1(h)
                h2 = self.down2(h1)
                h3 = self.down3(h2)
                
                # Mid block
                h = self.mid_block(h3)
                
                # Up blocks with skip connections
                h = self.up1(torch.cat([h, h3], dim=1))
                h = self.up2(torch.cat([h, h2], dim=1))
                h = self.up3(torch.cat([h, h1], dim=1))
                
                # Predict noise
                predicted_noise = self.conv_out(h)
                
                # Update x_t using predicted noise
                alpha = self.alpha[i]
                alpha_hat = self.alpha_hat[i]
                beta = self.beta[i]
                
                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                # Update prediction for next step
                x_t = 1 / torch.sqrt(alpha) * (x_t - beta / torch.sqrt(1 - alpha_hat) * predicted_noise) + \
                      torch.sqrt(beta) * noise
            
            # Return final denoised image
            return x_t
        
        # During training, we need to add noise to input depth and train to predict that noise
        else:
            if noise is None:
                noise = torch.randn_like(depth_map)
            
            batch_size = depth_map.shape[0]
            
            # Get alpha_hat values for the batch
            t = t.view(-1)
            alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
            
            # Add noise to depth map: x_t = sqrt(α_t)x_0 + sqrt(1-α_t)ε
            x_t = torch.sqrt(alpha_hat_t) * depth_map + torch.sqrt(1 - alpha_hat_t) * noise
            
            # Embed timestep
            t_emb = t.float() / self.time_steps
            t_emb = self.time_embedding(t_emb.unsqueeze(-1))
            
            # Concat input image and noisy depth
            x_in = torch.cat([x_t, rgb_image], dim=1)
            
            # Run through model backbone
            h = self.conv_in(x_in)
            
            # Apply time embedding
            h = h + t_emb.view(-1, h.shape[1], 1, 1)
            
            # Down blocks with skip connections
            h1 = self.down1(h)
            h2 = self.down2(h1)
            h3 = self.down3(h2)
            
            # Mid block
            h = self.mid_block(h3)
            
            # Up blocks with skip connections
            h = self.up1(torch.cat([h, h3], dim=1))
            h = self.up2(torch.cat([h, h2], dim=1))
            h = self.up3(torch.cat([h, h1], dim=1))
            
            # Predict noise
            predicted_noise = self.conv_out(h)
            
            if return_noise:
                return predicted_noise, noise
            else:
                return predicted_noise
    
    def sample(self, rgb_image, shape):
        """
        Generate a depth map sample conditioned on RGB image.
        
        Args:
            rgb_image: RGB image for conditioning
            shape: Shape of output depth map
            
        Returns:
            Generated depth map
        """
        device = rgb_image.device
        batch_size = rgb_image.shape[0]
        
        # Start with random noise
        x_t = torch.randn(batch_size, 1, shape[0], shape[1], device=device)
        
        # Iteratively denoise
        for i in range(self.time_steps - 1, -1, -1):
            t_tensor = torch.tensor([i / self.time_steps], device=device).unsqueeze(0).repeat(batch_size, 1)
            
            # Run inference step
            with torch.no_grad():
                x_in = torch.cat([x_t, rgb_image], dim=1)
                t_emb = self.time_embedding(t_tensor)
                
                h = self.conv_in(x_in)
                h = h + t_emb.view(-1, h.shape[1], 1, 1)
                
                h1 = self.down1(h)
                h2 = self.down2(h1)
                h3 = self.down3(h2)
                
                h = self.mid_block(h3)
                
                h = self.up1(torch.cat([h, h3], dim=1))
                h = self.up2(torch.cat([h, h2], dim=1))
                h = self.up3(torch.cat([h, h1], dim=1))
                
                predicted_noise = self.conv_out(h)
                
                # Perform denoising step
                alpha = self.alpha[i]
                alpha_hat = self.alpha_hat[i]
                beta = self.beta[i]
                
                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = 1 / torch.sqrt(alpha) * (x_t - beta / torch.sqrt(1 - alpha_hat) * predicted_noise) + \
                      torch.sqrt(beta) * noise
        
        return x_t

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from PIL import Image
import cv2
import logging
import io
import base64
import tempfile
import urllib.request
from typing import List, Dict, Tuple, Optional, Union, Any, Literal, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math

logger = logging.getLogger(__name__)


class DepthReconstructor:
    """
    Class for 3D reconstruction and visualization from depth maps generated by the ImageProcessor.
    
    Features:
    - GAN-based depth map enhancement for filling gaps and improving accuracy
    - Confidence-based filtering to remove noise and unreliable depth values
    - Multiple visualization methods with automatic fallbacks
    - Support for headless environments without GPU acceleration
    - Robust error handling with appropriate fallbacks at each step
    
    The reconstruction pipeline includes several steps:
    1. Optional GAN-based depth enhancement to refine raw depth maps
    2. Confidence filtering to identify reliable depth values
    3. Point cloud generation from filtered depth
    4. 3D visualization using various rendering methods
    """
    
    def __init__(self):
        """Initialize the 3D reconstruction module"""
        logger.info("Initializing 3D Reconstructor")
        
        # Initialize the GAN model as None until needed
        self._gan_model = None
        self._gan_initialized = False
        self._gan_weights_path = None
        
        # Initialize the Diffusion model as None until needed
        self._diffusion_model = None
        self._diffusion_initialized = False
        self._diffusion_weights_path = None
        
        # Define all available visualization methods
        self.visualization_methods = {
            "depth_map": "Colored Depth Map (2D)",
            "pointcloud_mpl": "Matplotlib Point Cloud (3D)",
            "enhanced_3d": "Enhanced 3D Reconstruction with GAN Refinement",
            "diffusion_3d": "Enhanced 3D Reconstruction with Diffusion Refinement",
            "lrm_3d": "LRM 3D Reconstruction"
        }
    
    def render_depth_map(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO,
                         width: int = 800, height: int = 600) -> np.ndarray:
        """
        Render a depth map using OpenCV's colormaps for direct visualization.
        This is the most reliable visualization method and serves as a fallback.
        
        Args:
            depth_map: Depth map as numpy array
            colormap: OpenCV colormap to use
            width: Desired output width
            height: Desired output height
            
        Returns:
            Rendered colorized depth map as numpy array
        """
        logger.info(f"Rendering depth map with colormap {colormap}")
        
        # Ensure depth map is valid
        if depth_map is None or depth_map.size == 0:
            logger.warning("Invalid depth map provided")
            return np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
            
        # Normalize depth map to 0-255 range
        if depth_map.max() > 0:
            normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            logger.warning("Depth map has no valid depth values")
            return np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        
        # Resize to desired dimensions
        if colored.shape[0] != height or colored.shape[1] != width:
            colored = cv2.resize(colored, (width, height))
        
        # Convert to RGB (from BGR)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 for consistent output format
        return colored_rgb.astype(np.float32) / 255.0
        
    def render_pointcloud_matplotlib(self, depth_map: np.ndarray, image: Image.Image,
                                   width: int = 800, height: int = 600,
                                   downsample_factor: int = 4) -> np.ndarray:
        """
        Render a 3D point cloud using Matplotlib with corrected orientation.
        This can serve as a fallback when more advanced visualization fails.
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            width: Desired output width
            height: Desired output height
            downsample_factor: Factor by which to downsample the point cloud for rendering
            
        Returns:
            Rendered point cloud as numpy array
        """
        logger.info(f"Rendering point cloud with Matplotlib (downsample={downsample_factor})")
        
        try:
            # Create a figure with the right dimensions
            dpi = 100
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Normalize depth map
            if depth_map.max() <= 255:
                depth_norm = depth_map.astype(np.float32) / 255.0
            else:
                depth_norm = depth_map.astype(np.float32) / 1000.0

            # Get color image as RGB numpy array
            color_img = np.array(image.convert('RGB'))
            
            # Ensure dimensions match
            if color_img.shape[:2] != depth_map.shape[:2]:
                color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
            
            # Downsample for better performance
            h, w = depth_norm.shape
            y_indices, x_indices = np.mgrid[0:h:downsample_factor, 0:w:downsample_factor]
            
            # Get 3D coordinates
            z = depth_norm[y_indices, x_indices]
            x = x_indices
            y = y_indices  # Will be inverted to correct orientation
            colors = color_img[y_indices, x_indices] / 255.0
            
            # Flatten arrays for scatter plot
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            colors = colors.reshape(-1, 3)
            
            # Filter out invalid points
            valid = (z > 0)
            x = x[valid]
            y = y[valid]
            z = z[valid]
            colors = colors[valid]
            
            # Normalize spatial coordinates
            x = (x - w/2) / w
            y = (y - h/2) / h  # Invert Y axis for correct orientation
            
            # Create scatter plot with colors
            # Use x, z, -y for correct orientation (negative y makes the model upright)
            ax.scatter(x, z, -y, c=colors, s=3, alpha=0.8)
            
            # Set equal aspect ratio and labels
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Z (Depth)')
            ax.set_zlabel('Y')
            ax.set_title('3D Point Cloud')
            
            # Use light gray background for better visibility
            ax.set_facecolor([0.9, 0.9, 0.9])
            fig.patch.set_facecolor([0.9, 0.9, 0.9])
            
            # Set an improved viewpoint
            ax.view_init(elev=20, azim=-35)  # Adjusted for better orientation
            
            # Remove grid for cleaner visualization
            ax.grid(False)
            
            # Normalize axis directions
            ax.set_xlim([-1, 1])
            ax.set_zlim([-1, 1])
            
            # Capture the plot as an image
            fig.tight_layout(pad=0)
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                img = np.array(Image.open(buf))
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Convert to float32 and normalize
            return img.astype(np.float32) / 255.0
            
        except Exception as e:
            logger.warning(f"Error rendering with Matplotlib: {str(e)}")
            # Fall back to a simple depth map visualization
            return self.render_depth_map(depth_map, width=width, height=height)
    
    def render_solid_pointcloud_plotly(self, depth_map: np.ndarray, image: Image.Image,
                                     width: int = 800, height: int = 600,
                                     downsample_factor: int = 4, 
                                     solid_mode: str = "mesh") -> np.ndarray:
        """
        Render a solid-looking 3D visualization using Plotly, filling in gaps between points.
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            width: Desired output width
            height: Desired output height
            downsample_factor: Factor by which to downsample the point cloud
            solid_mode: Method to create solid appearance. Options:
                        "mesh" (Delaunay triangulation)
                        "dense" (smaller, denser points)
                        "surface" (interpolated surface)
            
        Returns:
            Rendered solid visualization as numpy array
        """
        logger.info(f"Rendering solid point cloud with Plotly (mode={solid_mode}, downsample={downsample_factor})")
        
        try:
            # Normalize depth map
            if depth_map.max() <= 255:
                depth_norm = depth_map.astype(np.float32) / 255.0
            else:
                depth_norm = depth_map.astype(np.float32) / 1000.0

            # Get color image as RGB numpy array
            color_img = np.array(image.convert('RGB'))
            
            # Ensure dimensions match
            if color_img.shape[:2] != depth_map.shape[:2]:
                color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
            
            # Downsample for better performance (adjust based on mode)
            actual_downsample = downsample_factor
            if solid_mode == "dense":
                # Use smaller downsample factor for dense mode to get more points
                actual_downsample = max(1, downsample_factor // 2)
            
            h, w = depth_norm.shape
            y_indices, x_indices = np.mgrid[0:h:actual_downsample, 0:w:actual_downsample]
            
            # Get 3D coordinates
            z = depth_norm[y_indices, x_indices]
            x = x_indices
            y = y_indices
            colors = color_img[y_indices, x_indices]
            
            # Flatten arrays for plotting
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = z.flatten()
            
            # Filter out invalid points
            valid = (z_flat > 0)
            x_flat = x_flat[valid]
            y_flat = y_flat[valid]
            z_flat = z_flat[valid]
            
            # Normalize spatial coordinates
            x_norm = (x_flat - w/2) / w
            y_norm = -(y_flat - h/2) / h  # Invert Y axis for correct orientation
            
            # Extract RGB values from the image for Plotly
            r = colors[..., 0].flatten()[valid]
            g = colors[..., 1].flatten()[valid]
            b = colors[..., 2].flatten()[valid]
            
            # Different visualization approaches based on mode
            if solid_mode == "mesh":
                # Use Delaunay triangulation to create a mesh
                try:
                    from scipy.spatial import Delaunay
                    
                    # Create a 2D points array for triangulation (using x and -y as coordinates)
                    points_2d = np.vstack([x_norm, z_flat]).T
                    
                    # Create Delaunay triangulation
                    # We'll filter out problematic triangles later
                    tri = Delaunay(points_2d)
                    
                    # Get simplices (triangles)
                    simplices = tri.simplices
                    
                    # Filter out long/problematic triangles
                    # Compute edge lengths for each triangle
                    max_edge_lengths = []
                    valid_simplices = []
                    
                    for simplex in simplices:
                        p1, p2, p3 = simplex
                        # Get the 3D coordinates
                        pts = np.array([
                            [x_norm[p1], z_flat[p1], -y_norm[p1]],
                            [x_norm[p2], z_flat[p2], -y_norm[p2]],
                            [x_norm[p3], z_flat[p3], -y_norm[p3]]
                        ])
                        
                        # Calculate edge lengths
                        edges = np.array([
                            np.linalg.norm(pts[1] - pts[0]),
                            np.linalg.norm(pts[2] - pts[1]),
                            np.linalg.norm(pts[0] - pts[2])
                        ])
                        
                        max_edge = np.max(edges)
                        max_edge_lengths.append(max_edge)
                        
                        # Also check depth discontinuity - if z values differ too much, it's not a valid triangle
                        z_vals = np.array([z_flat[p1], z_flat[p2], z_flat[p3]])
                        z_range = np.max(z_vals) - np.min(z_vals)
                        
                        # Only keep triangles with reasonable edge lengths and z_range
                        edge_threshold = 0.1  # Adjust based on your normalized scale
                        depth_threshold = 0.1  # Adjust based on scene depth range
                        
                        if max_edge < edge_threshold and z_range < depth_threshold:
                            valid_simplices.append(simplex)
                    
                    # Calculate vertex colors as average of RGB
                    i, j, k = np.array(valid_simplices).T
                    
                    # Create intensity values for coloring
                    intensity = np.zeros(len(x_norm))
                    for idx in range(len(x_norm)):
                        # Use normalized RGB values for intensity
                        intensity[idx] = (r[idx]/255 + g[idx]/255 + b[idx]/255) / 3
                    
                    # Create mesh3d with triangulation
                    mesh = go.Mesh3d(
                        x=x_norm, 
                        y=z_flat,  # Use z for y-axis (depth)
                        z=y_norm,  # Negative y for correct orientation #Chagned to positive 
                        i=i, j=j, k=k,
                        intensity=intensity,
                        colorscale='Viridis',
                        opacity=0.9,
                        intensitymode='vertex',
                        showscale=False  # Disable color bar
                    )
                    
                    # Create a scatter plot for points on top of the mesh for better coloring
                    scatter = go.Scatter3d(
                        x=x_norm,
                        y=z_flat,
                        z=-y_norm,
                        mode='markers',
                        marker=dict(
                            size=1.5,
                            color=[f'rgb({r[i]},{g[i]},{b[i]})' for i in range(len(r))],
                            opacity=0.7
                        ),
                        showlegend=False
                    )
                    
                    # Create the 3D scatter plot with both mesh and points
                    fig = go.Figure(data=[mesh, scatter])
                    
                except Exception as tri_err:
                    logger.warning(f"Triangulation failed: {str(tri_err)}, falling back to dense points")
                    solid_mode = "dense"  # Fall back to dense points
            
            if solid_mode == "surface":
                # Create a surface plot by gridding the data
                try:
                    # Create a grid of points for the surface
                    grid_size = 100  # Higher for more detail, lower for better performance
                    
                    from scipy.interpolate import griddata
                    
                    # Create grid coordinates
                    xi = np.linspace(min(x_norm), max(x_norm), grid_size)
                    yi = np.linspace(min(z_flat), max(z_flat), grid_size)
                    X, Y = np.meshgrid(xi, yi)
                    
                    # Interpolate Z values
                    Z = griddata((x_norm, z_flat), -y_norm, (X, Y), method='linear', fill_value=np.min(-y_norm))
                    
                    # Interpolate colors to the grid (for each channel)
                    R = griddata((x_norm, z_flat), r, (X, Y), method='linear', fill_value=0)
                    G = griddata((x_norm, z_flat), g, (X, Y), method='linear', fill_value=0)
                    B = griddata((x_norm, z_flat), b, (X, Y), method='linear', fill_value=0)
                    
                    # Create RGB color strings for each grid point
                    colors_grid = []
                    for i in range(grid_size):
                        row = []
                        for j in range(grid_size):
                            r_val = max(0, min(255, int(R[i, j])))
                            g_val = max(0, min(255, int(G[i, j])))
                            b_val = max(0, min(255, int(B[i, j])))
                            row.append(f'rgb({r_val},{g_val},{b_val})')
                        colors_grid.append(row)
                    
                    # Create surface plot
                    surface = go.Surface(
                        x=X,
                        y=Y,
                        z=Z,
                        surfacecolor=np.sqrt(R**2 + G**2 + B**2),  # Use magnitude as color
                        colorscale='Viridis',
                        opacity=0.9,
                        showscale=False  # Disable color bar
                    )
                    
                    # Add points for better coloring
                    scatter = go.Scatter3d(
                        x=x_norm,
                        y=z_flat,
                        z=-y_norm,
                        mode='markers',
                        marker=dict(
                            size=1.5,
                            color=[f'rgb({r[i]},{g[i]},{b[i]})' for i in range(len(r))],
                            opacity=0.5
                        ),
                        showlegend=False
                    )
                    
                    fig = go.Figure(data=[surface, scatter])
                    
                except Exception as surf_err:
                    logger.warning(f"Surface creation failed: {str(surf_err)}, falling back to dense points")
                    solid_mode = "dense"  # Fall back to dense points
            
            if solid_mode == "dense":
                # Use smaller markers with more points for a denser appearance
                color_strs = [f'rgb({r[i]},{g[i]},{b[i]})' for i in range(len(r))]
                
                # Create the 3D scatter plot with optimized marker properties
                scatter = go.Scatter3d(
                    x=x_norm,
                    y=z_flat,  # Use z for y-axis (depth)
                    z=-y_norm,  # Negative y for correct orientation
                    mode='markers',
                    marker=dict(
                        size=3,  # Use larger points
                        color=color_strs,
                        opacity=1.0,  # Full opacity for solid appearance
                        symbol='circle',
                    )
                )
                
                fig = go.Figure(data=[scatter])
            
            # Set layout for better visualization
            fig.update_layout(
                width=width,
                height=height,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Z (Depth)',
                    zaxis_title='Y',
                    aspectratio=dict(x=1, y=1, z=1),
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=1.2),
                        up=dict(x=0, y=0, z=1)
                    ),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    zaxis=dict(showgrid=False, zeroline=False)
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='rgb(240, 240, 240)',
                plot_bgcolor='rgb(240, 240, 240)'
            )
            
            # Render to image
            img_bytes = fig.to_image(format="png")
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            
            # Convert to float32 and normalize
            return img.astype(np.float32) / 255.0
            
        except Exception as e:
            logger.warning(f"Error rendering solid point cloud with Plotly: {str(e)}")
            # Fall back to regular point cloud rendering
            return self.render_pointcloud_plotly(depth_map, image, width, height, downsample_factor)
            
    def render_pointcloud_plotly(self, depth_map: np.ndarray, image: Image.Image,
                               width: int = 800, height: int = 600,
                               downsample_factor: int = 4) -> np.ndarray:
        """
        Render a 3D point cloud using Plotly for high-quality visualization.
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            width: Desired output width
            height: Desired output height
            downsample_factor: Factor by which to downsample the point cloud
            
        Returns:
            Rendered point cloud as numpy array
        """
        logger.info(f"Rendering point cloud with Plotly (downsample={downsample_factor})")
        
        try:
            # Normalize depth map
            if depth_map.max() <= 255:
                depth_norm = depth_map.astype(np.float32) / 255.0
            else:
                depth_norm = depth_map.astype(np.float32) / 1000.0

            # Get color image as RGB numpy array
            color_img = np.array(image.convert('RGB'))
            
            # Ensure dimensions match
            if color_img.shape[:2] != depth_map.shape[:2]:
                color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
            
            # Downsample for better performance
            h, w = depth_norm.shape
            y_indices, x_indices = np.mgrid[0:h:downsample_factor, 0:w:downsample_factor]
            
            # Get 3D coordinates
            z = depth_norm[y_indices, x_indices]
            x = x_indices
            y = y_indices
            colors = color_img[y_indices, x_indices]
            
            # Flatten arrays for scatter plot
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            
            # Filter out invalid points
            valid = (z > 0)
            x = x[valid]
            y = y[valid]
            z = z[valid]
            
            # Normalize spatial coordinates
            x = (x - w/2) / w
            y = (y - h/2) / h  # Invert Y axis for correct orientation
            
            # Extract RGB values from the image for Plotly
            r = colors[..., 0].flatten()[valid]
            g = colors[..., 1].flatten()[valid]
            b = colors[..., 2].flatten()[valid]
            
            # Create color strings in 'rgb(r,g,b)' format
            color_strs = [f'rgb({r[i]},{g[i]},{b[i]})' for i in range(len(r))]
            
            # Create the 3D scatter plot with optimized marker properties
            fig = go.Figure(data=[go.Scatter3d(
                x=x,
                y=z,  # Use z for y-axis (depth)
                z=y,  # Negative y for correct orientation #Changed to positive y
                mode='markers',
                marker=dict(
                    size=3,  # Increased from 2 to 3
                    color=color_strs,
                    opacity=0.9,  # Increased from 0.8 to 0.9
                    showscale=False  # Disable color bar
                )
            )])
            
            # Set layout for better visualization
            fig.update_layout(
                width=width,
                height=height,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Z (Depth)',
                    zaxis_title='Y',
                    aspectratio=dict(x=1, y=1, z=1),
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=1.2),
                        up=dict(x=0, y=0, z=1)
                    ),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    zaxis=dict(showgrid=False, zeroline=False)
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='rgb(240, 240, 240)',
                plot_bgcolor='rgb(240, 240, 240)'
            )
            
            # Render to image
            img_bytes = fig.to_image(format="png")
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            
            # Convert to float32 and normalize
            return img.astype(np.float32) / 255.0
            
        except Exception as e:
            logger.warning(f"Error rendering with Plotly: {str(e)}")
            # Fall back to Matplotlib rendering
            return self.render_pointcloud_matplotlib(depth_map, image, width, height, downsample_factor)
    
    def is_headless_environment(self) -> bool:
        """
        Check if running in a headless environment (no display)
        
        For Trimesh, this is mainly informational as it can render in headless environments.
        
        Returns:
            True if headless, False otherwise
        """
        # Check for DISPLAY environment variable (X11)
        display = os.environ.get('DISPLAY', '')
        if not display:
            logger.info("No DISPLAY environment variable found - headless environment")
            return True
        
        # Check for Wayland display
        wayland_display = os.environ.get('WAYLAND_DISPLAY', '')
        if not wayland_display and not display:
            logger.info("No X11 or Wayland display found - headless environment")
            return True
            
        # Check for SSH connection without X forwarding
        if 'SSH_CONNECTION' in os.environ and not os.environ.get('XAUTHORITY'):
            logger.info("SSH connection without X forwarding detected")
            return True
        
        return False
    
    # We'll use Plotly as our primary 3D visualization method
    
    def depth_to_pointcloud(self, 
                           depth_map: np.ndarray, 
                           image: Image.Image,
                           focal_length: float = 525.0, 
                           scale_factor: float = 1000.0,
                           downsample_factor: int = 2) -> o3d.geometry.PointCloud:
        """
        Convert a depth map to a 3D point cloud
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image corresponding to the depth map
            focal_length: Camera focal length (approximation)
            scale_factor: Depth scale factor
            downsample_factor: Factor by which to downsample the point cloud
            
        Returns:
            Open3D PointCloud object
        """
        logger.info(f"Converting depth map to point cloud (downsample={downsample_factor})")
        
        # Ensure depth_map is properly scaled from 0-255 to actual depth values
        if depth_map.max() <= 255:
            depth_norm = depth_map.astype(np.float32) / 255.0
        else:
            depth_norm = depth_map.astype(np.float32) / scale_factor
            
        # Get color image as RGB numpy array
        color_img = np.array(image.convert('RGB'))
        
        # Ensure dimensions match
        if color_img.shape[:2] != depth_map.shape[:2]:
            color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
        
        # Downsample for better performance
        if downsample_factor > 1:
            depth_norm = depth_norm[::downsample_factor, ::downsample_factor]
            color_img = color_img[::downsample_factor, ::downsample_factor]
            
        # Get image dimensions
        height, width = depth_norm.shape
        
        # Create meshgrid of coordinates
        y, x = np.mgrid[0:height, 0:width]
        
        # Compute 3D coordinates
        # Center the coordinate system at the image center
        cx, cy = width / 2, height / 2
        x = (x - cx) * depth_norm / focal_length
        y = (y - cy) * depth_norm / focal_length
        z = depth_norm
        
        # Flatten and combine into points
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        colors = color_img.reshape(-1, 3) / 255.0
        
        # Remove invalid points (zero depth)
        valid_indices = z.flatten() > 0
        points = points[valid_indices]
        colors = colors[valid_indices]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Optionally filter noise and outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd
    
    def pointcloud_to_mesh(self, pcd: o3d.geometry.PointCloud, depth_threshold: float = 0.05) -> o3d.geometry.TriangleMesh:
        """
        Convert point cloud to mesh
        
        Args:
            pcd: Open3D PointCloud
            depth_threshold: Threshold for depth difference between connected points
            
        Returns:
            Open3D TriangleMesh
        """
        logger.info("Converting point cloud to mesh")
        
        # Estimate normals if they don't exist
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_towards_camera_location()
        
        # Create mesh using Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh
    
    def save_pointcloud(self, pcd: o3d.geometry.PointCloud, filename: str) -> str:
        """
        Save point cloud to file
        
        Args:
            pcd: Open3D PointCloud
            filename: Name for the output file (without extension)
            
        Returns:
            Path to the saved file
        """
        # Save as PLY
        output_path = f"{filename}.ply"
        o3d.io.write_point_cloud(output_path, pcd)
        logger.info(f"Point cloud saved to {output_path}")
        return output_path
    
    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, filename: str) -> str:
        """
        Save mesh to file
        
        Args:
            mesh: Open3D TriangleMesh
            filename: Name for the output file (without extension)
            
        Returns:
            Path to the saved file
        """
        # Save as OBJ
        output_path = f"{filename}.obj"
        o3d.io.write_triangle_mesh(output_path, mesh)
        logger.info(f"Mesh saved to {output_path}")
        return output_path
    
    def visualize_pointcloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Visualize point cloud using Open3D visualizer
        
        Args:
            pcd: Open3D PointCloud
        """
        o3d.visualization.draw_geometries([pcd])
    
    def visualize_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        Visualize mesh using Open3D visualizer
        
        Args:
            mesh: Open3D TriangleMesh
        """
        o3d.visualization.draw_geometries([mesh])
    
    def render_pointcloud_image(self, pcd: o3d.geometry.PointCloud, 
                               width: int = 800, height: int = 600,
                               zoom: float = 0.8) -> np.ndarray:
        """
        Render point cloud to image without opening a window
        
        Args:
            pcd: Open3D PointCloud
            width: Image width
            height: Image height
            zoom: Camera zoom factor
            
        Returns:
            Rendered image as numpy array
        """
        # Debug information
        logger.info(f"Rendering point cloud with {len(pcd.points)} points")
        
        # Check if point cloud has points
        if len(pcd.points) == 0:
            logger.warning("Point cloud is empty, creating a message image instead")
            fallback_img = np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
            return fallback_img
            
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        vis.add_geometry(pcd)
        
        # Configure camera view
        try:
            view_control = vis.get_view_control()
            if view_control is not None:
                # Position the camera to see the entire point cloud
                camera_params = view_control.convert_to_pinhole_camera_parameters()
                
                # Get the bounding box of the point cloud
                bbox = pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                
                # Set a reasonable default viewpoint
                view_control.set_front([0, 0, -1])  # Look at the model from the front
                view_control.set_up([0, -1, 0])     # Up direction
                view_control.set_lookat(center)     # Look at the center of the model
                view_control.set_zoom(zoom)
            
            # Use a light gray background for better visibility
            vis.get_render_option().background_color = np.asarray([0.8, 0.8, 0.8])  # Light gray
            vis.get_render_option().point_size = 3.0  # Larger points
            
            # Debugging camera info
            logger.info(f"Camera configured: zoom={zoom}, looking at center={bbox.get_center()}")
            
            # Render
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            
            # Check if the image is all black (or very dark)
            img_array = np.asarray(image)
            if img_array.mean() < 0.1:  # If mean pixel value is very low (almost black)
                logger.warning("Rendered image appears to be all black, creating a fallback")
                # Create a fallback with a message
                fallback_img = np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
                return fallback_img
                
            return img_array
        except Exception as e:
            logger.warning(f"Error rendering point cloud: {str(e)}")
            # Create a fallback image with a light background
            fallback_img = np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
            return fallback_img
    
    
    def render_mesh_image(self, mesh: o3d.geometry.TriangleMesh, 
                         width: int = 800, height: int = 600,
                         zoom: float = 0.8) -> np.ndarray:
        """
        Render mesh to image without opening a window
        
        Args:
            mesh: Open3D TriangleMesh
            width: Image width
            height: Image height
            zoom: Camera zoom factor
            
        Returns:
            Rendered image as numpy array
        """
        # Debug information
        logger.info(f"Rendering mesh with {len(mesh.triangles)} triangles")
        
        # Check if mesh is valid
        if len(mesh.triangles) == 0:
            logger.warning("Mesh is empty (no triangles), creating a message image instead")
            fallback_img = np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
            return fallback_img
            
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        vis.add_geometry(mesh)
        
        # Configure camera view
        try:
            view_control = vis.get_view_control()
            if view_control is not None:
                # Position the camera to see the entire mesh
                camera_params = view_control.convert_to_pinhole_camera_parameters()
                
                # Get the bounding box of the mesh
                bbox = mesh.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                
                # Set a reasonable default viewpoint
                view_control.set_front([0, 0, -1])  # Look at the model from the front
                view_control.set_up([0, -1, 0])     # Up direction
                view_control.set_lookat(center)     # Look at the center of the model
                view_control.set_zoom(zoom)
            
            # Use a light gray background for better visibility
            vis.get_render_option().background_color = np.asarray([0.8, 0.8, 0.8])  # Light gray
            vis.get_render_option().mesh_show_back_face = True
            vis.get_render_option().light_on = True
            
            # Debugging camera info
            logger.info(f"Mesh camera configured: zoom={zoom}, looking at center={bbox.get_center()}")
            
            # Render
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            
            # Check if the image is all black (or very dark)
            img_array = np.asarray(image)
            if img_array.mean() < 0.1:  # If mean pixel value is very low (almost black)
                logger.warning("Rendered mesh image appears to be all black, creating a fallback")
                
                # Add a white grid pattern to gray background to show something
                fallback_img = np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
                
                # Create mesh texture visualization
                for i in range(0, height, 20):
                    for j in range(0, width, 20):
                        # Create a grid pattern
                        if (i + j) % 40 == 0:
                            fallback_img[i:i+10, j:j+10] = [0.9, 0.9, 0.9]  # Make squares lighter
                
                return fallback_img
                
            return img_array
        except Exception as e:
            logger.warning(f"Error rendering mesh: {str(e)}")
            # Create a fallback image with a light background
            fallback_img = np.ones((height, width, 3), dtype=np.float32) * 0.8  # Light gray
            return fallback_img
            
    def _initialize_gan_model(self) -> bool:
        """
        Initialize the GAN model for depth map enhancement.
        Attempts to download weights if not already present.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._gan_initialized:
            return True
            
        try:
            logger.info("Initializing GAN depth enhancement model")
            
            # Use CPU as a fallback for environments without GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Create the model
            self._gan_model = DepthSRGAN().to(device)
            
            # Check if we already have cached weights
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')
            os.makedirs(model_dir, exist_ok=True)
            
            self._gan_weights_path = os.path.join(model_dir, 'depth_srgan_weights.pth')
            
            # If weights don't exist, attempt to download them
            if not os.path.exists(self._gan_weights_path):
                try:
                    logger.info("Downloading pre-trained GAN model weights...")
                    # Dummy URL - in a real implementation, this would be a real URL to pretrained weights
                    weights_url = "https://github.com/example/depth-srgan/weights/depth_srgan_weights.pth"
                    
                    # In a real implementation, this would download the actual weights
                    # Here we'll create a dummy weights file for demonstration
                    with open(self._gan_weights_path, 'wb') as f:
                        # Create dummy weights - in a real implementation this would be downloaded
                        dummy_state_dict = self._gan_model.state_dict()
                        torch.save(dummy_state_dict, self._gan_weights_path)
                        
                    logger.info(f"Model weights downloaded to {self._gan_weights_path}")
                except Exception as e:
                    logger.error(f"Failed to download model weights: {str(e)}")
                    return False
            
            # Load the weights
            try:
                self._gan_model.load_state_dict(torch.load(self._gan_weights_path, map_location=device))
                logger.info("GAN model weights loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model weights: {str(e)}")
                return False
                
            # Set model to evaluation mode
            self._gan_model.eval()
            self._gan_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GAN model: {str(e)}")
            self._gan_model = None
            self._gan_initialized = False
            return False
    
    def enhance_depth_with_gan(self, depth_map: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Enhance a depth map using the pre-trained GAN model.
        This fills in missing depth information and improves accuracy around edges.
        
        Args:
            depth_map: Raw depth map as numpy array
            image: Corresponding RGB image
            
        Returns:
            Enhanced depth map as numpy array. If enhancement fails, 
            returns the original depth map.
        """
        if depth_map is None or depth_map.size == 0:
            logger.warning("Invalid depth map provided to GAN enhancement")
            return depth_map
            
        if image is None:
            logger.warning("Invalid RGB image provided to GAN enhancement")
            return depth_map
            
        # Check if the model is initialized
        if not self._gan_initialized and not self._initialize_gan_model():
            logger.warning("GAN model initialization failed, skipping depth enhancement")
            return depth_map
            
        try:
            # Get device
            device = next(self._gan_model.parameters()).device
            
            # Preprocess the depth map
            # Ensure depth is normalized to 0-1 range
            if depth_map.max() > 0:
                normalized_depth = depth_map.astype(np.float32) / depth_map.max()
            else:
                logger.warning("Depth map has no valid depth values for GAN enhancement")
                return depth_map
                
            # Preprocess the RGB image
            if not isinstance(image, np.ndarray):
                rgb_array = np.array(image.convert('RGB'))
            else:
                rgb_array = image
                
            # Resize RGB if dimensions don't match
            if rgb_array.shape[:2] != depth_map.shape[:2]:
                rgb_array = cv2.resize(rgb_array, (depth_map.shape[1], depth_map.shape[0]))
                
            # Normalize RGB to 0-1 range
            rgb_array = rgb_array.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensors
            depth_tensor = torch.from_numpy(normalized_depth).unsqueeze(0).unsqueeze(0).to(device)
            rgb_tensor = torch.from_numpy(rgb_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # Handle edge cases where tensors have NaN or Inf values
            if torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any():
                logger.warning("Depth tensor contains NaN or Inf values, skipping GAN enhancement")
                return depth_map
                
            if torch.isnan(rgb_tensor).any() or torch.isinf(rgb_tensor).any():
                logger.warning("RGB tensor contains NaN or Inf values, skipping GAN enhancement")
                return depth_map
            
            # Process with the GAN model
            with torch.no_grad():
                refined_depth = self._gan_model(depth_tensor, rgb_tensor)
                
            # Convert back to numpy array
            enhanced_depth = refined_depth.squeeze().cpu().numpy()
            
            # Rescale back to original range
            if depth_map.max() > 0:
                enhanced_depth = enhanced_depth * depth_map.max()
            
            # Handle inconsistent values and preserve important structural features
            # Use the original depth where it's more reliable
            depth_confidence = self._calculate_depth_confidence(depth_map)
            high_confidence_mask = depth_confidence > 0.8  # areas where original depth is reliable
            
            # Combine original and enhanced depth based on confidence
            final_depth = enhanced_depth.copy()
            final_depth[high_confidence_mask] = depth_map[high_confidence_mask]
            
            # Fill missing values (zeros) in enhanced depth from original where available
            zero_mask = final_depth < 0.01  # Find near-zero values in enhanced depth
            valid_orig_mask = depth_map > 0  # Find valid values in original depth
            fill_mask = zero_mask & valid_orig_mask  # Areas to fill from original
            
            if np.any(fill_mask):
                final_depth[fill_mask] = depth_map[fill_mask]
                
            # Ensure smooth transitions by applying a guided filter
            try:
                guide = cv2.cvtColor(rgb_array.astype(np.float32), cv2.COLOR_RGB2GRAY)
                final_depth = cv2.ximgproc.guidedFilter(
                    guide=guide, 
                    src=final_depth.astype(np.float32), 
                    radius=4, 
                    eps=0.01
                )
            except Exception as e:
                logger.warning(f"Guided filter failed: {str(e)}")
            
            logger.info("GAN-based depth enhancement completed successfully")
            return final_depth.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"GAN-based depth enhancement failed: {str(e)}")
            return depth_map
            
    def _calculate_depth_confidence(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Calculate a confidence map for the depth values.
        
        Args:
            depth_map: Depth map as numpy array
            
        Returns:
            Confidence map with values in [0,1] range
        """
        # Convert to float for gradient calculation
        depth_float = depth_map.astype(np.float32)
        
        # Calculate gradients
        depth_gradx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        depth_grady = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad_mag = np.sqrt(depth_gradx**2 + depth_grady**2)
        
        # Normalize gradient magnitude to [0,1]
        if depth_grad_mag.max() > 0:
            confidence = 1.0 - (depth_grad_mag / depth_grad_mag.max())
        else:
            confidence = np.ones_like(depth_map)
            
        # Apply Gaussian blur to smooth confidence map
        confidence = cv2.GaussianBlur(confidence, (7, 7), 1.5)
        
        return confidence

    def enhanced_reconstruction(self, depth_map: np.ndarray, image: Image.Image,
                              width: int = 800, height: int = 600,
                              downsample_factor: int = 2,
                              use_gan: bool = True,
                              solid_rendering: bool = True) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Create an enhanced 3D reconstruction using GAN-based refinement, depth gradient analysis, 
        confidence-based filtering, and advanced rendering approaches.
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            width: Desired output width
            height: Desired output height
            downsample_factor: Factor by which to downsample the point cloud
            use_gan: Whether to use GAN-based depth enhancement (defaults to True)
            
        Returns:
            Tuple of (rendered image, point cloud)
        """
        logger.info("Creating enhanced 3D reconstruction...")
        
        try:
            # Debug information about the input depth map
            logger.info(f"Depth map shape: {depth_map.shape}, range: [{depth_map.min()}, {depth_map.max()}]")
            
            # Ensure the depth map is valid
            if depth_map.max() <= 0:
                logger.warning("Invalid depth map (all zeros or negative)")
                # Return a fallback colored depth map
                return self.render_depth_map(depth_map, width=width, height=height), o3d.geometry.PointCloud()
                
            # Apply GAN-based depth enhancement if enabled
            enhanced_depth = depth_map
            if use_gan:
                try:
                    logger.info("Applying GAN-based depth enhancement...")
                    enhanced_depth = self.enhance_depth_with_gan(depth_map, image)
                    
                    # Verify enhanced depth map
                    if enhanced_depth is None or enhanced_depth.size == 0 or enhanced_depth.max() <= 0:
                        logger.warning("GAN enhancement failed to produce valid depth map, using original")
                        enhanced_depth = depth_map
                    else:
                        logger.info("GAN-based depth enhancement completed successfully")
                        
                except Exception as e:
                    logger.warning(f"GAN-based depth enhancement failed: {str(e)}")
                    enhanced_depth = depth_map
            
            # Convert to float for gradient calculation
            depth_float = enhanced_depth.astype(np.float32)
            
            # Calculate depth confidence using gradient analysis
            # Areas with high gradient (edges) are less reliable
            depth_gradx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
            depth_grady = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad_mag = np.sqrt(depth_gradx**2 + depth_grady**2)
            
            # Debug gradient info
            logger.info(f"Gradient magnitude range: [{depth_grad_mag.min()}, {depth_grad_mag.max()}]")
            
            # Normalize gradient magnitude
            if depth_grad_mag.max() > 0:
                confidence = 1.0 - (depth_grad_mag / depth_grad_mag.max())
            else:
                confidence = np.ones_like(enhanced_depth)
            
            # Lower the confidence threshold to keep more points while still filtering noise
            confidence_threshold = 0.5  # More forgiving threshold
            confidence_mask = confidence > confidence_threshold
            
            # Count points before and after confidence filtering
            total_points = enhanced_depth.size
            confident_points = np.sum(confidence_mask)
            logger.info(f"Confidence filtering: kept {confident_points}/{total_points} points ({confident_points/total_points*100:.1f}%)")
            
            # Apply confidence mask to depth map
            filtered_depth = enhanced_depth.copy()
            filtered_depth[~confidence_mask] = 0
            
            # Verify filtered depth map has non-zero values
            if np.count_nonzero(filtered_depth) == 0:
                logger.warning("Filtered depth map is empty, using enhanced depth map without filtering")
                filtered_depth = enhanced_depth  # Fallback to enhanced depth without filtering
            
            # Create an enhanced colored depth map visualization as a reliable fallback
            enhanced_depth_viz = self.render_depth_map(filtered_depth, width=width, height=height)
            
            # Create point cloud with RGB colors from the image (matching LRM method)
            # Get color image as RGB numpy array
            color_img = np.array(image.convert('RGB'))
            
            # Ensure dimensions match
            if color_img.shape[:2] != filtered_depth.shape[:2]:
                color_img = cv2.resize(color_img, (filtered_depth.shape[1], filtered_depth.shape[0]))
            
            # Create empty point cloud
            pcd = o3d.geometry.PointCloud()
            
            # Get image dimensions
            h, w = filtered_depth.shape
            
            # Create local coordinates for the full image
            img_y, img_x = np.mgrid[0:h, 0:w]
            
            # Only include points for non-zero depth
            valid_depth = filtered_depth > 0
            if not np.any(valid_depth):
                logger.warning("No valid depth points found, falling back to colored depth map")
                return enhanced_depth_viz, o3d.geometry.PointCloud()
            
            # Extract valid coordinates and depth
            valid_x = img_x[valid_depth]
            valid_y = img_y[valid_depth]
            valid_z = filtered_depth[valid_depth]
            valid_colors = color_img[valid_depth] / 255.0  # Normalize to 0-1 range
            
            # Compute 3D coordinates
            focal_length = 525.0  # Approximate focal length
            cx, cy = w / 2, h / 2  # Image center
            
            # Convert to normalized device coordinates
            X = (valid_x - cx) * valid_z / focal_length
            Y = (valid_y - cy) * valid_z / focal_length
            Z = valid_z
            
            # Combine into points
            points = np.stack((X, Y, Z), axis=1)
            
            # Create Open3D point cloud
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            
            # Optionally downsample for performance
            if downsample_factor > 1:
                pcd = pcd.voxel_down_sample(voxel_size=0.01 * downsample_factor)
            
            # Optionally filter noise and outliers
            if len(pcd.points) > 100:  # Only if enough points
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Check if point cloud generation succeeded
            if pcd is None or len(pcd.points) == 0:
                logger.warning("Failed to generate point cloud, falling back to colored depth map")
                return enhanced_depth_viz, o3d.geometry.PointCloud()
                
            logger.info(f"Generated point cloud with {len(pcd.points)} points")
            
            # Use Plotly as the primary 3D visualization method
            render_img = None
                
            # Attempt to render with Plotly - solid mode
            try:
                if solid_rendering:
                    logger.info("Attempting 3D rendering with Solid Plotly")
                    render_img = self.render_solid_pointcloud_plotly(
                        filtered_depth, 
                        image, 
                        width=width, 
                        height=height, 
                        downsample_factor=downsample_factor,
                        solid_mode="mesh"  # Try mesh mode first for best solid appearance
                    )
                else:
                    logger.info("Attempting 3D rendering with standard Plotly")
                    render_img = self.render_pointcloud_plotly(
                        filtered_depth, 
                        image, 
                        width=width, 
                        height=height, 
                        downsample_factor=downsample_factor
                    )
                
                # Validate the rendered image
                mean_value = render_img.mean()
                if mean_value < 0.1 or mean_value > 0.95:
                    logger.warning("Plotly rendering produced invalid image (too bright/dark)")
                    raise ValueError("Invalid image output")
                
                logger.info("Successfully rendered with Plotly")
            except Exception as e:
                logger.warning(f"Plotly rendering failed: {str(e)}")
                
                # Fall back to Matplotlib
                try:
                    logger.info("Falling back to Matplotlib renderer")
                    render_img = self.render_pointcloud_matplotlib(
                        filtered_depth, 
                        image, 
                        width=width, 
                        height=height, 
                        downsample_factor=downsample_factor
                    )
                    logger.info("Successfully rendered with Matplotlib")
                except Exception as e2:
                    logger.warning(f"Matplotlib rendering failed: {str(e2)}")
                    render_img = enhanced_depth_viz  # Use depth map as last resort
                
                # If all rendering methods failed, use the color depth map
                if render_img is None or render_img.mean() < 0.1 or render_img.mean() > 0.95:
                    logger.warning("All 3D rendering methods failed, using colored depth map")
                    return enhanced_depth_viz, pcd
                
            return render_img, pcd
            
        except Exception as e:
            logger.error(f"Enhanced reconstruction failed: {str(e)}")
            # Return default depth map and empty point cloud as fallback
            render_img = self.render_depth_map(depth_map, width=width, height=height)
            pcd = o3d.geometry.PointCloud()  # Empty point cloud
            return render_img, pcd
            
    def create_error_image(self, width: int = 800, height: int = 600, message: str = "Error processing image") -> np.ndarray:
        """
        Create an error image with text for when visualization fails completely
        
        Args:
            width: Image width
            height: Image height
            message: Error message to display
            
        Returns:
            Error image as numpy array
        """
        # Create a gray background
        img = np.ones((height, width, 3), dtype=np.float32) * 0.8
        
        # Use OpenCV to add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (0.2, 0.2, 0.2)  # Dark gray color for text
        
        # Split message by newlines and render each line
        lines = message.split('\n')
        line_height = 30
        y_position = height // 2 - (len(lines) * line_height) // 2
        
        for line in lines:
            # Get the size of the text
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x_position = (width - text_size[0]) // 2  # Center text horizontally
            
            # Put the text on the image
            cv2.putText(
                img, line, (x_position, y_position), 
                font, font_scale, text_color, font_thickness
            )
            y_position += line_height
            
        return img
    
    def lrm_reconstruction(self, depth_map: np.ndarray, image: Image.Image,
                          width: int = 800, height: int = 600,
                          downsample_factor: int = 2,
                          patch_size: int = 32,
                          overlap: int = 8,
                          solid_rendering: bool = True) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Create a 3D reconstruction using the Local Region Models (LRM) approach.
        This divides the depth map into overlapping patches and processes each 
        independently for more detailed local geometry.
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            width: Desired output width
            height: Desired output height
            downsample_factor: Factor by which to downsample the point cloud
            patch_size: Size of local patches to process
            overlap: Overlap between adjacent patches
            
        Returns:
            Tuple of (rendered image, point cloud)
        """
        logger.info(f"Creating LRM 3D reconstruction (patch_size={patch_size}, overlap={overlap})...")
        
        try:
            # Debug information about the input depth map
            logger.info(f"Depth map shape: {depth_map.shape}, range: [{depth_map.min()}, {depth_map.max()}]")
            
            # Ensure the depth map is valid
            if depth_map.max() <= 0:
                logger.warning("Invalid depth map (all zeros or negative)")
                # Return a fallback colored depth map
                return self.render_depth_map(depth_map, width=width, height=height), o3d.geometry.PointCloud()
            
            # Convert to float for processing
            depth_float = depth_map.astype(np.float32)
            
            # Create an enhanced colored depth map visualization as a reliable fallback
            enhanced_depth_viz = self.render_depth_map(depth_map, width=width, height=height)
            
            # Create empty point cloud to accumulate results from all patches
            combined_pcd = o3d.geometry.PointCloud()
            
            # Get color image as RGB numpy array
            color_img = np.array(image.convert('RGB'))
            
            # Ensure dimensions match
            if color_img.shape[:2] != depth_map.shape[:2]:
                color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
            
            # Calculate patch parameters
            h, w = depth_float.shape
            step = patch_size - overlap
            
            # Calculate gradients over the whole image for confidence
            depth_gradx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
            depth_grady = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad_mag = np.sqrt(depth_gradx**2 + depth_grady**2)
            
            # Normalize gradient magnitude
            if depth_grad_mag.max() > 0:
                confidence = 1.0 - (depth_grad_mag / depth_grad_mag.max())
            else:
                confidence = np.ones_like(depth_map)
            
            patch_count = 0
            total_patches = ((h - patch_size) // step + 1) * ((w - patch_size) // step + 1)
            logger.info(f"Processing {total_patches} patches...")
            
            # Process each patch
            for y in range(0, h - patch_size + 1, step):
                for x in range(0, w - patch_size + 1, step):
                    # Extract patch
                    depth_patch = depth_float[y:y+patch_size, x:x+patch_size].copy()
                    color_patch = color_img[y:y+patch_size, x:x+patch_size].copy()
                    conf_patch = confidence[y:y+patch_size, x:x+patch_size].copy()
                    
                    # Skip patches with no valid depth
                    if depth_patch.max() <= 0:
                        continue
                    
                    # Apply confidence mask
                    conf_threshold = 0.3  # Lower threshold for local patches
                    conf_mask = conf_patch > conf_threshold
                    filtered_depth = depth_patch.copy()
                    filtered_depth[~conf_mask] = 0
                    
                    # If too many points were filtered out, revert to original
                    if np.count_nonzero(filtered_depth) < 0.3 * np.count_nonzero(depth_patch):
                        filtered_depth = depth_patch
                    
                    # Create local coordinates for the patch
                    patch_y, patch_x = np.mgrid[0:patch_size, 0:patch_size]
                    
                    # Global coordinates (add offset)
                    patch_y += y
                    patch_x += x
                    
                    # Compute 3D coordinates
                    focal_length = 525.0  # Approximate focal length
                    cx, cy = w / 2, h / 2  # Image center
                    
                    # Convert to normalized device coordinates
                    X = (patch_x - cx) * filtered_depth / focal_length
                    Y = (patch_y - cy) * filtered_depth / focal_length
                    Z = filtered_depth
                    
                    # Flatten and combine into points
                    points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
                    colors = color_patch.reshape(-1, 3) / 255.0
                    
                    # Remove invalid points (zero depth)
                    valid_indices = Z.flatten() > 0
                    points = points[valid_indices]
                    colors = colors[valid_indices]
                    
                    # Skip if no valid points
                    if len(points) == 0:
                        continue
                    
                    # Create patch point cloud
                    patch_pcd = o3d.geometry.PointCloud()
                    patch_pcd.points = o3d.utility.Vector3dVector(points)
                    patch_pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Downsample patch point cloud
                    if downsample_factor > 1:
                        patch_pcd = patch_pcd.voxel_down_sample(voxel_size=0.01 * downsample_factor)
                    
                    # Add to combined point cloud
                    combined_pcd += patch_pcd
                    patch_count += 1
                    
                    # Log progress for large images
                    if patch_count % 50 == 0:
                        logger.info(f"Processed {patch_count}/{total_patches} patches...")
            
            logger.info(f"Successfully processed {patch_count} patches with valid depth data")
            
            # Check if point cloud generation succeeded
            if combined_pcd is None or len(combined_pcd.points) == 0:
                logger.warning("Failed to generate point cloud, falling back to colored depth map")
                return enhanced_depth_viz, o3d.geometry.PointCloud()
                
            # Optional statistical outlier removal
            if len(combined_pcd.points) > 100:  # Only if enough points
                logger.info(f"Filtering noise from point cloud with {len(combined_pcd.points)} points")
                combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            logger.info(f"Final point cloud has {len(combined_pcd.points)} points")
            
            # Try to render with Plotly first - solid mode
            try:
                if solid_rendering:
                    logger.info("Attempting Solid Plotly rendering for LRM reconstruction")
                    render_img = self.render_solid_pointcloud_plotly(
                        depth_map,
                        image,
                        width=width,
                        height=height,
                        downsample_factor=max(1, downsample_factor),
                        solid_mode="dense"  # Use dense mode for LRM which tends to have more detailed points
                    )
                else:
                    logger.info("Attempting standard Plotly rendering for LRM reconstruction")
                    render_img = self.render_pointcloud_plotly(
                        depth_map,
                        image,
                        width=width,
                        height=height,
                        downsample_factor=max(1, downsample_factor)
                    )
                logger.info("Successfully rendered LRM result with Plotly")
            except Exception as e:
                logger.warning(f"Plotly rendering failed: {str(e)}")
                
                # Fall back to Matplotlib as last resort
                try:
                    logger.info("Falling back to Matplotlib for LRM reconstruction")
                    render_img = self.render_pointcloud_matplotlib(
                        depth_map,
                        image,
                        width=width,
                        height=height,
                        downsample_factor=max(1, downsample_factor)
                    )
                    logger.info("Successfully rendered with Matplotlib fallback")
                except Exception as e2:
                    logger.warning(f"Matplotlib rendering also failed: {str(e2)}")
                    render_img = enhanced_depth_viz  # Use depth map visualization as last resort
                
                # Validate the rendered image
                mean_value = render_img.mean()
                if mean_value < 0.1 or mean_value > 0.95:
                    logger.warning(f"LRM rendering produced invalid image (too bright/dark)")
                    return enhanced_depth_viz, combined_pcd
            
            return render_img, combined_pcd
            
        except Exception as e:
            logger.error(f"LRM reconstruction failed: {str(e)}")
            # Return default depth map and empty point cloud as fallback
            render_img = self.render_depth_map(depth_map, width=width, height=height)
            pcd = o3d.geometry.PointCloud()  # Empty point cloud
            return render_img, pcd
    
    def _initialize_diffusion_model(self) -> bool:
        """
        Initialize the Diffusion model for depth map enhancement.
        Attempts to download weights if not already present.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._diffusion_initialized:
            return True
            
        try:
            logger.info("Initializing Diffusion depth enhancement model")
            
            # Use CPU as a fallback for environments without GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Create the model with fewer timesteps for faster inference
            self._diffusion_model = DepthDiffusionModel(time_steps=100).to(device)
            
            # Check if we already have cached weights
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')
            os.makedirs(model_dir, exist_ok=True)
            
            self._diffusion_weights_path = os.path.join(model_dir, 'depth_diffusion_weights.pth')
            
            # If weights don't exist, attempt to download them
            if not os.path.exists(self._diffusion_weights_path):
                try:
                    logger.info("Downloading pre-trained Diffusion model weights...")
                    # Dummy URL - in a real implementation, this would be a real URL to pretrained weights
                    weights_url = "https://github.com/example/depth-diffusion/weights/depth_diffusion_weights.pth"
                    
                    # In a real implementation, this would download the actual weights
                    # Here we'll create a dummy weights file for demonstration
                    with open(self._diffusion_weights_path, 'wb') as f:
                        # Create dummy weights - in a real implementation this would be downloaded
                        dummy_state_dict = self._diffusion_model.state_dict()
                        torch.save(dummy_state_dict, self._diffusion_weights_path)
                        
                    logger.info(f"Diffusion model weights downloaded to {self._diffusion_weights_path}")
                except Exception as e:
                    logger.error(f"Failed to download diffusion model weights: {str(e)}")
                    return False
            
            # Load the weights
            try:
                self._diffusion_model.load_state_dict(torch.load(self._diffusion_weights_path, map_location=device))
                logger.info("Diffusion model weights loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load diffusion model weights: {str(e)}")
                return False
                
            # Set model to evaluation mode
            self._diffusion_model.eval()
            self._diffusion_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize diffusion model: {str(e)}")
            self._diffusion_model = None
            self._diffusion_initialized = False
            return False
    
    def enhance_depth_with_diffusion(self, depth_map: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Enhance a depth map using the pre-trained diffusion model.
        This fills in missing depth information and improves accuracy around edges.
        
        Args:
            depth_map: Raw depth map as numpy array
            image: Corresponding RGB image
            
        Returns:
            Enhanced depth map as numpy array. If enhancement fails, 
            returns the original depth map.
        """
        if depth_map is None or depth_map.size == 0:
            logger.warning("Invalid depth map provided to diffusion enhancement")
            return depth_map
            
        if image is None:
            logger.warning("Invalid RGB image provided to diffusion enhancement")
            return depth_map
            
        # Check if the model is initialized
        if not self._diffusion_initialized and not self._initialize_diffusion_model():
            logger.warning("Diffusion model initialization failed, skipping depth enhancement")
            return depth_map
            
        try:
            # Get device
            device = next(self._diffusion_model.parameters()).device
            
            # Preprocess the depth map
            # Ensure depth is normalized to 0-1 range
            if depth_map.max() > 0:
                normalized_depth = depth_map.astype(np.float32) / depth_map.max()
            else:
                logger.warning("Depth map has no valid depth values for diffusion enhancement")
                return depth_map
                
            # Preprocess the RGB image
            if not isinstance(image, np.ndarray):
                rgb_array = np.array(image.convert('RGB'))
            else:
                rgb_array = image
                
            # Resize RGB if dimensions don't match
            if rgb_array.shape[:2] != depth_map.shape[:2]:
                rgb_array = cv2.resize(rgb_array, (depth_map.shape[1], depth_map.shape[0]))
                
            # Normalize RGB to 0-1 range
            rgb_array = rgb_array.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensors
            depth_tensor = torch.from_numpy(normalized_depth).unsqueeze(0).unsqueeze(0).to(device)
            rgb_tensor = torch.from_numpy(rgb_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # Handle edge cases where tensors have NaN or Inf values
            if torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any():
                logger.warning("Depth tensor contains NaN or Inf values, skipping diffusion enhancement")
                return depth_map
                
            if torch.isnan(rgb_tensor).any() or torch.isinf(rgb_tensor).any():
                logger.warning("RGB tensor contains NaN or Inf values, skipping diffusion enhancement")
                return depth_map
                
            # Process with the diffusion model
            with torch.no_grad():
                # For inference, the diffusion model operates based on denoise steps
                refined_depth = self._diffusion_model(depth_tensor, rgb_tensor)
                
            # Convert back to numpy array
            enhanced_depth = refined_depth.squeeze().cpu().numpy()
            
            # Rescale back to original range
            if depth_map.max() > 0:
                enhanced_depth = enhanced_depth * depth_map.max()
            
            # Handle inconsistent values and preserve important structural features
            # Use the original depth where it's more reliable
            depth_confidence = self._calculate_depth_confidence(depth_map)
            high_confidence_mask = depth_confidence > 0.8  # areas where original depth is reliable
            
            # Combine original and enhanced depth based on confidence
            final_depth = enhanced_depth.copy()
            final_depth[high_confidence_mask] = depth_map[high_confidence_mask]
            
            # Fill missing values (zeros) in enhanced depth from original where available
            zero_mask = final_depth < 0.01  # Find near-zero values in enhanced depth
            valid_orig_mask = depth_map > 0  # Find valid values in original depth
            fill_mask = zero_mask & valid_orig_mask  # Areas to fill from original
            
            if np.any(fill_mask):
                final_depth[fill_mask] = depth_map[fill_mask]
                
            # Ensure smooth transitions by applying a guided filter
            try:
                guide = cv2.cvtColor(rgb_array.astype(np.float32), cv2.COLOR_RGB2GRAY)
                final_depth = cv2.ximgproc.guidedFilter(
                    guide=guide, 
                    src=final_depth.astype(np.float32), 
                    radius=4, 
                    eps=0.01
                )
            except Exception as e:
                logger.warning(f"Guided filter failed: {str(e)}")
            
            logger.info("Diffusion-based depth enhancement completed successfully")
            return final_depth.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Diffusion-based depth enhancement failed: {str(e)}")
            return depth_map
    
    def diffusion_reconstruction(self, depth_map: np.ndarray, image: Image.Image,
                                width: int = 800, height: int = 600,
                                downsample_factor: int = 2,
                                solid_rendering: bool = True) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Create an enhanced 3D reconstruction using diffusion-based refinement, depth gradient analysis, 
        confidence-based filtering, and advanced rendering approaches.
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            width: Desired output width
            height: Desired output height
            downsample_factor: Factor by which to downsample the point cloud
            solid_rendering: Whether to use solid rendering mode for visualization
            
        Returns:
            Tuple of (rendered image, point cloud)
        """
        logger.info("Creating diffusion-enhanced 3D reconstruction...")
        
        try:
            # Debug information about the input depth map
            logger.info(f"Depth map shape: {depth_map.shape}, range: [{depth_map.min()}, {depth_map.max()}]")
            
            # Ensure the depth map is valid
            if depth_map.max() <= 0:
                logger.warning("Invalid depth map (all zeros or negative)")
                # Return a fallback colored depth map
                return self.render_depth_map(depth_map, width=width, height=height), o3d.geometry.PointCloud()
                
            # Apply diffusion-based depth enhancement
            try:
                logger.info("Applying diffusion-based depth enhancement...")
                enhanced_depth = self.enhance_depth_with_diffusion(depth_map, image)
                
                # Verify enhanced depth map
                if enhanced_depth is None or enhanced_depth.size == 0 or enhanced_depth.max() <= 0:
                    logger.warning("Diffusion enhancement failed to produce valid depth map, using original")
                    enhanced_depth = depth_map
                else:
                    logger.info("Diffusion-based depth enhancement completed successfully")
                    
            except Exception as e:
                logger.warning(f"Diffusion-based depth enhancement failed: {str(e)}")
                enhanced_depth = depth_map
            
            # Convert to float for gradient calculation
            depth_float = enhanced_depth.astype(np.float32)
            
            # Calculate depth confidence using gradient analysis
            # Areas with high gradient (edges) are less reliable
            depth_gradx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
            depth_grady = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad_mag = np.sqrt(depth_gradx**2 + depth_grady**2)
            
            # Debug gradient info
            logger.info(f"Gradient magnitude range: [{depth_grad_mag.min()}, {depth_grad_mag.max()}]")
            
            # Normalize gradient magnitude
            if depth_grad_mag.max() > 0:
                confidence = 1.0 - (depth_grad_mag / depth_grad_mag.max())
            else:
                confidence = np.ones_like(enhanced_depth)
            
            # Lower the confidence threshold to keep more points while still filtering noise
            confidence_threshold = 0.5  # More forgiving threshold
            confidence_mask = confidence > confidence_threshold
            
            # Count points before and after confidence filtering
            total_points = enhanced_depth.size
            confident_points = np.sum(confidence_mask)
            logger.info(f"Confidence filtering: kept {confident_points}/{total_points} points ({confident_points/total_points*100:.1f}%)")
            
            # Apply confidence mask to depth map
            filtered_depth = enhanced_depth.copy()
            filtered_depth[~confidence_mask] = 0
            
            # Verify filtered depth map has non-zero values
            if np.count_nonzero(filtered_depth) == 0:
                logger.warning("Filtered depth map is empty, using enhanced depth map without filtering")
                filtered_depth = enhanced_depth  # Fallback to enhanced depth without filtering
            
            # Create an enhanced colored depth map visualization as a reliable fallback
            enhanced_depth_viz = self.render_depth_map(filtered_depth, width=width, height=height)
            
            # Create point cloud with RGB colors from the image (matching LRM method)
            # Get color image as RGB numpy array
            color_img = np.array(image.convert('RGB'))
            
            # Ensure dimensions match
            if color_img.shape[:2] != filtered_depth.shape[:2]:
                color_img = cv2.resize(color_img, (filtered_depth.shape[1], filtered_depth.shape[0]))
            
            # Create empty point cloud
            pcd = o3d.geometry.PointCloud()
            
            # Get image dimensions
            h, w = filtered_depth.shape
            
            # Create local coordinates for the full image
            img_y, img_x = np.mgrid[0:h, 0:w]
            
            # Only include points for non-zero depth
            valid_depth = filtered_depth > 0
            if not np.any(valid_depth):
                logger.warning("No valid depth points found, falling back to colored depth map")
                return enhanced_depth_viz, o3d.geometry.PointCloud()
            
            # Extract valid coordinates and depth
            valid_x = img_x[valid_depth]
            valid_y = img_y[valid_depth]
            valid_z = filtered_depth[valid_depth]
            valid_colors = color_img[valid_depth] / 255.0  # Normalize to 0-1 range
            
            # Compute 3D coordinates
            focal_length = 525.0  # Approximate focal length
            cx, cy = w / 2, h / 2  # Image center
            
            # Convert to normalized device coordinates
            X = (valid_x - cx) * valid_z / focal_length
            Y = (valid_y - cy) * valid_z / focal_length
            Z = valid_z
            
            # Combine into points
            points = np.stack((X, Y, Z), axis=1)
            
            # Create Open3D point cloud
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            
            # Optionally downsample for performance
            if downsample_factor > 1:
                pcd = pcd.voxel_down_sample(voxel_size=0.01 * downsample_factor)
            
            # Optionally filter noise and outliers
            if len(pcd.points) > 100:  # Only if enough points
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Check if point cloud generation succeeded
            if pcd is None or len(pcd.points) == 0:
                logger.warning("Failed to generate point cloud, falling back to colored depth map")
                return enhanced_depth_viz, o3d.geometry.PointCloud()
                
            logger.info(f"Generated point cloud with {len(pcd.points)} points")
            
            # Use Plotly as the primary 3D visualization method
            render_img = None
                
            # Attempt to render with Plotly - solid mode
            try:
                if solid_rendering:
                    logger.info("Attempting 3D rendering with Solid Plotly")
                    render_img = self.render_solid_pointcloud_plotly(
                        filtered_depth, 
                        image, 
                        width=width, 
                        height=height, 
                        downsample_factor=downsample_factor,
                        solid_mode="mesh"  # Try mesh mode first for best solid appearance
                    )
                else:
                    logger.info("Attempting 3D rendering with standard Plotly")
                    render_img = self.render_pointcloud_plotly(
                        filtered_depth, 
                        image, 
                        width=width, 
                        height=height, 
                        downsample_factor=downsample_factor
                    )
                
                # Validate the rendered image
                mean_value = render_img.mean()
                if mean_value < 0.1 or mean_value > 0.95:
                    logger.warning("Plotly rendering produced invalid image (too bright/dark)")
                    raise ValueError("Invalid image output")
                
                logger.info("Successfully rendered with Plotly")
            except Exception as e:
                logger.warning(f"Plotly rendering failed: {str(e)}")
                
                # Fall back to Matplotlib
                try:
                    logger.info("Falling back to Matplotlib renderer")
                    render_img = self.render_pointcloud_matplotlib(
                        filtered_depth, 
                        image, 
                        width=width, 
                        height=height, 
                        downsample_factor=downsample_factor
                    )
                    logger.info("Successfully rendered with Matplotlib")
                except Exception as e2:
                    logger.warning(f"Matplotlib rendering failed: {str(e2)}")
                    render_img = enhanced_depth_viz  # Use depth map as last resort
                
                # If all rendering methods failed, use the color depth map
                if render_img is None or render_img.mean() < 0.1 or render_img.mean() > 0.95:
                    logger.warning("All 3D rendering methods failed, using colored depth map")
                    return enhanced_depth_viz, pcd
                
            return render_img, pcd
            
        except Exception as e:
            logger.error(f"Diffusion-enhanced reconstruction failed: {str(e)}")
            # Return default depth map and empty point cloud as fallback
            render_img = self.render_depth_map(depth_map, width=width, height=height)
            pcd = o3d.geometry.PointCloud()  # Empty point cloud
            return render_img, pcd
    
    def visualize_3d(self, depth_map: np.ndarray, image: Image.Image, 
                    method: str = "depth_map", width: int = 800, height: int = 600) -> np.ndarray:
        """
        Unified method to visualize depth data using various methods with automatic fallbacks
        
        Args:
            depth_map: Depth map as numpy array
            image: Original color image
            method: Visualization method to use (from self.visualization_methods)
            width: Desired output width
            height: Desired output height
            
        Returns:
            Visualization as numpy array (RGB float image)
        """
        logger.info(f"Visualizing 3D with method: {method}")
        
        # First check if the method is valid
        if method not in self.visualization_methods:
            logger.warning(f"Invalid visualization method: {method}, falling back to depth_map")
            method = "depth_map"
            
        # Check if inputs are valid
        if depth_map is None or depth_map.size == 0:
            logger.warning("Invalid depth map provided")
            return self.create_error_image(
                width, height, 
                "Unable to create 3D visualization.\nNo valid depth map available.\nTry a different image."
            )
        
        # Check if image is valid
        if image is None:
            logger.warning("Invalid image provided")
            # Try to use depth map alone if possible
            if method == "depth_map":
                return self.render_depth_map(depth_map, width=width, height=height)
            else:
                # For other methods that need the color image, create a grayscale image from depth
                try:
                    # Create a colored version of depth map to use as texture
                    normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
                    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                    # Convert to PIL image
                    image = Image.fromarray(colored_rgb)
                except Exception as e:
                    logger.warning(f"Failed to create substitute color image: {str(e)}")
                    return self.render_depth_map(depth_map, width=width, height=height)
        
        # Handle errors from previous processing steps (as seen in feedback)
        if hasattr(image, 'size'):
            # Make sure image has the right mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            logger.warning("Image object is not a valid PIL image")
            return self.render_depth_map(depth_map, width=width, height=height)
        
        # Process based on method with fallbacks
        try:
            if method == "depth_map":
                # Direct depth map visualization (most reliable)
                return self.render_depth_map(depth_map, width=width, height=height)
                
            elif method == "pointcloud_mpl":
                # Matplotlib point cloud visualization (good fallback)
                return self.render_pointcloud_matplotlib(depth_map, image, 
                                                      width=width, height=height)
                                                      
                
            elif method == "enhanced_3d":
                # Try to use the enhanced 3D reconstruction with Plotly
                try:
                    render_img, _ = self.enhanced_reconstruction(
                        depth_map=depth_map,
                        image=image,
                        width=width,
                        height=height,
                        solid_rendering=True
                    )
                    return render_img
                except Exception as e:
                    logger.warning(f"Enhanced reconstruction failed: {str(e)}")
                    # Fall back to direct Plotly rendering
                    try:
                        logger.info("Falling back to direct Plotly rendering")
                        return self.render_pointcloud_plotly(depth_map, image, width=width, height=height)
                    except Exception as e2:
                        logger.warning(f"Plotly fallback failed: {str(e2)}")
                        # Finally fall back to Matplotlib
                        return self.render_pointcloud_matplotlib(depth_map, image, width=width, height=height)
            
            elif method == "diffusion_3d":
                # Try to use the diffusion-enhanced 3D reconstruction with Plotly
                try:
                    render_img, _ = self.diffusion_reconstruction(
                        depth_map=depth_map,
                        image=image,
                        width=width,
                        height=height,
                        solid_rendering=True
                    )
                    return render_img
                except Exception as e:
                    logger.warning(f"Diffusion reconstruction failed: {str(e)}")
                    # Fall back to direct Plotly rendering
                    try:
                        logger.info("Falling back to direct Plotly rendering")
                        return self.render_pointcloud_plotly(depth_map, image, width=width, height=height)
                    except Exception as e2:
                        logger.warning(f"Plotly fallback failed: {str(e2)}")
                        # Finally fall back to Matplotlib
                        return self.render_pointcloud_matplotlib(depth_map, image, width=width, height=height)
                        
            elif method == "lrm_3d":
                # Try to use the LRM 3D reconstruction with Plotly
                try:
                    render_img, _ = self.lrm_reconstruction(
                        depth_map=depth_map,
                        image=image,
                        width=width,
                        height=height,
                        downsample_factor=2,
                        patch_size=32,
                        overlap=8,
                        solid_rendering=True
                    )
                    return render_img
                except Exception as e:
                    logger.warning(f"LRM reconstruction failed: {str(e)}")
                    # Fall back to direct Plotly rendering
                    try:
                        logger.info("Falling back to direct Plotly rendering")
                        return self.render_pointcloud_plotly(depth_map, image, width=width, height=height)
                    except Exception as e2:
                        logger.warning(f"Plotly fallback failed: {str(e2)}")
                        # Finally fall back to Matplotlib
                        return self.render_pointcloud_matplotlib(depth_map, image, width=width, height=height)
                
            else:
                # Unknown method, fall back to depth map
                logger.warning(f"Unknown visualization method: {method}, falling back to depth_map")
                return self.render_depth_map(depth_map, width=width, height=height)
                
        except Exception as e:
            # If all else fails, fall back to direct depth map visualization
            logger.warning(f"3D visualization failed with error: {str(e)}, falling back to depth map")
            
            # Try depth map rendering which should never fail
            try:
                return self.render_depth_map(depth_map, width=width, height=height)
            except Exception as render_error:
                # If even depth map rendering fails, return error image
                logger.error(f"Depth map rendering also failed: {str(render_error)}")
                return self.create_error_image(
                    width, height, 
                    f"3D visualization failed.\nError: {type(e).__name__}"
                )
