# Interior Reimaginer

A project exploring interior design using AI to generate and optimize design solutions with depth-based 3D reconstruction capabilities.

## Features

- **Full Room Redesign**: Transform interior spaces with AI-powered style transfer
- **Style Explorer**: Visualize different design styles for your space
- **3D Reconstruction**: Generate 3D point clouds and meshes from interior images using optimized depth estimation
- **Diffusion-Enhanced Depth Maps**: High-quality depth maps enhanced with a lightweight diffusion model
- **User-Friendly Interface**: Simple Gradio UI with both local and Colab support

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CV_Project.git
cd CV_Project
```

2. Install dependencies:
```bash
pip install -r interior_reimaginer/requirements.txt
```

### Google Colab Setup

To run Interior Reimaginer on Google Colab with optimal performance:

> **IMPORTANT**: An A100 GPU or equivalent is recommended for optimal performance. The diffusion model and 3D reconstruction processes are computationally intensive and will perform significantly better on high-end GPUs.

1. Open our provided example notebook in Colab: [interior_reimaginer/examples/colab_3d_reconstruction.ipynb](https://colab.research.google.com/github/your-username/CV_Project/blob/main/interior_reimaginer/examples/colab_3d_reconstruction.ipynb)

2. Before running the notebook, select **Runtime > Change runtime type** and choose:
   - **GPU** for Hardware Accelerator
   - **High-RAM** for Runtime shape (if available)

3. The notebook will automatically:
   - Clone the repository
   - Install all required dependencies
   - Set up the environment for optimal performance

If you prefer to set up manually:

```python
# Clone the repository
!git clone https://github.com/your-username/CV_Project.git
%cd CV_Project

# Install dependencies
!pip install -r interior_reimaginer/requirements.txt

# Install Open3D for 3D reconstruction
!pip install open3d

# Import required modules
from interior_reimaginer.models.interior_reimaginer import InteriorReimaginer
from interior_reimaginer.models.reconstruction_3d import DepthReconstructor
from PIL import Image
import numpy as np

# Check GPU availability (recommended: A100)
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Usage

### Local Usage

Run the Gradio interface:

```bash
python interior_reimaginer/main.py
```

This will start the application and generate both a local URL (http://127.0.0.1:7860) and a public shareable link that you can use to access the interface from other devices. The shareable link will be displayed in the console.

If you don't want to create a public link, use:

```bash
python interior_reimaginer/main.py --share=False
```

### Google Colab Usage

For running in Google Colab, you can use our example notebook:

1. Upload `interior_reimaginer/examples/colab_3d_reconstruction.ipynb` to Google Colab
2. Run the cells to set up the environment and process your images

Alternatively, use the following code in your own notebook:

```python
# Initialize the models
reimaginer = InteriorReimaginer()
depth_reconstructor = DepthReconstructor()

# Load an example image
from google.colab import files
uploaded = files.upload()  # This will prompt to upload an image

import io
image_name = list(uploaded.keys())[0]
image = Image.open(io.BytesIO(uploaded[image_name]))

# Process the image
processed = reimaginer.image_processor.process_image(image)

# For 3D reconstruction
if processed.depth_map is not None:
    # Create a point cloud
    pcd = depth_reconstructor.depth_to_pointcloud(
        depth_map=processed.depth_map,
        image=image,
        downsample_factor=2  # Higher values for faster processing but less detail
    )
    
    # Visualize (in Colab, this will create an interactive viewer)
    depth_reconstructor.visualize_pointcloud(pcd)
    
    # Optionally convert to mesh
    mesh = depth_reconstructor.pointcloud_to_mesh(pcd)
    depth_reconstructor.visualize_mesh(mesh)
    
    # Save the models
    depth_reconstructor.save_pointcloud(pcd, "interior_pointcloud")
    depth_reconstructor.save_mesh(mesh, "interior_mesh")
```

## 3D Reconstruction Details

The 3D reconstruction module converts depth maps to 3D point clouds and meshes:

1. **Point Clouds**: Direct conversion of depth maps to 3D coordinates with color information
2. **Meshes**: Surface reconstruction from point clouds using Poisson reconstruction
3. **Diffusion Enhancement**: Depth maps are enhanced using a lightweight diffusion model for improved quality

Adjust the `downsample_factor` to control quality vs. speed:
- Lower values (1-2): Higher quality, slower processing
- Higher values (4-8): Lower quality, faster processing

### Optimized Weight Loading

The system has been optimized to streamline the weight loading process:

1. First checks for cached weights to avoid redundant downloads
2. If no cached weights exist, loads from MiDaS small model via PyTorch Hub
3. Adapts weights to the lightweight architecture
4. Saves the adapted weights for future use

This optimization eliminates unnecessary network requests and failure paths, making the system more reliable and efficient, especially in environments with limited connectivity.

## Example

```python
# Load and process an image
image = Image.open("example_room.jpg")
processed = reimaginer.image_processor.process_image(image)

# Generate 3D point cloud
pcd = depth_reconstructor.depth_to_pointcloud(
    depth_map=processed.depth_map,
    image=image
)

# Generate 3D mesh
mesh = depth_reconstructor.pointcloud_to_mesh(pcd)

# Visualize the results
depth_reconstructor.visualize_pointcloud(pcd)
depth_reconstructor.visualize_mesh(mesh)
```

## Requirements

### Hardware Requirements
- **High-End GPU**: An NVIDIA A100 or similar high-performance GPU is **strongly recommended** for optimal performance
- 16GB+ RAM for local installation
- For Colab usage: Select A100 GPU runtime when available

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- open3d
- Gradio
- Transformers (Hugging Face)
- Diffusers (Stable Diffusion)
- Other dependencies in `requirements.txt`

## Performance Optimizations

The latest version includes several performance optimizations:

1. **Streamlined Weight Loading**: Direct loading of MiDaS small model weights with caching
2. **Removed GAN-based Enhancement**: Simplified to use only the diffusion-based method that provides better results
3. **UI Simplification**: Removed depth map comparison tab to streamline the interface
4. **Reduced Memory Usage**: Optimized model initialization and memory management

These optimizations make the system more reliable and efficient, especially in resource-constrained environments.
