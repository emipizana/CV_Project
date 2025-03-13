# Interior Reimaginer

A project exploring interior design using AI to generate and optimize design solutions with depth-based 3D reconstruction capabilities.

## Features

- **Full Room Redesign**: Transform interior spaces with AI-powered style transfer
- **Targeted Redesign**: Modify specific elements like walls, floors, or furniture
- **Material Explorer**: Compare different materials on interior surfaces
- **Style Explorer**: Visualize different design styles for your space
- **3D Reconstruction**: Generate 3D point clouds and meshes from interior images using depth estimation

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

To run Interior Reimaginer on Google Colab:

1. Create a new notebook (or use our provided example notebook at `interior_reimaginer/examples/colab_3d_reconstruction.ipynb`)
2. Clone the repository:
```python
!git clone https://github.com/your-username/CV_Project.git
%cd CV_Project
```

3. Install dependencies:
```python
!pip install -r interior_reimaginer/requirements.txt
```

4. Install Open3D (for 3D reconstruction):
```python
!pip install open3d
```

5. Import the required modules:
```python
from interior_reimaginer.models.interior_reimaginer import InteriorReimaginer
from interior_reimaginer.models.reconstruction_3d import DepthReconstructor
from PIL import Image
import numpy as np
```

## Usage

### Local Usage

Run the Gradio interface:

```bash
python interior_reimaginer/main.py
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

Adjust the `downsample_factor` to control quality vs. speed:
- Lower values (1-2): Higher quality, slower processing
- Higher values (4-8): Lower quality, faster processing

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

- Python 3.8+
- PyTorch 2.0+
- open3d
- Gradio
- Transformers (Hugging Face)
- Diffusers (Stable Diffusion)
- Other dependencies in `requirements.txt`
