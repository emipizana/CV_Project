# Interior Reimaginer

A project exploring interior design using AI to generate and optimize design solutions with depth-based 3D reconstruction capabilities.

## Project Structure

```
CV_Project/
├── README.md
├── interior_reimaginer/
│   ├── main.py
│   ├── requirements.txt
│   ├── run_reimaginer.ipynb
│   ├── examples/
│   │   ├── advanced_3d_visualization.ipynb
│   │   ├── colab_3d_reconstruction.ipynb
│   ├── models/
│   │   ├── __init__.py
│   │   ├── depth_map_comparison.py
│   │   ├── design_styles.py
│   │   ├── image_processor.py
│   │   ├── interior_reimaginer.py
│   │   ├── lightweight_diffusion.py
│   │   ├── reconstruction_3d.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── gradio_interface.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
```

## Features

- **Full Room Redesign**: Transform interior spaces with AI-powered style transfer
- **3D Reconstruction**: Generate 3D point clouds and meshes from interior images using optimized depth estimation
- **User-Friendly Interface**: Simple Gradio UI with both local and Colab support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CV_Project.git
cd CV_Project
```

2. Install dependencies:
```bash
pip install -r interior_reimaginer/requirements.txt
```
3. Run the Gradio interface:

```bash
python interior_reimaginer/main.py
```

This will start the application and generate both a local URL (http://127.0.0.1:7860) and a public shareable link that you can use to access the interface from other devices. The shareable link will be displayed in the console.

If you don't want to create a public link, use:

```bash
python interior_reimaginer/main.py --share=False
```

To run Interior Reimaginer on Google Colab with optimal performance:

## Image Reimagining

The Image Reimagining module allows users to transform the style of their interior spaces using AI. This process leverages the power of Stable Diffusion models to apply different design styles to the input image, offering a variety of aesthetic options.

**Key Capabilities:**

-   **Full Room Redesign:** Transform the entire interior space based on a textual style prompt. Options include preserving the original structure using ControlNet with depth maps and preserving the color scheme.
-   **Targeted Reimagining:** Modify specific parts of the room (e.g., walls, floor, furniture) using inpainting. The system automatically detects and masks common objects, or generates masks on-the-fly using CLIP segmentation for custom targets.
-   **Style Prompting:** Users can specify a target style using free-form text, or choose from predefined design styles.
-   **Batch Variations:** Generate multiple design variations with different style prompts.

The system uses the `diffusers` library and includes optimizations for memory usage and performance, such as attention slicing and xformers (when available).

## 3D Reconstruction Details

The 3D reconstruction module converts depth maps to 3D point clouds and meshes:

1. **Point Clouds**: Direct conversion of depth maps to 3D coordinates with color information
3. **Diffusion Enhancement**: Depth maps are enhanced using a lightweight diffusion model for improved quality

Adjust the `downsample_factor` to control quality vs. speed:
- Lower values (1-2): Higher quality, slower processing
- Higher values (4-8): Lower quality, faster processing

### Optimized Weight Loading

The system has been optimized to streamline the weight loading process:

1. First checks for cached weights to avoid redundant downloads
3. Adapts weights to the lightweight architecture

This optimization eliminates unnecessary network requests and failure paths, making the system more reliable and efficient, especially in environments with limited connectivity.

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
3. **UI Simplification**: Removed depth map comparison tab to streamline the interface
4. **Reduced Memory Usage**: Optimized model initialization and memory management

These optimizations make the system more reliable and efficient, especially in resource-constrained environments.

## Authors

- Emiliano Pizaña Vela - emiliano.pizana-vela@polytechnique.edu
- Alfonso Mateos Vicente - alfonso.mateos-vicente@polytechnique.edu
