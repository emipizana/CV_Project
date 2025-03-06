# Advanced Interior Reimagining AI

A sophisticated Python application for transforming interior spaces using AI-powered design tools. This project leverages state-of-the-art deep learning models including Stable Diffusion XL, ControlNet, and various computer vision techniques to analyze and reimagine interior designs.

## Features

- **Full Room Redesign**: Transform entire interiors with various design styles
- **Targeted Redesign**: Modify specific elements like walls, floors, or furniture
- **Material Explorer**: Compare different materials for specific areas
- **Style Explorer**: Visualize your space in different design styles
- **Automatic Room Analysis**: AI-powered analysis of existing interior style and elements

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU with 8GB+ VRAM recommended (NVIDIA)
- For best performance: NVIDIA GPU with CUDA support

### Installation

1. Clone this repository:
```bash
git clone https://github.com/emipizana/CV_Project.git
cd CV_Project/interior_reimaginer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

Launch the application using:
```bash
python main.py
```

Additional command-line options:
```
--device DEVICE     Device to run models on (cuda, mps, or cpu)
--cpu-only          Force CPU usage even if GPU is available
--share             Create a shareable link for the web interface
--debug             Enable debug logging
```

### Running on Google Colab

1. Create a new Colab notebook
2. Clone the repository:
```python
!git clone https://github.com/emipizana/CV_Project.git
%cd CV_Project/interior_reimaginer
```

3. Install dependencies:
```python
!pip install torch torchvision diffusers transformers gradio opencv-python Pillow accelerate xformers safetensors huggingface-hub sentencepiece
```

4. Run the application:
```python
!python main.py --share
```

5. Open the provided public URL to access the web interface

## Project Structure

```
interior_reimaginer/
│
├── models/
│   ├── __init__.py
│   ├── image_processor.py     # Image analysis functionality
│   ├── interior_reimaginer.py # Core reimagining functionality
│   └── design_styles.py       # Design style definitions
│
├── ui/
│   ├── __init__.py
│   └── gradio_interface.py    # Gradio UI implementation
│
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Common utility functions
│
├── main.py                    # Entry point script
└── run_reimaginer.ipynb       # Notebook to run the application
```

## Design Styles

The application includes several predefined interior design styles:

- **Minimalist**: Clean lines, minimal decoration, and functional furniture
- **Scandinavian**: Light, airy spaces with wooden elements and cozy textiles
- **Industrial**: Raw materials, exposed structures, and vintage elements
- **Mid-Century Modern**: Retro furniture, clean lines, and bold accent colors
- **Bohemian**: Eclectic, colorful spaces with mixed patterns and textures
- **Traditional**: Classic furniture, rich colors, and elegant details

## Tips for Best Results

- Use well-lit, clear photos of your interior spaces
- Provide specific design instructions for better results
- Use the style explorer to find inspiration before detailed redesign
- For targeted redesign, be specific about which area you want to modify
- Higher guidance scale values (10-15) produce results that more closely follow your prompt

## Troubleshooting

- **Out of Memory Errors**: Reduce image size or use CPU mode if your GPU has insufficient VRAM
- **Slow Processing**: Processing on CPU can be very slow, a CUDA-capable GPU is recommended
- **Model Downloads**: First run will download several large models (10+ GB total), ensure sufficient disk space and internet connection

## Technologies Used

- **Stable Diffusion XL**: State-of-the-art text-to-image generation
- **ControlNet**: Structure-preserving image generation
- **BLIP**: Image captioning and understanding
- **CLIPSeg**: Text-guided image segmentation
- **Depth Estimation**: Room structure analysis
- **Gradio**: Interactive web interface

## License

This project is provided for educational and research purposes only.

## Authors

This project was developed by École Polytechnique students:

- **Emiliano Pizana Vela** - [emiliano.pizana-vela@polytechnique.edu](mailto:emiliano.pizana-vela@polytechnique.edu)
- **Alfonso Mateos Vicente** - [alfonso.mateos-vicente@polytechnique.edu](mailto:alfonso.mateos-vicente@polytechnique.edu)

## Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion models
- [Hugging Face](https://huggingface.co/) for model hosting and libraries
- [Gradio](https://www.gradio.app/) for the web interface
