import os
import torch
import numpy as np
from PIL import Image
import cv2
import random
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLControlNetImg2ImgPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    DDIMScheduler
)
import torch.nn.functional as F

from .image_processor import ImageProcessor, ProcessedImage
from .design_styles import load_design_styles, DesignStyle

# Configure logging
logger = logging.getLogger(__name__)

class InteriorReimaginer:
    """Main class for interior reimagining functionality"""

    def __init__(self, base_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0", device: str = None):
        """
        Initialize the Interior Reimaginer with models and processors.

        Args:
            base_model_id: HuggingFace model ID for the Stable Diffusion model
            device: Device to run inference on ('cuda', 'mps', or 'cpu')
        """
        # Determine device
        if device is None:
            # First try CUDA (NVIDIA GPUs)
            if torch.cuda.is_available():
                self.device = "cuda"
            # Then try MPS (Apple Silicon GPUs)
            elif torch.backends.mps.is_available():
                self.device = "cpu"
                logger.warning("Using Apple Silicon GPU with MPS. Some operations may fall back to CPU.")
                logger.warning("For optimal performance on Apple Silicon, ensure PYTORCH_ENABLE_MPS_FALLBACK=1 is set.")
            # Fall back to CPU
            else:
                self.device = "cpu"
                logger.warning("No GPU detected, using CPU. Processing will be significantly slower.")
        else:
            self.device = device

        logger.info(f"Reimaginer using device: {self.device}")

        # Initialize image processor
        self.image_processor = ImageProcessor(device=self.device)

        # Load models
        logger.info("Loading diffusion models...")

        # Image-to-image model
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.img2img_pipe.enable_model_cpu_offload()
        self.img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.img2img_pipe.scheduler.config)

        # Inpainting model
        self.controlnet_inpainting = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        )
        self.inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=self.controlnet_inpainting, torch_dtype=torch.float16
        )
        self.inpaint_pipe.enable_model_cpu_offload()
        self.inpaint_pipe.enable_xformers_memory_efficient_attention()
        self.inpaint_pipe.scheduler = DDIMScheduler.from_config(self.inpaint_pipe.scheduler.config)

        # ControlNet models
        logger.info("Loading ControlNet models...")
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # Setup ControlNet pipelines
        self.vae_depth = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.depth_controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet_depth,
            vae=self.vae_depth,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.depth_controlnet_pipe.enable_model_cpu_offload()
        self.depth_controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.depth_controlnet_pipe.scheduler.config)

        self.vae_canny = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.canny_controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet_canny,
            vae=self.vae_canny,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.canny_controlnet_pipe.enable_model_cpu_offload()
        self.canny_controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.canny_controlnet_pipe.scheduler.config)

        # Apply memory optimizations if using GPU
        if self.device == "cuda":
            self._apply_memory_optimizations()

        # Load design styles
        self.design_styles = load_design_styles()

    def _apply_memory_optimizations(self):
        """Apply memory optimizations to diffusion models"""
        # Apply optimizations to all pipelines
        for pipe in [self.img2img_pipe, self.inpaint_pipe,
                    self.depth_controlnet_pipe, self.canny_controlnet_pipe]:
            # Attention slicing is compatible with all devices
            pipe.enable_attention_slicing()

            # xformers is CUDA-only and not available on MPS or CPU
            if self.device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers for memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
                    logger.warning("Install xformers for better performance on NVIDIA GPUs")

    def analyze_interior(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze an interior image to provide insights and recommendations.

        Args:
            image: PIL Image of the interior to analyze

        Returns:
            Dictionary with analysis results and recommendations
        """
        # Process the image
        processed = self.image_processor.process_image(image)

        # Extract key information from room analysis with default values if missing
        room_analysis = processed.room_analysis or {}
        current_style = room_analysis.get("style", "").lower()
        current_colors = room_analysis.get("colors", "").lower()
        current_materials = room_analysis.get("materials", "").lower()

        # Find most similar style from our predefined styles
        matched_style = None
        for style_name, style in self.design_styles.items():
            if any(keyword in current_style for keyword in [style_name, style.name.lower()]):
                matched_style = style
                break

        # If no match found, provide general analysis
        if matched_style is None:
            style_name = "custom"
            style_description = "Your current style appears to be a custom blend"
        else:
            style_name = matched_style.name
            style_description = matched_style.description

        # Generate style recommendations
        recommendations = []
        for style_name, style in self.design_styles.items():
            # Skip current style
            if style.name.lower() in current_style:
                continue

            recommendations.append({
                "style_name": style.name,
                "description": style.description,
                "compatibility": random.randint(60, 95)  # In a real app, this would use a more sophisticated algorithm
            })

        # Sort recommendations by compatibility
        recommendations = sorted(recommendations, key=lambda x: x["compatibility"], reverse=True)[:3]

        # Create the analysis result
        analysis_result = {
            "current_style": {
                "name": style_name,
                "description": style_description,
                "colors": current_colors,
                "materials": current_materials,
                "furniture": room_analysis.get("furniture", "")
            },
            "recommendations": recommendations,
            "room_caption": room_analysis.get("caption", ""),
            "detected_objects": list(processed.object_masks.keys()) if processed.object_masks else []
        }

        return analysis_result

    def reimagine_full(
        self,
        processed_image: ProcessedImage,
        style_prompt: str,
        style_strength: float = 0.75,
        preserve_structure: bool = True,
        preserve_color_scheme: bool = False,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_images: int = 4,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Reimagine a complete interior using advanced AI models and controls.

        Args:
            processed_image: ProcessedImage containing original and analysis data
            style_prompt: Text description of desired style changes
            style_strength: How much to transform the image (0-1)
            preserve_structure: Whether to maintain room structure
            preserve_color_scheme: Whether to preserve the original color scheme
            negative_prompt: Text description of what to avoid
            guidance_scale: How closely to follow the prompt
            num_images: Number of variations to generate
            seed: Random seed for reproducibility

        Returns:
            List of reimagined interior images
        """
        logger.info(f"Reimagining interior with prompt: {style_prompt}")

        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Apply style from predefined styles if it matches
        matched_style = None
        for style_name, style in self.design_styles.items():
            if style_name.lower() in style_prompt.lower() or style.name.lower() in style_prompt.lower():
                matched_style = style
                break

        # Create full prompt based on style and user input
        if matched_style:
            style_modifiers = ", ".join(matched_style.prompt_modifiers)
            style_negative = ", ".join(matched_style.negative_modifiers)
            full_prompt = f"Interior design: {style_prompt}, {style_modifiers}, photorealistic, high quality, high resolution"
            full_negative_prompt = f"low quality, blurry, distorted proportions, {style_negative}, {negative_prompt}"
        else:
            full_prompt = f"Interior design: {style_prompt}, high quality, photorealistic, high resolution"
            full_negative_prompt = f"low quality, blurry, distorted proportions, {negative_prompt}"

        # Choose pipeline based on settings
        if preserve_structure:
            # Use ControlNet with depth or edges for structure preservation
            try:
                # Convert depth map to RGB
                depth_map = processed_image.depth_map
                depth_image = Image.fromarray(cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO))

                # Generate with ControlNet-Depth
                logger.info("Using ControlNet-Depth for structure preservation")
                result = self.depth_controlnet_pipe(
                    prompt=full_prompt,
                    negative_prompt=full_negative_prompt,
                    image=depth_image,
                    controlnet_conditioning_scale=0.7,  # Adjust how strongly the depth controls generation
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=30,
                    generator=generator
                )
                return result.images
            except Exception as e:
                logger.error(f"Error using ControlNet: {e}")
                logger.info("Falling back to standard img2img")

        # Standard img2img as fallback or if structure preservation not needed
        result = self.img2img_pipe(
            prompt=full_prompt,
            negative_prompt=full_negative_prompt,
            image=processed_image.original,
            strength=style_strength,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator
        )

        return result.images

    def reimagine_targeted(
        self,
        processed_image: ProcessedImage,
        target_area: str,  # e.g., "walls", "floor", "furniture"
        style_prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_images: int = 4,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Reimagine a specific part of the interior using inpainting.

        Args:
            processed_image: ProcessedImage containing original and analysis data
            target_area: What part of the room to modify
            style_prompt: Text description of desired style changes
            negative_prompt: Text description of what to avoid
            guidance_scale: How closely to follow the prompt
            num_images: Number of variations to generate
            seed: Random seed for reproducibility

        Returns:
            List of reimagined interior images
        """
        logger.info(f"Targeted reimagining of {target_area} with prompt: {style_prompt}")

        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Get mask for the target area
        target_mask = None
        mapping = {
            "wall": "wall", "walls": "wall",
            "floor": "floor", "flooring": "floor",
            "sofa": "sofa", "couch": "sofa",
            "chair": "chair", "chairs": "chair",
            "table": "table", "tables": "table",
            "window": "window", "windows": "window",
            "door": "door", "doors": "door",
            "lamp": "lamp", "lighting": "lamp",
            "rug": "rug", "carpet": "rug",
            "plant": "plant", "plants": "plant",
            "artwork": "artwork", "art": "artwork"
        }

        normalized_target = mapping.get(target_area.lower())
        if normalized_target and normalized_target in processed_image.object_masks:
            mask_array = processed_image.object_masks[normalized_target]
            target_mask = Image.fromarray((mask_array * 255).astype(np.uint8))
        else:
            # If target not found in predefined masks, use CLIP to generate it on the fly
            target_prompts = [target_area]
            inputs = self.image_processor.clip_seg_processor(
                text=target_prompts,
                images=[processed_image.original],
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.image_processor.clip_seg_model(**inputs)

            # Process output to create mask
            logits = outputs.logits
            mask_probs = torch.sigmoid(logits)
            mask = (mask_probs > 0.5).float()
            mask = F.interpolate(
                mask,
                size=processed_image.original.size[::-1],
                mode='bilinear',
                align_corners=False
            )
            mask_array = mask.squeeze().cpu().numpy()
            target_mask = Image.fromarray((mask_array * 255).astype(np.uint8))

        # Create full prompt based on target area and style
        full_prompt = f"{target_area} in {style_prompt} style, high quality, photorealistic, high resolution"
        full_negative_prompt = f"low quality, blurry, distorted proportions, {negative_prompt}"

        # Use inpainting to modify only the target area
        result = self.inpaint_pipe(
            prompt=full_prompt,
            negative_prompt=full_negative_prompt,
            image=processed_image.original,
            mask_image=target_mask,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator
        )

        return result.images

    def create_batch_variations(
        self,
        original_image: Image.Image,
        style_prompts: List[str],
        negative_prompt: str = "",
        style_strength: float = 0.75,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, List[Image.Image]]:
        """
        Create variations of the interior with different style prompts.

        Args:
            original_image: PIL Image of the interior
            style_prompts: List of style prompts to apply
            negative_prompt: Text description of what to avoid
            style_strength: How much to transform the image (0-1)
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping style prompts to lists of generated images
        """
        logger.info(f"Creating batch variations with {len(style_prompts)} styles")

        # Process the image
        processed = self.image_processor.process_image(original_image)

        # Generate variations for each style
        results = {}
        for prompt in style_prompts:
            images = self.reimagine_full(
                processed_image=processed,
                style_prompt=prompt,
                style_strength=style_strength,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_images=2,  # Fewer per style since we're generating multiple styles
                seed=seed
            )
            results[prompt] = images

        return results

    def compare_materials(
        self,
        processed_image: ProcessedImage,
        target_area: str,
        material_options: List[str],
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Image.Image]:
        """
        Compare different materials for a specific area of the interior.

        Args:
            processed_image: ProcessedImage containing original and analysis data
            target_area: What part of the room to modify (e.g., "walls", "floor")
            material_options: List of materials to try (e.g., ["wooden", "marble", "concrete"])
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping material names to generated images
        """
        logger.info(f"Comparing {len(material_options)} material options for {target_area}")

        results = {}
        for material in material_options:
            style_prompt = f"{target_area} with {material} material"
            images = self.reimagine_targeted(
                processed_image=processed_image,
                target_area=target_area,
                style_prompt=style_prompt,
                guidance_scale=guidance_scale,
                num_images=1,
                seed=seed
            )
            if images:
                results[material] = images[0]

        return results