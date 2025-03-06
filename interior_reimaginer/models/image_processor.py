import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2
import logging
from enum import Enum

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation,
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPSegProcessor,
    CLIPSegForImageSegmentation
)
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

class RoomSegmentClass(Enum):
    """Enumeration of room segment classes for semantic segmentation"""
    WALL = 0
    FLOOR = 1
    CEILING = 2
    WINDOW = 3
    DOOR = 4
    FURNITURE = 5
    LIGHTING = 6
    DECOR = 7
    OTHER = 8

@dataclass
class ProcessedImage:
    """Container for processed image data and metadata"""
    original: Image.Image
    depth_map: Optional[np.ndarray] = None
    segmentation_map: Optional[np.ndarray] = None
    edge_map: Optional[np.ndarray] = None
    normal_map: Optional[np.ndarray] = None
    room_analysis: Optional[Dict[str, Any]] = None
    object_masks: Optional[Dict[str, np.ndarray]] = None

class ImageProcessor:
    """Class for processing and analyzing interior images"""

    def __init__(self, device: str = None):
        """Initialize image processing models"""
        # Determine device
        if device is None:
            # First try CUDA (NVIDIA GPUs)
            if torch.cuda.is_available():
                self.device = "cuda"
            # Then try MPS (Apple Silicon GPUs)
            elif torch.backends.mps.is_available():
                self.device = "cpu"
                logger.warning("Using Apple Silicon GPU with MPS. Some operations may fall back to CPU.")
            # Fall back to CPU
            else:
                self.device = "cpu"
                logger.warning("No GPU detected, using CPU. Processing will be significantly slower.")
        else:
            self.device = device

        logger.info(f"Image processor using device: {self.device}")

        # Load depth estimation model
        logger.info("Loading depth estimation model...")
        self.depth_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)

        # Load segmentation model
        logger.info("Loading segmentation model...")
        self.seg_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(self.device)

        # Load CLIP model for text-based segmentation
        logger.info("Loading CLIP segmentation model...")
        self.clip_seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clip_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)

        # Load image captioning model for automatic scene understanding
        logger.info("Loading image captioning model...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

    def process_image(self, image: Image.Image) -> ProcessedImage:
        """
        Process an interior image to extract depth, segmentation, and other information

        Args:
            image: PIL Image of an interior

        Returns:
            ProcessedImage object with all processing results
        """
        print(f"===== STARTING PROCESS_IMAGE =====")
        print(f"Input image type: {type(image)}")
        print(f"Input image size: {image.size} (width x height)")
        print(f"Input image mode: {image.mode}")

        # Resize image if needed for processing
        width, height = image.size
        # Ensure dimensions are appropriate for models (multiple of 32)
        proc_width = (width // 32) * 32
        proc_height = (height // 32) * 32
        print(f"Original dimensions: {width}x{height}")
        print(f"Processed dimensions (multiple of 32): {proc_width}x{proc_height}")

        if proc_width != width or proc_height != height:
            print(f"Resizing image from {width}x{height} to {proc_width}x{proc_height}")
            proc_image = image.resize((proc_width, proc_height))
        else:
            print(f"No resizing needed")
            proc_image = image

        print(f"Processed image size: {proc_image.size}")

        # Create result container
        result = ProcessedImage(original=image)
        print(f"Created ProcessedImage container with original image")

        # Get depth map
        print("\n----- ENTERING DEPTH ESTIMATION -----")
        print(f"Depth estimation input image size: {proc_image.size}")
        try:
            logger.info("Estimating depth...")
            print(f"depth_processor type: {type(self.depth_processor)}")
            print(f"depth_model type: {type(self.depth_model)}")
            print(f"depth_model device: {self.depth_model.device}")

            depth = self._estimate_depth(proc_image)
            print(f"Depth map created successfully, shape: {depth.shape}, dtype: {depth.dtype}")
            print(f"Depth map values - min: {depth.min()}, max: {depth.max()}, mean: {depth.mean()}")
            result.depth_map = depth
            print("----- DEPTH ESTIMATION COMPLETED -----")
        except Exception as e:
            print(f"ERROR IN DEPTH ESTIMATION: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("----- DEPTH ESTIMATION FAILED -----")
            # Continue with other processing

        # Get segmentation map
        print("\n----- ENTERING SEGMENTATION -----")
        print(f"Segmentation input image size: {proc_image.size}")
        try:
            logger.info("Generating segmentation map...")
            print(f"seg_processor type: {type(self.seg_processor)}")
            print(f"seg_model type: {type(self.seg_model)}")
            print(f"seg_model device: {self.seg_model.device}")

            segmentation = self._segment_room(proc_image)
            print(f"Segmentation map created successfully, shape: {segmentation.shape}, dtype: {segmentation.dtype}")
            print(f"Segmentation map values - min: {segmentation.min()}, max: {segmentation.max()}, unique values: {np.unique(segmentation)}")
            result.segmentation_map = segmentation
            print("----- SEGMENTATION COMPLETED -----")
        except Exception as e:
            print(f"ERROR IN SEGMENTATION: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("----- SEGMENTATION FAILED -----")
            # Continue with other processing

        # Get edge map
        print("\n----- ENTERING EDGE DETECTION -----")
        print(f"Edge detection input image size: {proc_image.size}")
        try:
            logger.info("Detecting edges...")
            edge_map = self._detect_edges(proc_image)
            print(f"Edge map created successfully, shape: {edge_map.shape}, dtype: {edge_map.dtype}")
            print(f"Edge map values - min: {edge_map.min()}, max: {edge_map.max()}, mean: {edge_map.mean()}")
            result.edge_map = edge_map
            print("----- EDGE DETECTION COMPLETED -----")
        except Exception as e:
            print(f"ERROR IN EDGE DETECTION: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("----- EDGE DETECTION FAILED -----")
            # Continue with other processing

        # Generate object masks using CLIP-guided segmentation
        print("\n----- ENTERING OBJECT MASK GENERATION -----")
        print(f"Object mask input image size: {proc_image.size}")
        try:
            logger.info("Generating object masks...")
            print(f"clip_seg_processor type: {type(self.clip_seg_processor)}")
            print(f"clip_seg_model type: {type(self.clip_seg_model)}")
            print(f"clip_seg_model device: {self.clip_seg_model.device}")

            object_labels = ["wall", "floor", "ceiling", "window", "door",
                            "sofa", "chair", "table", "lamp", "plant", "artwork", "rug"]
            print(f"Generating masks for {len(object_labels)} labels: {object_labels}")

            object_masks = self._generate_object_masks(proc_image, object_labels)
            print(f"Object masks created successfully, found {len(object_masks)} masks")
            for label, mask in object_masks.items():
                print(f"  - Mask '{label}' shape: {mask.shape}, dtype: {mask.dtype}, values range: {mask.min()}-{mask.max()}")
            result.object_masks = object_masks
            print("----- OBJECT MASK GENERATION COMPLETED -----")
        except Exception as e:
            print(f"ERROR IN OBJECT MASK GENERATION: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("----- OBJECT MASK GENERATION FAILED -----")
            # Continue with other processing

        # Analyze room content with captioning model
        print("\n----- ENTERING ROOM ANALYSIS -----")
        print(f"Room analysis input image size: {proc_image.size}")
        try:
            logger.info("Analyzing room content...")
            print(f"caption_processor type: {type(self.caption_processor)}")
            print(f"caption_model type: {type(self.caption_model)}")
            print(f"caption_model device: {self.caption_model.device}")

            room_analysis = self._analyze_room(proc_image)
            print(f"Room analysis completed successfully, keys: {list(room_analysis.keys())}")
            for key, value in room_analysis.items():
                print(f"  - {key}: {value[:50]}..." if isinstance(value, str) and len(value) > 50 else f"  - {key}: {value}")
            result.room_analysis = room_analysis
            print("----- ROOM ANALYSIS COMPLETED -----")
        except Exception as e:
            print(f"ERROR IN ROOM ANALYSIS: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("----- ROOM ANALYSIS FAILED -----")
            # Continue with other processing

        print("\n===== PROCESS_IMAGE COMPLETED =====")
        return result

    def _estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate depth map from image"""
        # Prepare image for the model
        inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)

        # Get depth prediction
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Normalize depth map
        depth_map = prediction.squeeze().cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = 255 * ((depth_map - depth_min) / (depth_max - depth_min))
        return depth_map.astype(np.uint8)

    def _segment_room(self, image: Image.Image) -> np.ndarray:
        """Create semantic segmentation of the room"""
        # Prepare image for the model
        inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)

        # Get segmentation prediction
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            seg_map = outputs.logits.argmax(dim=1).squeeze(0)

        # Convert to numpy and appropriate size
        seg_map = seg_map.cpu().numpy().astype(np.uint8)

        # Resize to original size if needed
        if seg_map.shape[:2] != image.size[::-1]:
            seg_map = cv2.resize(seg_map, image.size, interpolation=cv2.INTER_NEAREST)

        return seg_map

    def _detect_edges(self, image: Image.Image) -> np.ndarray:
        """Detect edges in the image for structural preservation"""
        # Convert PIL to OpenCV format
        img_cv = np.array(image.convert('RGB'))
        img_cv = img_cv[:, :, ::-1]  # RGB to BGR

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)

        return edges

    def _generate_object_masks(self, image: Image.Image, object_labels: List[str]) -> Dict[str, np.ndarray]:
        """Generate binary masks for specific objects using CLIP-guided segmentation"""
        # Prepare image
        inputs = self.clip_seg_processor(
            text=object_labels,
            images=[image] * len(object_labels),
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Get segmentation predictions
        with torch.no_grad():
            outputs = self.clip_seg_model(**inputs)

        logits = outputs.logits

        # Process logits to create binary masks
        masks = {}
        for i, label in enumerate(object_labels):
            try:
                mask_logits = logits[i]

                # Check mask_logits shape and fix if needed
                if len(mask_logits.shape) == 1:
                    # Reshape 1D tensor to 2D - find dimensions based on length
                    size = int(np.sqrt(mask_logits.shape[0]))
                    mask_logits = mask_logits.reshape(1, 1, size, size)
                elif len(mask_logits.shape) == 2:
                    # Add batch and channel dimensions if needed
                    mask_logits = mask_logits.unsqueeze(0).unsqueeze(0)
                elif len(mask_logits.shape) == 3:
                    # Add batch dimension if needed
                    mask_logits = mask_logits.unsqueeze(0)

                # Process mask logits
                mask_probs = torch.sigmoid(mask_logits)
                mask = (mask_probs > 0.5).float()

                # Use target size and ensure mask has proper dimensions
                target_size = image.size[::-1]  # (height, width)
                mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
                masks[label] = mask.squeeze().cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to generate mask for {label}: {str(e)}")
                # Continue with other labels

        return masks

    def _analyze_room(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze room content using image captioning model"""
        # Prepare image for basic caption
        inputs = self.caption_processor(image, return_tensors="pt").to(self.device)

        result = {}

        try:
            # Generate basic caption
            with torch.no_grad():
                output = self.caption_model.generate(**inputs, max_length=100)
                caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
                result["caption"] = caption
        except Exception as e:
            logger.warning(f"Error generating caption: {e}")
            result["caption"] = "Unable to generate caption"

        # Generate specific analysis by prompting with questions
        questions = [
            "What is the style of this room?",
            "What colors are dominant in this room?",
            "What furniture is present in this room?",
            "What is the lighting like in this room?",
            "What materials are visible in this room?"
        ]

        # Map questions to result keys
        question_keys = {
            "What is the style of this room?": "style",
            "What colors are dominant in this room?": "colors",
            "What furniture is present in this room?": "furniture",
            "What is the lighting like in this room?": "lighting",
            "What materials are visible in this room?": "materials"
        }

        for question in questions:
            try:
                with torch.no_grad():
                    # For BLIP model, we need to use the processor properly with text input
                    inputs = self.caption_processor(
                        images=image,
                        text=question,
                        return_tensors="pt"
                    ).to(self.device)

                    output = self.caption_model.generate(**inputs, max_length=50)
                    answer = self.caption_processor.decode(output[0], skip_special_tokens=True)

                    # Store the answer using the mapped key
                    key = question_keys[question]
                    result[key] = answer
            except Exception as e:
                logger.warning(f"Error generating answer for '{question}': {e}")
                key = question_keys[question]
                result[key] = ""

        return result