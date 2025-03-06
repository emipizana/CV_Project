#!/usr/bin/env python3
"""
Advanced Interior Reimagining AI - Main Application

This script initializes and launches the Interior Reimagining AI application.
"""

import os
import argparse
import logging
import torch

from models import InteriorReimaginer
from ui import create_advanced_ui
from utils import setup_logging, get_device, print_system_info

def main():
    """Main function to run the Advanced Interior Reimaginer application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Interior Reimagining AI")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"],
                        help="Device to run the models on (default: auto-detect)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU usage even if GPU is available (slower but more compatible)")
    parser.add_argument("--share", action="store_true",
                        help="Create a shareable link for the UI")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the Gradio app on")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0",
                        help="Base model ID for Stable Diffusion")

    args = parser.parse_args()

    # Configure logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Advanced Interior Reimagining AI")

    # Handle device selection
    device = None
    if args.cpu_only:
        device = "cpu"
    elif args.device:
        device = args.device
        logger.info(f"Using device: {device} as specified by command line argument")
    else:
        device = get_device(args.cpu_only)

    # Check for MPS (Apple Silicon) and set fallback environment variable if needed
    if device == "mps" or (device is None and torch.backends.mps.is_available()):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for Apple Silicon compatibility")

    # Print system info
    print_system_info()

    # Create the reimaginer object
    logger.info(f"Initializing Interior Reimaginer with model {args.model}...")
    reimaginer = InteriorReimaginer(base_model_id=args.model, device=device)

    # Create and launch the UI
    logger.info("Starting web interface...")
    ui = create_advanced_ui(reimaginer)
    ui.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()