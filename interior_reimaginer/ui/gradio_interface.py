import gradio as gr
import os
import logging
import time
from PIL import Image
import torch
import random
from typing import List, Dict
import numpy as np

from models.interior_reimaginer import InteriorReimaginer
from models.reconstruction_3d import DepthReconstructor

# Configure logging
logger = logging.getLogger(__name__)

def create_advanced_ui(reimaginer: InteriorReimaginer) -> gr.Blocks:
    """
    Create an advanced Gradio UI for the Interior Reimaginer.

    Args:
        reimaginer: Initialized InteriorReimaginer object

    Returns:
        Gradio Blocks interface
    """
    # Define CSS
    css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .tool-description {
        margin-bottom: 1rem;
    }
    .tab-content {
        padding: 1rem;
    }
    .gallery-container {
        margin-top: 1rem;
    }
    .result-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
    }
    """

    # Available design styles from the reimaginer
    style_choices = list(reimaginer.design_styles.keys())
    
    # Initialize the 3D reconstructor
    depth_reconstructor = DepthReconstructor()

    with gr.Blocks(css=css, title="Advanced Interior Reimagining AI") as ui:
        # Header
        with gr.Row(elem_classes="header"):
            gr.Markdown("# Advanced Interior Reimagining AI")
            gr.Markdown("Transform your interior spaces with AI-powered design tools")

        # Main UI with tabs
        with gr.Tabs() as tabs:
            # Full Reimagining Tab
            with gr.TabItem("Full Room Redesign", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Original Interior")
                        analyze_btn = gr.Button("Analyze Room")

                        with gr.Accordion("Design Settings", open=True):
                            style_dropdown = gr.Dropdown(
                                choices=["custom"] + style_choices,
                                value="custom",
                                label="Design Style"
                            )
                            style_prompt = gr.Textbox(
                                label="Design Instructions",
                                placeholder="e.g., 'Modern minimalist with natural materials and plants'"
                            )
                            negative_prompt = gr.Textbox(
                                label="Elements to Avoid",
                                placeholder="e.g., 'dark colors, ornate decoration, cluttered'"
                            )

                        with gr.Accordion("Advanced Options", open=False):
                            with gr.Row():
                                style_strength = gr.Slider(
                                    minimum=0.2, maximum=0.9, value=0.4, step=0.05,
                                    label="Transformation Strength"
                                )
                                guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=12, step=0.5,
                                    label="Guidance Scale"
                                )

                            with gr.Row():
                                preserve_structure = gr.Checkbox(
                                    label="Preserve Room Structure",
                                    value=True
                                )
                                preserve_colors = gr.Checkbox(
                                    label="Preserve Color Scheme",
                                    value=False
                                )

                            with gr.Row():
                                num_images = gr.Slider(
                                    minimum=1, maximum=4, value=4, step=1,
                                    label="Number of Variations"
                                )
                                seed = gr.Number(
                                    label="Seed (blank for random)",
                                    precision=0
                                )

                        generate_btn = gr.Button("Reimagine Interior", variant="primary")

                    with gr.Column(scale=1):
                        with gr.Accordion("Room Analysis", open=True):
                            analysis_json = gr.JSON(label="Room Analysis Results")
                            room_caption = gr.Textbox(label="Room Description")

                        with gr.Accordion("Reimagined Designs", open=True):
                            output_gallery = gr.Gallery(
                                label="Reimagined Interiors",
                                show_label=True,
                                columns=2,
                                height="auto",
                                elem_classes="gallery-container"
                            )
                            selected_image = gr.Image(type="pil", visible=False)
                            select_output_btn = gr.Button("Use Selected Design")

            # Targeted Redesign Tab
            with gr.TabItem("Targeted Redesign", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=1):
                        target_input_image = gr.Image(type="pil", label="Original Interior")
                        target_analyze_btn = gr.Button("Analyze Room Elements")

                        with gr.Row():
                            target_area = gr.Dropdown(
                                choices=["walls", "floor", "ceiling", "furniture", "sofa", "chairs", "lighting"],
                                label="Target Area"
                            )
                            target_style = gr.Textbox(
                                label="Design Instructions for Target Area",
                                placeholder="e.g., 'Modern wooden floor'"
                            )

                        target_generate_btn = gr.Button("Reimagine Target Area", variant="primary")

                    with gr.Column(scale=1):
                        target_elements = gr.JSON(label="Detected Room Elements")
                        target_output_gallery = gr.Gallery(
                            label="Redesigned Area Variations",
                            show_label=True,
                            columns=2,
                            height="auto",
                            elem_classes="gallery-container"
                        )

            # Material Explorer Tab
            with gr.TabItem("Material Explorer", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=1):
                        material_input_image = gr.Image(type="pil", label="Original Interior")

                        with gr.Row():
                            material_area = gr.Dropdown(
                                choices=["walls", "floor", "ceiling"],
                                label="Surface Area"
                            )

                        with gr.Row():
                            material_options = gr.CheckboxGroup(
                                choices=["wood", "marble", "concrete", "tile", "wallpaper", "brick", "painted"],
                                label="Material Options"
                            )

                        material_generate_btn = gr.Button("Compare Materials", variant="primary")

                    with gr.Column(scale=1):
                        material_output_gallery = gr.Gallery(
                            label="Material Comparison",
                            show_label=True,
                            columns=3,
                            height="auto",
                            elem_classes="gallery-container"
                        )

            # 3D Reconstruction Tab
            with gr.TabItem("3D Reconstruction", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=1):
                        recon_input_image = gr.Image(type="pil", label="Original Interior")
                        
                        with gr.Row():
                            downsample_factor = gr.Slider(
                                minimum=1, maximum=8, value=2, step=1,
                                label="Downsample Factor (higher = faster, less detailed)"
                            )
                        
                        with gr.Row():
                            recon_options = gr.Radio(
                                choices=["Point Cloud", "Mesh"],
                                value="Point Cloud",
                                label="3D Reconstruction Type"
                            )
                            
                        recon_generate_btn = gr.Button("Generate 3D Model", variant="primary")
                        recon_save_btn = gr.Button("Save 3D Model")
                        
                    with gr.Column(scale=1):
                        recon_output = gr.Image(label="3D Reconstruction Preview")
                        recon_status = gr.Textbox(label="Status")
                        recon_download = gr.File(label="Download 3D Model")

            # Style Explorer Tab
            with gr.TabItem("Style Explorer", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=1):
                        style_input_image = gr.Image(type="pil", label="Original Interior")

                        with gr.Row():
                            style_options = gr.CheckboxGroup(
                                choices=style_choices,
                                label="Styles to Explore"
                            )

                        style_generate_btn = gr.Button("Compare Styles", variant="primary")

                    with gr.Column(scale=1):
                        style_output = gr.HTML(label="Style Variations")

        # Functionality for Full Room Redesign tab
        def update_prompt(style):
            if style == "custom":
                return gr.update(value="")
            else:
                style_obj = reimaginer.design_styles.get(style)
                if style_obj:
                    prompt_suggestion = ", ".join(style_obj.prompt_modifiers[:3])
                    return gr.update(value=prompt_suggestion)

        style_dropdown.change(update_prompt, inputs=[style_dropdown], outputs=[style_prompt])

        # Function to analyze room
        def analyze_room(image):
            if image is None:
                return None, "Please upload an image first."

            try:
                analysis = reimaginer.analyze_interior(image)
                caption = analysis.get("room_caption", "No description available.")
                return analysis, caption
            except Exception as e:
                logger.error(f"Error analyzing room: {e}")
                return {"error": str(e)}, "Error analyzing the image."

        analyze_btn.click(analyze_room, inputs=[input_image], outputs=[analysis_json, room_caption])

        # Function to reimagine full room
        def reimagine_room(image, style_prompt, negative_prompt, style_strength,
                          guidance_scale, preserve_structure, preserve_colors,
                          num_images, seed):
            if image is None:
                return None

            try:
                # Process the image
                processed = reimaginer.image_processor.process_image(image)

                # Generate redesigns
                images = reimaginer.reimagine_full(
                    processed_image=processed,
                    style_prompt=style_prompt,
                    style_strength=style_strength,
                    preserve_structure=preserve_structure,
                    preserve_color_scheme=preserve_colors,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_images=num_images,
                    seed=int(seed) if seed else None
                )

                return images
            except Exception as e:
                logger.error(f"Error reimagining room: {e}")
                return None

        generate_btn.click(
            reimagine_room,
            inputs=[
                input_image, style_prompt, negative_prompt,
                style_strength, guidance_scale, preserve_structure,
                preserve_colors, num_images, seed
            ],
            outputs=[output_gallery]
        )

        # Targeted redesign functionality
        def analyze_room_elements(image):
            if image is None:
                return None

            try:
                # Process the image
                processed = reimaginer.image_processor.process_image(image)

                # Return detected elements
                elements = {
                    "detected_objects": list(processed.object_masks.keys()),
                    "room_type": processed.room_analysis.get("style", "Unknown"),
                    "dominant_colors": processed.room_analysis.get("colors", "Unknown"),
                    "materials": processed.room_analysis.get("materials", "Unknown")
                }

                return elements
            except Exception as e:
                logger.error(f"Error analyzing room elements: {e}")
                return {"error": str(e)}

        target_analyze_btn.click(analyze_room_elements, inputs=[target_input_image], outputs=[target_elements])

        def reimagine_target(image, target_area, style_prompt):
            if image is None or not target_area:
                return None

            try:
                # Process the image
                processed = reimaginer.image_processor.process_image(image)

                # Generate targeted redesigns
                images = reimaginer.reimagine_targeted(
                    processed_image=processed,
                    target_area=target_area,
                    style_prompt=style_prompt,
                    num_images=4
                )

                return images
            except Exception as e:
                logger.error(f"Error in targeted reimagining: {e}")
                return None

        target_generate_btn.click(
            reimagine_target,
            inputs=[target_input_image, target_area, target_style],
            outputs=[target_output_gallery]
        )

        # Material explorer functionality
        def compare_materials(image, area, materials):
            if image is None or not area or not materials:
                return None

            try:
                # Process the image
                processed = reimaginer.image_processor.process_image(image)

                # Compare materials
                results = reimaginer.compare_materials(
                    processed_image=processed,
                    target_area=area,
                    material_options=materials
                )

                # Convert dictionary to list for gallery
                result_images = list(results.values())

                return result_images
            except Exception as e:
                logger.error(f"Error comparing materials: {e}")
                return None

        material_generate_btn.click(
            compare_materials,
            inputs=[material_input_image, material_area, material_options],
            outputs=[material_output_gallery]
        )
        
        # 3D Reconstruction functionality
        def generate_3d_model(image, downsample, model_type):
            if image is None:
                return None, "Please upload an image first.", None
            
            try:
                # Process the image to get depth map
                processed = reimaginer.image_processor.process_image(image)
                
                if processed.depth_map is None:
                    return None, "Failed to generate depth map.", None
                
                # Create point cloud from depth map
                pcd = depth_reconstructor.depth_to_pointcloud(
                    depth_map=processed.depth_map,
                    image=image,
                    downsample_factor=int(downsample)
                )
                
                # Store the point cloud in a state variable for later use
                state = {"pcd": pcd, "type": model_type}
                
                # Create mesh if requested
                if model_type == "Mesh":
                    try:
                        mesh = depth_reconstructor.pointcloud_to_mesh(pcd)
                        state["mesh"] = mesh
                        # Render the mesh to an image
                        render_img = depth_reconstructor.render_mesh_image(mesh)
                    except Exception as e:
                        logger.error(f"Error creating mesh: {e}")
                        return None, f"Error creating mesh: {str(e)}", None
                else:
                    # Render the point cloud to an image
                    render_img = depth_reconstructor.render_pointcloud_image(pcd)
                
                # Convert the rendered image from float (0-1) to uint8 (0-255)
                if render_img.max() <= 1.0:
                    render_img = (render_img * 255).astype(np.uint8)
                
                # Create PIL image from numpy array
                preview_img = Image.fromarray(render_img)
                
                return preview_img, f"Successfully created 3D {model_type.lower()}.", state
                
            except Exception as e:
                logger.error(f"Error in 3D reconstruction: {e}")
                return None, f"Error in 3D reconstruction: {str(e)}", None
        
        def save_3d_model(state):
            if state is None or "pcd" not in state:
                return None, "No 3D model has been generated yet."
            
            try:
                timestamp = int(time.time())
                
                if state["type"] == "Mesh" and "mesh" in state:
                    # Save mesh
                    filename = f"reconstruction_mesh_{timestamp}"
                    filepath = depth_reconstructor.save_mesh(state["mesh"], filename)
                    return filepath, f"Mesh saved as {filepath}"
                else:
                    # Save point cloud
                    filename = f"reconstruction_pointcloud_{timestamp}"
                    filepath = depth_reconstructor.save_pointcloud(state["pcd"], filename)
                    return filepath, f"Point cloud saved as {filepath}"
                    
            except Exception as e:
                logger.error(f"Error saving 3D model: {e}")
                return None, f"Error saving 3D model: {str(e)}"
        
        recon_generate_btn.click(
            generate_3d_model,
            inputs=[recon_input_image, downsample_factor, recon_options],
            outputs=[recon_output, recon_status, gr.State(None)]
        )
        
        recon_save_btn.click(
            save_3d_model,
            inputs=[gr.State(None)],
            outputs=[recon_download, recon_status]
        )

        # Style explorer functionality
        def compare_styles(image, styles):
            if image is None or not styles:
                return """<div class="error">Please select at least one style to explore.</div>"""

            try:
                # Generate prompts from selected styles
                style_prompts = []
                for style_name in styles:
                    style = reimaginer.design_styles.get(style_name)
                    if style:
                        prompt = f"{style.name} style interior"
                        style_prompts.append(prompt)

                # Generate variations for each style
                results = reimaginer.create_batch_variations(
                    original_image=image,
                    style_prompts=style_prompts,
                    style_strength=0.75
                )

                # Create HTML output with style comparisons
                html = """<div class="style-comparison">"""

                for prompt, images in results.items():
                    if not images:
                        continue

                    html += f"""
                    <div class="style-section">
                        <h3>{prompt}</h3>
                        <div class="style-images">
                    """

                    for i, img in enumerate(images):
                        # Save image temporarily and get path
                        img_path = f"temp_{prompt.replace(' ', '_')}_{i}.jpg"
                        img.save(img_path)

                        html += f"""
                        <div class="style-image">
                            <img src="{img_path}" alt="{prompt} variation {i+1}">
                            <p>Variation {i+1}</p>
                        </div>
                        """

                    html += """
                        </div>
                    </div>
                    """

                html += """</div>"""

                return html
            except Exception as e:
                logger.error(f"Error comparing styles: {e}")
                return f"""<div class="error">Error: {str(e)}</div>"""

        style_generate_btn.click(
            compare_styles,
            inputs=[style_input_image, style_options],
            outputs=[style_output]
        )

        # Footer
        with gr.Row(elem_classes="footer"):
            gr.Markdown("""
            ## Tips for Best Results
            - Use well-lit, clear photos of your space
            - Be specific with style descriptions
            - Combine the full room redesign with targeted changes for best results
            - Use the Material Explorer to experiment with different surfaces
            - Save your favorite designs and iterate on them

            Advanced Interior Reimagining AI - Powered by Stable Diffusion, ControlNet, and 3D Reconstruction
            """)

    return ui
