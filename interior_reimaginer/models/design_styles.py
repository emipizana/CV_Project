from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class DesignStyle:
    """Definition of an interior design style"""
    name: str
    description: str
    prompt_modifiers: List[str]
    negative_modifiers: List[str]
    example_image: Optional[str] = None
    color_palette: Optional[List[str]] = None

def load_design_styles() -> Dict[str, DesignStyle]:
    """Load predefined interior design styles"""
    styles = {}

    # Minimalist style
    styles["minimalist"] = DesignStyle(
        name="Minimalist",
        description="Clean lines, minimal decoration, and functional furniture",
        prompt_modifiers=[
            "minimalist interior", "clean lines", "uncluttered",
            "functional furniture", "neutral colors", "simple shapes"
        ],
        negative_modifiers=[
            "ornate", "cluttered", "busy patterns", "excessive decoration"
        ],
        color_palette=["#FFFFFF", "#F5F5F5", "#E0E0E0", "#BDBDBD", "#212121"]
    )

    # Scandinavian style
    styles["scandinavian"] = DesignStyle(
        name="Scandinavian",
        description="Light, airy spaces with wooden elements and cozy textiles",
        prompt_modifiers=[
            "scandinavian interior", "light wood", "white walls",
            "cozy textiles", "natural light", "functional design"
        ],
        negative_modifiers=[
            "dark colors", "heavy furniture", "ornate decoration"
        ],
        color_palette=["#FFFFFF", "#F9F9F9", "#E0E0E0", "#D7CCC8", "#BCAAA4"]
    )

    # Industrial style
    styles["industrial"] = DesignStyle(
        name="Industrial",
        description="Raw materials, exposed structures, and vintage elements",
        prompt_modifiers=[
            "industrial interior", "exposed brick", "metal fixtures",
            "concrete floors", "raw materials", "factory-inspired"
        ],
        negative_modifiers=[
            "colorful", "ornate", "delicate", "formal"
        ],
        color_palette=["#212121", "#616161", "#9E9E9E", "#BDBDBD", "#37474F"]
    )

    # Mid-Century Modern style
    styles["mid_century"] = DesignStyle(
        name="Mid-Century Modern",
        description="Retro furniture, clean lines, and bold accent colors",
        prompt_modifiers=[
            "mid-century modern interior", "retro furniture", "clean lines",
            "organic curves", "bold accent colors", "1950s inspired"
        ],
        negative_modifiers=[
            "rustic", "traditional", "ornate", "cluttered"
        ],
        color_palette=["#FFCC80", "#A1887F", "#90CAF9", "#FFF176", "#A5D6A7"]
    )

    # Bohemian style
    styles["bohemian"] = DesignStyle(
        name="Bohemian",
        description="Eclectic, colorful spaces with mixed patterns and textures",
        prompt_modifiers=[
            "bohemian interior", "eclectic", "colorful textiles",
            "mixed patterns", "plants", "layered textures", "global influences"
        ],
        negative_modifiers=[
            "minimal", "monochrome", "formal", "structured"
        ],
        color_palette=["#FF8A65", "#FFD54F", "#81C784", "#4DB6AC", "#7986CB"]
    )

    # Traditional style
    styles["traditional"] = DesignStyle(
        name="Traditional",
        description="Classic furniture, rich colors, and elegant details",
        prompt_modifiers=[
            "traditional interior", "classic furniture", "rich colors",
            "elegant details", "symmetry", "refined", "vintage", "rural", "wood"
        ],
        negative_modifiers=[
            "modern", "industrial", "minimalist", "casual", "metal", "lights"
        ],
        color_palette=["#5D4037", "#3E2723", "#BF360C", "#827717", "#1B5E20"]
    )

    return styles