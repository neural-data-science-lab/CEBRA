"""
Valence-Arousal to RGB Color Mapping

Provides functions to convert valence-arousal affective dimensions
into visually meaningful RGB colors using angle-vector mapping.

Used for color-encoding emotion in embedding visualizations.

Required packages:
    - numpy
    - matplotlib

Author:
Created: 10.06.2025
Last updated: 21.07.2025
"""


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from typing import Tuple, Union
import numpy as np
import matplotlib.colors as mcolors

# --------------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------------

# Define custom emotional color wheel (angle → color): 
# QI (yellow), QII (red), QIII (blue), QIV (green)
angle_degrees = np.array([45, 135, 225, 315])
color_hex = ['#ffff00', '#ff0000', '#0000ff', '#00ff00']

rgb_colors = np.array([mcolors.to_rgb(c) for c in color_hex])

# Define RdYlGn and RdGy colormaps with 3 colors each (for valence and arousal)
valence_cmap = mcolors.LinearSegmentedColormap.from_list(
    "valence_rdylgn", ["red", "yellow", "green"]
)

arousal_cmap = mcolors.LinearSegmentedColormap.from_list(
    "arousal_rdgy", ["gray", "lightgray", "red"]
)

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def compute_angle_vector_length(
    valence: Union[np.ndarray, float],
    arousal: Union[np.ndarray, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angle in degrees and normalized vector length from valence-arousal values.

    Args:
        valence (Union[np.ndarray, float]): Valence values, expected range [-1, 1].
        arousal (Union[np.ndarray, float]): Arousal values, expected range [-1, 1].

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - angle_deg: Angles in degrees [0, 360).
            - vector_length: Normalized vector lengths [0, 1].
    """
    valence = np.asarray(valence, dtype=float)
    arousal = np.asarray(arousal, dtype=float)

    # Angle in degrees using raw scales
    angle_rad = np.arctan2(arousal, valence)
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    # Vector length = Euclidean distance from origin
    
    vector_length = np.sqrt(valence**2 + arousal**2)
    max_length = np.sqrt(2)
    vector_length = vector_length / max_length
    
    return angle_deg, vector_length


def interpolate_rgb_from_angle(angle_deg: np.ndarray) -> np.ndarray:
    """
    Interpolate RGB colors based on input angles on the emotional color wheel.

    Args:
        angle_deg (np.ndarray): Array of angles in degrees [0, 360).

    Returns:
        np.ndarray: Interpolated RGB colors as array of shape (len(angle_deg), 3).
    """
    angle_deg = np.asarray(angle_deg)
    interpolated_rgb = np.zeros((len(angle_deg), 3))

    for i, angle in enumerate(angle_deg):
        if angle < angle_degrees[0] or angle >= angle_degrees[-1]:
            # Wrap-around case: interpolate between last and first colors
            angle1, angle2 = angle_degrees[-1], angle_degrees[0] + 360
            color1, color2 = rgb_colors[-1], rgb_colors[0]
        else:
            idx = np.searchsorted(angle_degrees, angle) - 1
            idx = np.clip(idx, 0, len(angle_degrees) - 2)
            angle1, angle2 = angle_degrees[idx], angle_degrees[idx + 1]
            color1, color2 = rgb_colors[idx], rgb_colors[idx + 1]

        t = (angle - angle1) / (angle2 - angle1)
        interpolated_rgb[i] = (1 - t) * color1 + t * color2

    return interpolated_rgb


def valence_arousal_emotion_color(
    valence: np.ndarray,
    arousal: np.ndarray,
    desaturate_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    mode: str = "combined"  # "combined" for original angle method, "split" for RdYlGn/RdGy
) -> np.ndarray:
    """
    Convert valence-arousal to RGB color using either original emotional wheel
    or separate RdYlGn (valence) and RdGy (arousal) colormaps blended together.

    Args:
        valence (np.ndarray): Valence values [-1,1].
        arousal (np.ndarray): Arousal values [-1,1].
        desaturate_color (tuple): RGB color to desaturate towards.
        mode (str): "combined" (default) for angle-based; "split" for RdYlGn/RdGy.

    Returns:
        np.ndarray: RGB color array of shape (N, 3).
    """
    valence = np.asarray(valence)
    arousal = np.asarray(arousal)
    if mode == "combined":
        # Original color wheel method
        angle, vector_length = compute_angle_vector_length(valence, arousal)
        base_rgb = interpolate_rgb_from_angle(angle)
        final_rgb = (1 - vector_length[:, None]) * desaturate_color + vector_length[:, None] * base_rgb
        final_rgb = np.clip(final_rgb, 0, 1)
        return final_rgb

    elif mode == "valence":
        return valence_color_rdylgn(valence)
    elif mode == "arousal":
        return arousal_color_rdgy(arousal)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'valence', 'arousal' or 'combined'.")

def valence_color_rdylgn(valence: Union[np.ndarray, float]) -> np.ndarray:
    """
    Map valence values [-1, 1] to RGB colors using RdYlGn colormap (red-yellow-green).

    Args:
        valence (Union[np.ndarray, float]): Valence values in [-1, 1].

    Returns:
        np.ndarray: RGB colors (shape (N, 3) or (3,) for single value).
    """
    valence = np.clip(np.asarray(valence, dtype=float), -1, 1)
    # Normalize to [0,1] for colormap input
    norm_valence = (valence + 1) / 2
    # Get RGB colors from colormap
    rgb = valence_cmap(norm_valence)[..., :3]  # drop alpha
    return rgb

def arousal_color_rdgy(arousal: Union[np.ndarray, float]) -> np.ndarray:
    """
    Map arousal values [-1, 1] to RGB colors using RdGy colormap (gray-lightgray-red).

    Args:
        arousal (Union[np.ndarray, float]): Arousal values in [-1, 1].

    Returns:
        np.ndarray: RGB colors (shape (N, 3) or (3,) for single value).
    """
    arousal = np.clip(np.asarray(arousal, dtype=float), -1, 1)
    norm_arousal = (arousal + 1) / 2
    rgb = arousal_cmap(norm_arousal)[..., :3]
    return rgb

