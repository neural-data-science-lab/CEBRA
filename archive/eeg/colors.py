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
    val_clipped = np.clip(valence, -1, 1)
    aro_clipped = np.clip(arousal, -1, 1)
    x, y = val_clipped, aro_clipped

    angle_rad = np.arctan2(y, x)
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    vector_length = np.sqrt(x**2 + y**2) / np.sqrt(2)

    return angle_deg, np.clip(vector_length, 0, 1)


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
    desaturate_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Convert valence-arousal pairs to RGB colors using the emotional color wheel with saturation.

    Args:
        valence (np.ndarray): Array of valence values.
        arousal (np.ndarray): Array of arousal values.
        desaturate_color (Tuple[float, float, float]): RGB color to mix toward for desaturation (default white).

    Returns:
        np.ndarray: Array of RGB colors, shape (len(valence), 3).
    """
    angle, vector_length = compute_angle_vector_length(valence, arousal)
    base_rgb = interpolate_rgb_from_angle(angle)

    # Apply radial saturation: mix toward desaturate_color (usually white)
    final_rgb = (1 - vector_length[:, None]) * desaturate_color + vector_length[:, None] * base_rgb
    return np.clip(final_rgb, 0, 1)
