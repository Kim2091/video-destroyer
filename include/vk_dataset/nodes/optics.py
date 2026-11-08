"""
Lens, focus and camera softness: one 2-D blur kernel per chunk, convolved
with every frame.

The kernel shapes are the family from Real-ESRGAN (Wang et al., "Real-ESRGAN:
Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCVW
2021, section 3.1), written here from the paper's formulas. With C the pixel
offset and Sigma the covariance R diag(s1^2, s2^2) R^T of a kernel rotated by
an angle:

    gaussian     exp(-1/2 C' Sigma^-1 C)
    generalized  exp(-1/2 (C' Sigma^-1 C)^beta)      beta < 1 has long tails,
                                                     beta > 1 a flat top
    plateau      1 / (1 + (C' Sigma^-1 C)^beta)      a flat top with tails

each normalised to sum 1. Anisotropic kernels have s2 = s1 / ratio. Blur is
done in linear light by default, as optics blur light, not coded values.
"""

import math

import cv2
import numpy as np
import vapoursynth as vs

from ..framefx import modify_planes
from ..registry import Param, node
from .mixes import LIGHTS, MIX, draw, draw_light, from_working, to_working

core = vs.core

# Real-ESRGAN's kernel weights, with each shape's isotropic and
# anisotropic variants merged and anisotropy drawn separately.
SHAPES = {"gaussian": 0.7, "generalized": 0.15, "plateau": 0.15}
MAX_SIZE = 21


def blur_kernel(shape, sigma, ratio, angle, beta):
    """The normalised kernel as a float32 array of odd size."""
    tails = 3.0 if shape == "gaussian" else 4.0
    size = MAX_SIZE if shape == "plateau" else min(MAX_SIZE, max(3, 2 * math.ceil(tails * sigma) + 1))
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    s1, s2 = sigma, sigma / max(1.0, ratio)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    inverse = np.linalg.inv(rotation @ np.diag([s1 * s1, s2 * s2]) @ rotation.T)
    distance = inverse[0, 0] * x * x + (inverse[0, 1] + inverse[1, 0]) * x * y + inverse[1, 1] * y * y
    if shape == "gaussian":
        kernel = np.exp(-0.5 * distance)
    elif shape == "generalized":
        kernel = np.exp(-0.5 * distance ** beta)
    elif shape == "plateau":
        kernel = 1.0 / (1.0 + distance ** beta)
    else:
        raise ValueError(f"unknown kernel shape {shape!r}; known: {', '.join(SHAPES)}")
    return (kernel / kernel.sum()).astype(np.float32)


@node("optics_blur", category="Degrade", label="Lens Blur", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("shape", "choice", MIX, choices=[MIX, *SHAPES],
                help="Kernel shape; 'mix' picks per chunk as Real-ESRGAN does."),
          Param("sigma", "float", {"uniform": [0.2, 1.2]}, random=True, help="Blur size in pixels.",
                min=0.1, max=5.0),
          Param("anisotropy", "float", {"choice": [1.0, 1.5, 2.0, 3.0], "weights": [0.6, 0.15, 0.15, 0.1]}, random=True,
                help="1 is round; higher squeezes the blur along one axis.", min=1.0, max=6.0),
          Param("beta", "float", {"uniform": [0.5, 2.0]}, random=True,
                help="Generalized and plateau shapes: below 1 long tails, above 1 a flat top (plateau 1 or more).",
                min=0.3, max=4.0),
          Param("light", "choice", "mostly linear", choices=LIGHTS, help="Blur light or coded values."),
      ])
def optics_blur(ctx, clip, *, shape, sigma, anisotropy, beta, light):
    rng = ctx.rng
    chosen = draw(rng, shape, SHAPES)
    angle = rng.uniform(0.0, math.pi)
    working_light = draw_light(rng, light)
    if chosen == "plateau":
        beta = max(1.0, beta)
    kernel = blur_kernel(chosen, sigma, anisotropy, angle, beta)
    ctx.note("shape", chosen)
    ctx.note("sigma", round(sigma, 3))
    ctx.note("anisotropy", round(anisotropy, 3))
    ctx.note("angle", round(math.degrees(angle), 1))
    if chosen != "gaussian":
        ctx.note("beta", round(beta, 3))
    ctx.note("light", working_light)
    ctx.note("kernel_size", kernel.shape[0])

    working = to_working(clip, working_light)

    def func(n, plane, arrays):
        return cv2.filter2D(arrays[0], -1, kernel, borderType=cv2.BORDER_REFLECT)
    blurred = modify_planes(working, range(3), func)
    return {"clip": from_working(blurred, clip, working_light)}
