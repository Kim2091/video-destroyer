"""
Estimating the gamma that makes one picture's tones match another's.

The estimator is VSR-Dataset-Creator's port of chaiNNer's Match Gamma (see
NOTICE.md). Gamma acts on every channel alike, so log(x^g) = g * log(x) holds
for each sample, and the fit works on matched quantiles of the log samples.
Samples at or past 1/255 from black or white carry no information about a
gamma curve, which fixes 0 and 1, so they are left out of the fit.
"""

import math

import numpy as np

CLIP_LOW = 1.0 / 255.0
CLIP_HIGH = 1.0 - CLIP_LOW
# The fit uses the 10th to 90th percentile so tail content cannot dominate.
FIT_QUANTILES = np.linspace(0.1, 0.9, 17)
MIN_FIT_SAMPLES = 32


def collect_log_samples(image, max_samples):
    """
    A 1D array of the log of an image's channel values in (0, 1). The image
    is subsampled first, with a fixed seed, so max_samples bounds the cost of
    the whole estimate and the result is the same on every run.
    """
    pixels = image.reshape(image.shape[0] * image.shape[1], -1)
    if 0 < max_samples < pixels.shape[0]:
        picks = np.sort(np.random.default_rng(0x5EED).integers(0, pixels.shape[0], size=max_samples))
        pixels = pixels[picks]
    flat = pixels.astype(np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    return np.log(flat[(flat > 0.0) & (flat < 1.0)])


def _censored_quantiles(sorted_logs, low, high):
    """Quantiles of the sorted samples strictly inside (low, high), or None when too few remain."""
    start = int(np.searchsorted(sorted_logs, low, side="right"))
    stop = int(np.searchsorted(sorted_logs, high, side="left"))
    count = stop - start
    if count < MIN_FIT_SAMPLES:
        return None
    position = FIT_QUANTILES * (count - 1)
    below = np.floor(position).astype(np.int64)
    fraction = position - below
    return (sorted_logs[start + below] * (1.0 - fraction)
            + sorted_logs[start + np.minimum(below + 1, count - 1)] * fraction)


def _fit(image_logs, reference_logs, gamma):
    """
    One least-squares fit of gamma over matched quantiles, g = <qi, qr> / <qi, qi>.
    Censoring has to stay consistent between the two pictures: content that
    the reference censors at t sits at t^(1/gamma) in the image, so the image
    is censored at the matching bound for the current guess.
    """
    low, high = math.log(CLIP_LOW), math.log(CLIP_HIGH)
    q_image = _censored_quantiles(image_logs, max(low, low / gamma), min(high, high / gamma))
    q_reference = _censored_quantiles(reference_logs, max(low, low * gamma), min(high, high * gamma))
    if q_image is None or q_reference is None:
        return None
    denominator = float(np.dot(q_image, q_image))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    fitted = float(np.dot(q_image, q_reference) / denominator)
    return fitted if math.isfinite(fitted) and fitted > 0.0 else None


def estimate_from_logs(image_logs, reference_logs):
    """
    The gamma mapping the image's samples onto the reference's. The fit is
    repeated because the censoring bounds depend on gamma; it contracts by
    about half each step.
    """
    image_logs, reference_logs = np.sort(image_logs), np.sort(reference_logs)
    gamma = 1.0
    for _ in range(32):
        fitted = _fit(image_logs, reference_logs, gamma)
        if fitted is None:
            break
        converged = abs(fitted - gamma) < 1e-5
        gamma = fitted
        if converged:
            break
    return gamma


def estimate(image, reference, max_samples=1_000_000):
    """The gamma such that image ** gamma matches reference; both are float arrays in 0-1."""
    return estimate_from_logs(collect_log_samples(image, max_samples),
                              collect_log_samples(reference, max_samples))


def apply(image, gamma):
    """image ** gamma on a float array in 0-1; a gamma within 1e-4 of 1 returns the array itself."""
    if abs(gamma - 1.0) < 1e-4:
        return image
    return np.power(np.clip(image, 0.0, 1.0), np.float32(gamma)).astype(np.float32, copy=False)
