"""Server aggregation and Byzantine perturbations for the local simulation."""

import numpy as np
import torch


BR_DRAG_METHODS = ("br-drag", "br-drag-root-direction", "br-drag-root-norm")


def directions(updates, reference):
    norms = torch.linalg.vector_norm(updates, dim=1)
    root_norm = torch.linalg.vector_norm(reference)
    tiny = torch.finfo(updates.dtype).tiny
    unit_updates = updates / norms.clamp_min(tiny)[:, None]
    unit_reference = reference / root_norm.clamp_min(tiny)
    cosine = (unit_updates @ unit_reference).clamp(-1, 1)
    return norms, root_norm, unit_updates, unit_reference, cosine


def divergence_correction(updates, reference, scale, robust):
    """Equations (10)-(11) or (15)-(16), with zero updates mapped to zero."""
    norms, root_norm, units, unit_root, cosine = directions(updates, reference)
    if root_norm == 0:
        return torch.zeros_like(updates) if robust else updates
    dod = (scale * (1 - cosine))[:, None]
    corrected_direction = (1 - dod) * units + dod * unit_root
    if robust:
        magnitude = torch.where(norms > 0, root_norm, 0)
    else:
        magnitude = norms
    return corrected_direction * magnitude[:, None]


def geometric_median(updates, iterations, smoothing, tolerance=0.0):
    center = updates.mean(dim=0)
    for _ in range(iterations):
        distance = torch.linalg.vector_norm(updates - center, dim=1)
        weights = distance.clamp_min(smoothing).reciprocal()
        next_center = (weights / weights.sum()) @ updates
        if tolerance and torch.linalg.vector_norm(next_center - center) <= tolerance:
            return next_center
        center = next_center
    return center


def aggregate(method, updates, reference=None, scale=0.5):
    if method == "drag":
        return divergence_correction(updates, reference, scale, robust=False).mean(dim=0)
    if method in BR_DRAG_METHODS:
        br_update = divergence_correction(updates, reference, scale, robust=True).mean(dim=0)
        if method == "br-drag":
            return br_update
        if method == "br-drag-root-direction":
            direction, length_source = reference, br_update
        else:
            direction, length_source = br_update, reference
        direction_norm = torch.linalg.vector_norm(direction)
        if direction_norm == 0:
            return torch.zeros_like(direction)
        return (direction / direction_norm) * torch.linalg.vector_norm(length_source)
    if method in ("fltrust", "br-trust"):
        _, root_norm, units, _, cosine = directions(updates, reference)
        scores = cosine.clamp_min(0)
        weighted = root_norm * (scores @ units)
        denominator = scores.sum() if method == "fltrust" else updates.shape[0]
        return weighted / torch.as_tensor(denominator, device=updates.device).clamp_min(1e-12)
    if method == "rfa":
        return geometric_median(updates, iterations=5, smoothing=1e-6)
    if method == "raga":
        return geometric_median(updates, iterations=50, smoothing=1e-10, tolerance=1e-7)
    average = updates.mean(dim=0)
    if method == "fedexp":
        step = updates.square().sum() / (2 * len(updates) * (average.square().sum() + 0.001))
        average = average * step.clamp_min(1)
    return average


def distance_stress_update(benign, family):
    """Shared replacement vector for a simulated Min-Max or Min-Sum round."""
    center = benign.mean(dim=0)
    direction = -center.sign()
    if len(benign) == 1 or not torch.any(direction):
        return center
    # Evaluate distances in the low-dimensional span of the perturbation ray.
    offsets = center - benign
    direction = direction / torch.linalg.vector_norm(direction)
    # Calculate Euclidean distances as described in Section VI-B.
    pairwise = torch.cdist(benign, benign, compute_mode="donot_use_mm_for_euclid_dist")
    offset_squares = offsets.square().sum(dim=1)
    cross = 2 * (offsets @ direction)
    limit = pairwise.max() if family == "min-max" else pairwise.sum(dim=1).max()
    upper = limit + offset_squares.sqrt().max()
    lower = torch.zeros_like(upper)
    for _ in range(32):
        midpoint = (lower + upper) / 2
        distances = (offset_squares + midpoint * cross + midpoint.square()).clamp_min(0).sqrt()
        score = distances.max() if family == "min-max" else distances.sum()
        feasible = score <= limit
        lower, upper = torch.where(feasible, midpoint, lower), torch.where(feasible, upper, midpoint)
    return center + lower * direction


def perturb(updates, malicious, family, rng):
    mask = torch.as_tensor(malicious, dtype=torch.bool, device=updates.device)
    count = sum(malicious)
    if not count or family in ("none", "label-flip"):
        return updates
    result = updates.clone()
    if family == "sign-flip":
        result[mask] = -updates[mask]
    elif family == "noise":
        factors = torch.as_tensor(
            rng.normal(0, np.sqrt(3), count), dtype=updates.dtype, device=updates.device
        )
        result[mask] = factors[:, None] * updates[mask]
    elif family in ("min-max", "min-sum"):
        # A round without a benign worker supplies no benign geometry.
        result[mask] = distance_stress_update(updates[~mask], family) if count < len(mask) else 0
    return result
