"""Cinematic render profile for Isaac Sim demo footage.

Isaac Lab boots into a real-time raster path (``rendering_mode="balanced"``,
``/rtx/rendermode="RaytracedLighting"``) because that is what a training loop
wants. It is also why sim footage reads as flat plastic: no true global
illumination, contact shadows that die within a few centimetres of the feet,
and no environment reflections on the robot's shells.

Demo renders have no frame-budget, so they should not pay for that tradeoff.
This module carries the settings that move a shot onto the offline path:

* ``rendering_mode="quality"`` at :class:`~isaaclab.app.AppLauncher`, which
  loads ``IsaacLab/apps/rendering_modes/quality.kit`` (RT2 path-traced
  lighting, shadows, ambient occlusion, DLSS Quality).
* ``/rtx/rendermode = "PathTracing"`` after boot, which is the actual offline
  renderer rather than the interactive approximation of it.
* An HDRI on the dome light, so the shells catch a real environment instead of
  reflecting a constant colour.

Path tracing ACCUMULATES: a frame is only converged once roughly ``totalSpp``
render calls have landed on it, and reading the colour buffer early yields a
noisy frame. Callers must render :func:`settle_frames` times per captured
frame. That is the whole reason this is not the default.

This module is deliberately free of any project-specific import so it can be
dropped into a sibling sim repo unchanged. This file is a verbatim copy of
``go2-phoenix/src/phoenix/demo/render_profile.py``; edit that one and re-copy,
so the two demo pipelines cannot drift into rendering differently.

Setting names are verified against Isaac Lab 4.5.22
(``source/isaaclab/isaaclab/sim/simulation_cfg.py::RenderCfg``) and the Isaac
Sim runtime (``/rtx/rendermode``, ``/rtx/pathtracing/spp``,
``/rtx/pathtracing/totalSpp`` as used by ``omni.replicator.core``).
"""

from __future__ import annotations

import glob
import os

__all__ = [
    "resolve_hdri",
    "launcher_kwargs",
    "enable_path_tracing",
    "settle_frames",
    "STUDIO_HDRI",
    "OUTDOOR_HDRI",
]

# Basenames of HDRIs that ship inside the isaacsim wheel. Bundled rather than
# streamed from the Isaac asset server so a render never depends on the network.
STUDIO_HDRI = "photo_studio_01_4k.hdr"   # neutral studio: hero stills, product look
OUTDOOR_HDRI = "StinsonBeach.hdr"        # bright outdoor sky + sun: exterior shots

# Path tracing converges asymptotically. 64 spp is where a lit interior stops
# showing visible chroma noise in the shadow side of the robot; below ~32 the
# noise survives h264 encoding and reads as a dirty frame.
DEFAULT_SPP = 64


def resolve_hdri(basename: str = STUDIO_HDRI) -> str:
    """Absolute path to a bundled HDRI, or ``""`` if it is not in this install.

    Globbed rather than hard-coded because the ``extscache`` directory that
    holds these is version-pinned in its own path segment, so it moves on every
    Isaac upgrade. Returns a plain string: callers put this on a config object
    that gets deep-copied, and a module handle would not survive that.

    An empty return is not an error. Every caller here treats it as "fall back
    to a flat coloured dome", which is dimmer and less interesting but valid.
    """
    try:
        import isaacsim
    except ImportError:
        return ""
    root = os.path.dirname(isaacsim.__file__)
    hits = glob.glob(f"{root}/**/{basename}", recursive=True)
    # Prefer a domeLight/ copy when one exists: several extensions vendor the
    # same file, and the domeLight variant is the one authored for sky use.
    hits.sort(key=lambda p: ("domeLight" not in p, len(p)))
    return hits[0] if hits else ""


def launcher_kwargs(width: int = 1920, height: int = 1080) -> dict:
    """AppLauncher kwargs for an offline-quality render.

    Merge into the dict passed to :class:`~isaaclab.app.AppLauncher`. This must
    happen at construction: ``rendering_mode`` selects a ``.kit`` preset that is
    read while Kit boots, so setting it afterwards does nothing.

    1080p is the floor for a demo that will be watched full-screen or uploaded;
    below it, YouTube's encoder is handed a soft source and the result looks
    worse than the render actually is.
    """
    return {
        "rendering_mode": "quality",
        "width": width,
        "height": height,
    }


def enable_path_tracing(spp: int = DEFAULT_SPP, denoiser: bool = True) -> bool:
    """Switch the RTX renderer to offline path tracing. Call AFTER app boot.

    ``spp`` sets both the per-frame sample count and the accumulation ceiling,
    so the renderer stops burning GPU once a frame has converged instead of
    refining a frame the caller already captured.

    The OptiX denoiser is left ON by default: at the sample counts that are
    practical for a few hundred frames on a consumer GPU, the denoised frame is
    cleaner than the raw one. Turn it off only for a stills render at very high
    spp, where it can smear fine geometry like the LiDAR housing.

    Returns True if the settings were applied, False if no simulation app is
    running. Note that ``import carb`` SUCCEEDS outside a running app and it is
    ``get_settings()`` that then fails, raising RuntimeError out of the plugin
    loader rather than ImportError, so both have to be caught here. Reported
    rather than raised so a caller can degrade to the real-time path.
    """
    try:
        import carb

        settings = carb.settings.get_settings()
    except (ImportError, RuntimeError):
        return False
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", int(spp))
    settings.set("/rtx/pathtracing/totalSpp", int(spp))
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 1 if denoiser else 0)
    return True


def settle_frames(spp: int = DEFAULT_SPP, path_tracing: bool = True) -> int:
    """Render calls needed per captured frame before the buffer is converged.

    Under path tracing each render call contributes roughly ``spp`` samples up
    to the ``totalSpp`` ceiling, so in principle one call suffices; in practice
    the first call after a camera move still carries reprojected history, and a
    small fixed number of extra calls removes it. Under the real-time path this
    only has to cover TAA/DLSS settling, which is a few frames.
    """
    return 8 if path_tracing else 4
