import os
import re
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# Maximum canvas-placement jitter offset in pixels (applied around centre).
# Jitter displaces the signature ±_MAX_JITTER_PX around the centred position,
# clamped to available slack so strokes never leave the canvas.
_MAX_JITTER_PX = 20


# =============================================================================
# AUGMENTATION PARAMS SAMPLING
# =============================================================================

def sample_augment_params(shared_flip=False):
    """
    Sample one set of geometric augmentation parameters for a single image.

    Called independently per image for rotation, zoom, and jitter —
    this forces the model to learn stroke-level similarity invariant to
    spatial transformation. Flip is sampled once at the triplet level and
    passed in via shared_flip so anchor, positive, and negative are all
    flipped consistently.

    Augmentation ranges match the baseline study (Kandeil et al., 2023):
        rotation  : ±20°   (paper: rotation range 20°)
        zoom      : 0.80–1.20  (paper: zoom range ±20%)
        flip      : 50 % probability  (paper: horizontal flip)

    Width/height shift from the paper is replaced by canvas-placement
    jitter (applied post-resize). Pre-crop translation is cancelled by
    the tight crop and risks clipping strokes at scan boundaries. Jitter
    achieves the same positional variation safely within canvas bounds.

    Parameters
    ----------
    shared_flip : bool
        Horizontal flip decision sampled at the triplet level and shared
        across anchor, positive, and negative. Prevents orientation
        mismatch within a triplet from corrupting the loss signal.

    Returns
    -------
    dict:
        angle         (float) : rotation in degrees    [-20, +20]
        zoom          (float) : zoom factor            [0.80, 1.20]
        jitter_frac_y (float) : vertical jitter        [0.0, 1.0]
        jitter_frac_x (float) : horizontal jitter      [0.0, 1.0]
        flip          (bool)  : horizontal flip — shared across triplet
    """
    return {
        'angle':         random.uniform(-20.0, 20.0),
        'zoom':          random.uniform(0.80,  1.20),
        'jitter_frac_y': random.random(),
        'jitter_frac_x': random.random(),
        'flip':          shared_flip,
    }


# =============================================================================
# PREPROCESSING & AUGMENTATION
# =============================================================================

def preprocess_image(img, img_size=(224, 224), augment=False,
                     augment_params=None):
    """
    Signature preprocessing pipeline — augmentation applied post-crop.

    Produces white strokes on a black background after Otsu inversion.
    This aligns with CBAM attention (highlights high-activation regions)
    and ensures zero-padded canvas borders blend with the background.

    Key design decision: rotation and zoom are applied AFTER tight crop
    and aspect-aware resize. Pre-crop augmentations are cancelled by the
    crop step — the crop re-normalises position regardless of where the
    signature was placed. By augmenting post-crop, the rotation acts on
    actual signature content and zoom controls how much of the 224×224
    canvas the signature occupies.

    Flip is applied PRE-CROP (after inversion) so the full stroke extent
    is preserved before the bounding box is computed.

    Augmentation modes:
        Triplet-level : augment=False, augment_params=<dict>
                        Shared flip + independent rotation/zoom/jitter.
        Image-level   : augment=True,  augment_params=None
                        Fresh params sampled per call (flip random).
        Inference     : augment=False, augment_params=None
                        No augmentation; signature centred exactly.

    Pipeline:
        1.  PIL / numpy  →  numpy RGB
        2.  Grayscale
        3.  Otsu binarisation    (background=255, strokes=0)
        4.  Stroke inversion     (background=0,   strokes=255)
        5.  Horizontal flip      (if params['flip']) — pre-crop
        6.  Tight crop           findNonZero + 10 px margin
        7.  Aspect-aware resize  zoom_target = target × zoom, clamped to target
        8.  Post-crop rotation   warpAffine on resized image, borderValue=0
        9.  Canvas placement     centred ± _MAX_JITTER_PX, clamped to slack
        10. Grayscale  →  RGB
        11. /255  +  ImageNet normalisation

    Parameters
    ----------
    img            : PIL.Image or numpy array.
    img_size       : (H, W) canvas size. Default (224, 224).
    augment        : Apply image-level augmentation (fresh params per call).
    augment_params : Pre-sampled dict from sample_augment_params().
                     Takes precedence over the augment flag.

    Returns
    -------
    torch.Tensor: [3, H, W] float tensor, ImageNet-normalised.
    None if img is None.
    """
    if img is None:
        return None

    # ── 1. Ensure numpy RGB ───────────────────────────────────────────────────
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = np.array(img)

    # ── 2. Grayscale ──────────────────────────────────────────────────────────
    img_gray = (
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    )

    # ── 3. Otsu binarisation ──────────────────────────────────────────────────
    # Adaptive global threshold — white background (255), black strokes (0).
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── 4. Stroke inversion — white strokes on black background ──────────────
    # Inverted so findNonZero directly locates stroke pixels (> 0).
    # Black canvas border blends with background at placement step.
    img_inv = cv2.bitwise_not(thresh)

    # ── Resolve augmentation params ───────────────────────────────────────────
    params = None
    if augment_params is not None:
        params = augment_params
    elif augment:
        params = sample_augment_params(shared_flip=random.random() < 0.5)

    # ── 5. Horizontal flip (pre-crop) ─────────────────────────────────────────
    # Applied before tight crop so the full stroke extent is preserved.
    # The flip decision is shared across the triplet — see SplitTripletDataset.
    # Preserves stroke directionality consistently within each triplet.
    if params is not None and params.get('flip', False):
        img_inv = cv2.flip(img_inv, 1)

    # ── 6. Tight crop with 10 px margin ──────────────────────────────────────
    # findNonZero directly on white-stroke image — strokes are > 0.
    # Bounding box expanded by 10 px on each side, clipped to image bounds.
    coords = cv2.findNonZero(img_inv)
    if coords is not None:
        x, y, w_c, h_c = cv2.boundingRect(coords)
        margin = 10
        x_s = max(0, x - margin)
        y_s = max(0, y - margin)
        x_e = min(img_inv.shape[1], x + w_c + margin)
        y_e = min(img_inv.shape[0], y + h_c + margin)
        img_crop = img_inv[y_s:y_e, x_s:x_e]
    else:
        img_crop = img_inv

    # ── 7. Aspect-aware resize with zoom ─────────────────────────────────────
    # zoom controls what fraction of the 224 canvas the signature occupies:
    #   zoom=1.0 → longest side fills exactly 224 px (baseline behaviour)
    #   zoom=0.8 → longest side fills 179 px — more canvas slack for jitter
    #   zoom=1.2 → clamped to 224 px (same as zoom=1.0 when already maxed)
    # This gives jitter meaningful room to vary position even when the
    # signature would otherwise fill the entire canvas at zoom=1.0.
    target_size = img_size[0]
    h_c, w_c    = img_crop.shape

    zoom         = params['zoom'] if params is not None else 1.0
    zoom_target  = min(int(target_size * zoom), target_size)
    zoom_target  = max(zoom_target, 1)   # guard against degenerate input

    base_scale   = zoom_target / max(h_c, w_c)
    nw           = int(w_c * base_scale)
    nh           = int(h_c * base_scale)

    if nw == 0 or nh == 0:
        img_resized = cv2.resize(img_crop, img_size, interpolation=cv2.INTER_AREA)
        nw, nh = img_size[1], img_size[0]
    else:
        img_resized = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # ── 8. Post-crop rotation ─────────────────────────────────────────────────
    # Rotating into the (nw, nh) canvas directly clips strokes at extreme
    # aspect ratios (e.g. nw=224, nh=70 at 20° loses ~100% of height).
    #
    # Fix: embed the resized image into the full 224×224 work canvas first,
    # rotate around the canvas centre so strokes can expand into the padding
    # without leaving the array, then re-crop to the new stroke bounding box.
    # nw/nh are updated to the post-rotation dimensions for Step 9.
    if params is not None and abs(params['angle']) > 0.1:
        work = np.zeros(img_size, dtype=np.uint8)
        wy   = (target_size - nh) // 2
        wx   = (target_size - nw) // 2
        work[wy:wy + nh, wx:wx + nw] = img_resized

        center_w = (target_size // 2, target_size // 2)
        M        = cv2.getRotationMatrix2D(center_w, params['angle'], 1.0)
        work     = cv2.warpAffine(
            work, M, (target_size, target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        rot_coords = cv2.findNonZero(work)
        if rot_coords is not None:
            rx, ry, rw_c, rh_c = cv2.boundingRect(rot_coords)
            m    = 10
            rx_s = max(0, rx - m);  ry_s = max(0, ry - m)
            rx_e = min(target_size, rx + rw_c + m)
            ry_e = min(target_size, ry + rh_c + m)
            img_resized = work[ry_s:ry_e, rx_s:rx_e]
            nh = img_resized.shape[0]
            nw = img_resized.shape[1]
        else:
            img_resized = work
            nh, nw = target_size, target_size

    # ── 9. Canvas placement ───────────────────────────────────────────────────
    # slack = available space on each axis after placing the resized image.
    # Jitter displaces ±_MAX_JITTER_PX around the centred position,
    # clamped to [0, slack] so strokes never leave the canvas.
    # At inference params=None → exact centre placement.
    canvas   = np.zeros(img_size, dtype=np.uint8)
    y_slack  = max(0, target_size - nh)
    x_slack  = max(0, target_size - nw)
    y_center = y_slack // 2
    x_center = x_slack // 2

    if params is not None:
        y_delta = int((params['jitter_frac_y'] * 2.0 - 1.0) * _MAX_JITTER_PX)
        x_delta = int((params['jitter_frac_x'] * 2.0 - 1.0) * _MAX_JITTER_PX)
        y_off   = max(0, min(y_slack, y_center + y_delta))
        x_off   = max(0, min(x_slack, x_center + x_delta))
    else:
        y_off = y_center
        x_off = x_center

    canvas[y_off:y_off + nh, x_off:x_off + nw] = img_resized

    # ── 10. Grayscale → RGB ───────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # ── 11. Float + ImageNet normalisation ────────────────────────────────────
    img_float   = img_rgb.astype("float32") / 255.0
    tensor      = torch.from_numpy(img_float).permute(2, 0, 1)
    norm_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)

    return norm_tensor


# =============================================================================
# PICKLABLE TRANSFORM WRAPPER
# =============================================================================

class PreprocessTransform:
    """
    Picklable wrapper around preprocess_image for DataLoader workers.

    Replaces transforms.Lambda, which is not picklable under Python 3.14+
    forkserver / Windows spawn multiprocessing.

    Parameters
    ----------
    input_shape (tuple): Canvas size (H, W). Default (224, 224).
    augment     (bool) : Apply random augmentation. Default False.
    """

    def __init__(self, input_shape=(224, 224), augment=False):
        self.input_shape = input_shape
        self.augment     = augment

    def __call__(self, img):
        return preprocess_image(
            img, img_size=self.input_shape, augment=self.augment
        )

    def __repr__(self):
        return (
            f"PreprocessTransform("
            f"input_shape={self.input_shape}, augment={self.augment})"
        )


# =============================================================================
# TRANSFORM FACTORY
# =============================================================================

def get_transforms(mode='train', input_shape=(224, 224), preprocess=True):
    """
    Return the image-level transform pipeline for a given evaluation mode.

    Used by:
        SplitDataset (baseline)      — train / val / test splits.
        SplitPairDataset (proposed)  — val / test splits only.

    Not used for SplitTripletDataset training splits. There,
    preprocess_image() is called directly inside __getitem__ with
    independently sampled augmentation params per image.

    Parameters
    ----------
    mode        : 'train' | 'val' | 'test'
    input_shape : Canvas size (H, W). Default (224, 224).
    preprocess  : True  → full signature binarisation pipeline.
                  False → standard resize + ToTensor + Normalize.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError(
            f"mode must be 'train', 'val', or 'test'. Got: '{mode}'"
        )

    augment = (mode == 'train')

    if preprocess:
        # PreprocessTransform replaces transforms.Lambda — fully picklable
        # under Python 3.14+ forkserver multiprocessing.
        return transforms.Compose([
            PreprocessTransform(input_shape=input_shape, augment=augment)
        ])

    return transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
