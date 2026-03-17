import random
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


# =============================================================================
# AUGMENTATION PARAMETER SAMPLING
# =============================================================================

def sample_augment_params():
    """
    Sample one set of geometric augmentation parameters for a single image.

    Called independently for each image in a triplet (anchor, positive,
    negative). Independent sampling forces the model to learn stroke-level
    similarity invariant to spatial transformation, rather than relying on
    co-registration between images.

    Horizontal flipping is deliberately excluded to preserve stroke
    directionality — a discriminative feature in handwritten signatures.

    Parameter bounds simulate realistic handwriting variability:

        angle         [-15°, +15°]   Natural pen slant variation.
        scale         [0.85, 1.15]   Natural size and pressure variation.
        jitter_frac_y [0.0,  1.0]    Vertical canvas placement fraction.
        jitter_frac_x [0.0,  1.0]    Horizontal canvas placement fraction.

    Returns:
        dict: Keys — 'angle', 'scale', 'jitter_frac_y', 'jitter_frac_x'.
    """
    return {
        'angle':         random.uniform(-15, 15),
        'scale':         random.uniform(0.85, 1.15),
        'jitter_frac_y': random.random(),
        'jitter_frac_x': random.random(),
    }


# =============================================================================
# SIGNATURE PREPROCESSING PIPELINE
# =============================================================================

def preprocess_image(img, img_size=(224, 224), augment=False,
                     augment_params=None):
    """
    Full preprocessing pipeline for offline signature images.

    Produces a normalized RGB tensor suitable for DenseNet-121 with
    ImageNet pretrained weights. The pipeline preserves white background
    (255) and black strokes (0) throughout all intermediate stages.

    Pipeline stages:
        1. PIL → numpy RGB conversion.
        2. Grayscale conversion via OpenCV COLOR_RGB2GRAY.
        3. Otsu binarization — adaptive global threshold separates
           ink strokes from background without manual threshold tuning.
           Output: white background (255), black strokes (0).
        4. Geometric augmentation (if augment=True or augment_params given):
           Combined rotation + scale via affine transform. Border regions
           filled with white (255) to match background. Horizontal flip
           excluded to preserve stroke directionality.
        5. Tight crop with 10px margin — bounding box of stroke pixels
           found via cv2.findNonZero on the inverted image (strokes > 0),
           then expanded by 10px on each side and clipped to canvas bounds.
        6. Aspect-aware resize — longest dimension scaled to target_size,
           shorter dimension scaled proportionally. Prevents signature
           distortion from non-uniform scaling.
        7. Canvas placement on 224×224 white canvas — centered for
           inference, jitter-offset during augmentation to simulate
           natural placement variation.
        8. Gaussian noise (σ=5.0) added during augmentation only — breaks
           pure white/zero-padding artifacts that could become discriminative
           features if left as perfect constants.
        9. Grayscale → RGB conversion + float normalization + ImageNet
           standardization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).

    Args:
        img           (PIL.Image or np.ndarray): Input signature image.
        img_size      (tuple): Target canvas size (H, W). Default (224, 224).
        augment       (bool) : If True, sample and apply random augmentation.
                               Ignored if augment_params is provided.
        augment_params(dict) : Pre-sampled augmentation parameters from
                               sample_augment_params(). Takes precedence
                               over the augment flag when provided.

    Returns:
        torch.Tensor: Normalized tensor of shape [3, H, W], dtype float32.
        None: If img is None.
    """
    if img is None:
        return None

    # ── 1. PIL → numpy RGB ────────────────────────────────────────────────────
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))

    # ── 2. Grayscale conversion ───────────────────────────────────────────────
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()

    # ── 3. Otsu binarization ──────────────────────────────────────────────────
    # Adaptive global threshold — white background (255), black strokes (0).
    _, img_binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── 4. Geometric augmentation ─────────────────────────────────────────────
    params = augment_params if augment_params is not None else (
        sample_augment_params() if augment else None
    )

    if params is not None:
        h, w  = img_binary.shape
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])
        img_binary = cv2.warpAffine(
            img_binary, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255         # fill new regions with white background
        )

    # ── 5. Tight crop with 10px margin ────────────────────────────────────────
    # findNonZero operates on pixel values > 0, so the image is inverted
    # temporarily to locate black stroke coordinates (strokes become 255).
    coords = cv2.findNonZero(cv2.bitwise_not(img_binary))

    if coords is not None:
        x, y, w_c, h_c = cv2.boundingRect(coords)
        margin = 10
        x_s = max(0, x - margin)
        y_s = max(0, y - margin)
        x_e = min(img_binary.shape[1], x + w_c + margin)
        y_e = min(img_binary.shape[0], y + h_c + margin)
        img_crop = img_binary[y_s:y_e, x_s:x_e]
    else:
        img_crop = img_binary

    # ── 6. Aspect-aware resize ────────────────────────────────────────────────
    target_size = img_size[0]
    h_c, w_c    = img_crop.shape
    scale        = target_size / max(h_c, w_c)
    nw           = int(w_c * scale)
    nh           = int(h_c * scale)

    if nw == 0 or nh == 0:
        # Degenerate case — force resize to target and reset dimensions
        img_resized = cv2.resize(img_crop, img_size,
                                  interpolation=cv2.INTER_AREA)
        nw, nh = img_size[1], img_size[0]
    else:
        img_resized = cv2.resize(img_crop, (nw, nh),
                                  interpolation=cv2.INTER_AREA)

    # ── 7. Canvas placement ───────────────────────────────────────────────────
    canvas  = np.full(img_size, 255, dtype=np.uint8)   # white background
    y_slack = max(0, target_size - nh)
    x_slack = max(0, target_size - nw)

    if params is not None:
        y_off = int(params['jitter_frac_y'] * y_slack)
        x_off = int(params['jitter_frac_x'] * x_slack)
    else:
        y_off = y_slack // 2    # center for inference
        x_off = x_slack // 2

    canvas[y_off:y_off + nh, x_off:x_off + nw] = img_resized

    # ── 8. Gaussian noise (augmentation only) ────────────────────────────────
    # Breaks pure white/zero-padding artifacts that could otherwise
    # become spurious discriminative features learned during training.
    if params is not None:
        noise  = np.random.normal(loc=0, scale=5.0, size=canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise,
                         0, 255).astype(np.uint8)

    # ── 9. Grayscale → RGB + float + ImageNet normalization ──────────────────
    img_rgb   = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    img_float = img_rgb.astype("float32") / 255.0
    tensor    = torch.from_numpy(img_float).permute(2, 0, 1)

    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)


# =============================================================================
# TRANSFORM FACTORY
# =============================================================================

def get_transforms(mode='train', input_shape=(224, 224), preprocess=True):
    """
    Return the image-level transform pipeline for a given evaluation mode.

    Used by:
        SplitDataset (baseline)      — train / val / test splits.
        SplitPairDataset (proposed)  — val / test splits only.

    Not used for augmentation in SplitTripletDataset's training split.
    There, preprocess_image(augment=True) is called directly inside
    __getitem__ with independently sampled parameters per image.

    Args:
        mode        (str)  : 'train' → augmentation ON (for baseline classifier).
                             'val' / 'test' → preprocessing only, no augmentation.
        input_shape (tuple): Target canvas size (H, W). Default (224, 224).
        preprocess  (bool) : True  → full signature binarization pipeline
                                     via preprocess_image().
                             False → standard resize + ToTensor + Normalize
                                     for raw RGB inputs (no binarization).

    Returns:
        torchvision.transforms.Compose
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError(
            f"mode must be 'train', 'val', or 'test'. Got: '{mode}'"
        )

    augment = (mode == 'train')

    if preprocess:
        return transforms.Compose([
            transforms.Lambda(
                lambda img: preprocess_image(
                    img, img_size=input_shape, augment=augment
                )
            )
        ])

    return transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])