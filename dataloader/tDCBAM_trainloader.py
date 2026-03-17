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
_MAX_JITTER_PX = 20


# =============================================================================
# AUGMENTATION PARAMS SAMPLING
# =============================================================================

def sample_augment_params():
    """
    Sample one set of geometric augmentation parameters.

    No horizontal flip — signatures are directional and independent flip
    between images in a triplet corrupts the loss signal.

    No affine translation — pre-crop translation is cancelled by the tight
    crop and risks clipping strokes at scan boundaries. Position variation
    is handled instead by canvas-placement jitter after resize.

    Returns
    -------
    dict:
        angle         (float): rotation in degrees    [-15, 15]
        scale         (float): zoom factor            [0.85, 1.15]
        jitter_frac_y (float): vertical placement     [0.0, 1.0]
        jitter_frac_x (float): horizontal placement   [0.0, 1.0]
    """
    return {
        'angle':         random.uniform(-15.0, 15.0),
        'scale':         random.uniform(0.85,  1.15),
        'jitter_frac_y': random.random(),
        'jitter_frac_x': random.random(),
    }


# =============================================================================
# PREPROCESSING & AUGMENTATION
# =============================================================================

def preprocess_image(img, img_size=(224, 224), augment=False, augment_params=None):
    """
    Signature preprocessing pipeline.

    Produces white strokes on a black background (bitwise_not after Otsu).
    This aligns with CBAM's attention mechanism, which highlights high-activation
    regions, and ensures zero-padded canvas borders blend with the background.

    Augmentation modes:
        Triplet-level : augment=False, augment_params=<dict>
                        Shared params across anchor/positive/negative.
        Image-level   : augment=True,  augment_params=None
                        Fresh params sampled per call.
        Inference     : augment=False, augment_params=None
                        No augmentation; signature centred exactly.

    Pipeline:
        1.  PIL / numpy  →  numpy RGB
        2.  Grayscale
        3.  Otsu binarisation    (background=255, strokes=0)
        4.  Stroke inversion     (background=0,   strokes=255)
        5.  Rotation + scale via warpAffine
        6.  Tight crop           findNonZero + 10 px margin
        7.  Aspect-aware resize  scale = target / max(h, w)
        8.  Canvas placement     centred ± _MAX_JITTER_PX px, clamped to slack
        9.  Grayscale  →  RGB
        10. /255  +  ImageNet normalisation

    Parameters
    ----------
    img            : PIL.Image or numpy array.
    img_size       : (H, W) canvas size. Default (224, 224).
    augment        : Apply image-level augmentation.
    augment_params : Pre-sampled dict for triplet-level augmentation.

    Returns
    -------
    torch.Tensor: [3, H, W] float tensor, ImageNet-normalised.
    """
    if img is None:
        return None

    # 1. Ensure numpy RGB
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = np.array(img)

    # 2. Grayscale
    img_gray = (
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    )

    # 3. Otsu binarisation
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Stroke inversion — white strokes on black background
    img_inv = cv2.bitwise_not(thresh)

    # 5. Geometric augmentation
    params = None
    if augment_params is not None:
        params = augment_params
    elif augment:
        params = sample_augment_params()

    if params is not None:
        h, w = img_inv.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])
        img_inv = cv2.warpAffine(
            img_inv, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # 6. Tight crop with 10 px margin
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

    # 7. Aspect-aware resize
    target_size = img_size[0]
    h_c, w_c = img_crop.shape
    scale = target_size / max(h_c, w_c)
    nw = int(w_c * scale)
    nh = int(h_c * scale)

    if nw == 0 or nh == 0:
        img_resized = cv2.resize(img_crop, img_size, interpolation=cv2.INTER_AREA)
        nw, nh = img_size[1], img_size[0]
    else:
        img_resized = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # 8. Canvas placement
    #    jitter_frac=0.5 → centred (inference default).
    #    Other values shift up to ±_MAX_JITTER_PX px around centre,
    #    clamped to available slack so strokes never leave the canvas.
    canvas = np.zeros(img_size, dtype=np.uint8)

    y_slack = target_size - nh
    x_slack = target_size - nw
    y_center = y_slack // 2
    x_center = x_slack // 2

    if params is not None:
        y_delta = int((params['jitter_frac_y'] * 2.0 - 1.0) * _MAX_JITTER_PX)
        x_delta = int((params['jitter_frac_x'] * 2.0 - 1.0) * _MAX_JITTER_PX)
        y_off = max(0, min(y_slack, y_center + y_delta))
        x_off = max(0, min(x_slack, x_center + x_delta))
    else:
        y_off = y_center
        x_off = x_center

    canvas[y_off:y_off + nh, x_off:x_off + nw] = img_resized

    # 9. Grayscale → RGB
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # 10. Float + ImageNet normalisation
    img_float = img_rgb.astype("float32") / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
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
    Returns the image-level transform pipeline for a given mode.

    Used by baseline SplitDataset and proposed SplitEpisodicDataset.
    Not used for SplitTripletDataset training splits (triplet-level aug).

    Parameters
    ----------
    mode        : 'train' | 'val' | 'test'
    input_shape : Canvas size. Default (224, 224).
    preprocess  : True → full signature pipeline. False → resize + normalize.

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
        return transforms.Compose([
            PreprocessTransform(input_shape=input_shape, augment=augment)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# =============================================================================
# LEGACY: DIRECTORY-BASED DATASET (kept for backward compatibility)
# =============================================================================

class SignaturePretrainDataset(Dataset):
    """
    [LEGACY] Triplet dataset reading from org_dir / forg_dir directories.

    The active pipeline uses SplitTripletDataset in proposed_cedar.py.
    Retained for backward compatibility only.
    """

    def __init__(self, org_dir, forg_dir, transform=None, user_list=None):
        self.transform = transform
        self.org_images = []
        self.forg_images = []

        valid_exts = ('.png', '.tif', '.jpg', '.jpeg')

        for root, _, files in os.walk(org_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.org_images.append(os.path.join(root, file))

        for root, _, files in os.walk(forg_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.forg_images.append(os.path.join(root, file))

        if user_list is not None:
            user_list = set(str(u) for u in user_list)
            self.org_images = [
                x for x in self.org_images
                if self._get_user_id(os.path.basename(x)) in user_list
            ]
            self.forg_images = [
                x for x in self.forg_images
                if self._get_user_id(os.path.basename(x)) in user_list
            ]

        self.user_genuine_map = {}
        for path in self.org_images:
            uid = self._get_user_id(os.path.basename(path))
            if uid not in self.user_genuine_map:
                self.user_genuine_map[uid] = []
            self.user_genuine_map[uid].append(path)

        self.users = list(self.user_genuine_map.keys())
        self.triplets = []
        self.on_epoch_end()

    def _get_user_id(self, filename):
        match = re.search(r'\d+', filename)
        if match:
            number = str(int(match.group(0)))
            if 'H-' in filename:
                return f"H-{number}"
            elif 'B-' in filename:
                return f"B-{number}"
            else:
                return number
        return "unknown"

    def on_epoch_end(self):
        self.triplets = []
        all_user_ids = list(self.user_genuine_map.keys())

        for anchor_path in self.org_images:
            anchor_uid = self._get_user_id(os.path.basename(anchor_path))
            positives = self.user_genuine_map.get(anchor_uid, [])

            if len(positives) < 2:
                continue

            possible_pos = [p for p in positives if p != anchor_path]
            if not possible_pos:
                continue
            positive_path = random.choice(possible_pos)

            current_forgeries = [
                f for f in self.forg_images
                if self._get_user_id(os.path.basename(f)) == anchor_uid
            ]
            is_hard_mining = (random.random() < 0.7) and (len(current_forgeries) > 0)

            if is_hard_mining:
                negative_path = random.choice(current_forgeries)
            else:
                other_uid = random.choice([u for u in all_user_ids if u != anchor_uid])
                negatives_from_other = self.user_genuine_map.get(other_uid, [])
                if not negatives_from_other:
                    continue
                negative_path = random.choice(negatives_from_other)

            self.triplets.append((anchor_path, positive_path, negative_path))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img    = Image.open(pos_path).convert('RGB')
        neg_img    = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor_img)
            pos    = self.transform(pos_img)
            neg    = self.transform(neg_img)

        return anchor, pos, neg, torch.tensor([1], dtype=torch.float32)