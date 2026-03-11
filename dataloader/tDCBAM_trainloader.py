import os
import re
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# =============================================================================
# PREPROCESSING & AUGMENTATION
# =============================================================================

import random
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(img, img_size=(224, 224), augment=False):
    """
    Preprocesses a signature image through the full pipeline:

    Pipeline (in order):
        1. Convert to grayscale
        2. Otsu binarization
        3. Stroke inversion (strokes = white, background = black)
        4. [Train only] Geometric augmentation (flip, rotation, translation, scale)
           *Applied BEFORE cropping to prevent stroke cut-off.*
        5. Tight crop with 10px margin
        6. Aspect-aware resize preserving stroke proportions
        7. Zero-padded centering onto 224x224 black canvas
        8. Convert grayscale canvas to 3-channel RGB
        9. Normalize with ImageNet statistics
    """
    if img is None:
        return None

    # --- 1. Ensure numpy RGB array ---
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = np.array(img)

    # --- 2. Grayscale conversion ---
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()

    # --- 3. Otsu binarization + stroke inversion ---
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv = cv2.bitwise_not(thresh)

    # --- 4. Geometric Augmentation (training only) ---
    # Applied BEFORE the tight crop so that transformations do not push
    # the signature strokes outside the final 224x224 canvas boundaries.
    if augment:
        h, w = img_inv.shape

        # Horizontal Flip (50% probability)
        if random.random() < 0.5:
            img_inv = cv2.flip(img_inv, 1)

        # Geometric Parameters:
        # Rotation: ±20 degrees
        # Scale (Zoom): ±20% (0.80 to 1.20)
        # Translation (Shift): ±20% of width and height
        angle = random.uniform(-20, 20)
        scale_factor = random.uniform(0.80, 1.20)
        
        center = (w // 2, h // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, scale_factor)

        tx = random.uniform(-0.20, 0.20) * w
        ty = random.uniform(-0.20, 0.20) * h
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply transformations with black background padding (fill=0)
        img_inv = cv2.warpAffine(
            img_inv, M_rot, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # --- 5. Tight crop with margin ---
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

    # --- 6. Aspect-aware resize ---
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

    # --- 7. Zero-padded centering onto black canvas ---
    canvas = np.zeros(img_size, dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = img_resized

    # --- 8. Grayscale → RGB ---
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # --- 9. Float conversion + ImageNet normalization ---
    img_float = img_rgb.astype("float32") / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    norm_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)

    return norm_tensor

# =============================================================================
# TRANSFORM FACTORY
# =============================================================================

def get_transforms(mode='train', input_shape=(224, 224), preprocess=True):
    """
    Returns the appropriate transform pipeline for a given mode.

    Args:
        mode (str): One of 'train', 'val', or 'test'.
                    'train' applies augmentation.
                    'val' and 'test' apply preprocessing only (no augmentation).
        input_shape (tuple): Target image size. Default (224, 224).
        preprocess (bool): If True, applies the full signature preprocessing
                           pipeline via preprocess_image(). If False, applies
                           standard resize + ToTensor + Normalize (for raw RGB
                           pipelines that skip binarization). Default True.

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.

    Example:
        train_transform = get_transforms(mode='train')
        val_transform   = get_transforms(mode='val')
        test_transform  = get_transforms(mode='test')
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError(f"mode must be 'train', 'val', or 'test'. Got: '{mode}'")

    augment = (mode == 'train')

    if preprocess:
        return transforms.Compose([
            transforms.Lambda(
                lambda img: preprocess_image(img, img_size=input_shape, augment=augment)
            )
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
    [LEGACY] Triplet dataset that reads from org_dir / forg_dir directories.

    This class is retained for backward compatibility with directory-based
    pipelines. The active training pipeline uses SplitTripletDataset (in
    proposed_cedar.py) which reads from split JSON user dictionaries instead.

    For new experiments, use SplitTripletDataset + get_transforms().
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
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor_img)
            pos = self.transform(pos_img)
            neg = self.transform(neg_img)

        return anchor, pos, neg, torch.tensor([1], dtype=torch.float32)