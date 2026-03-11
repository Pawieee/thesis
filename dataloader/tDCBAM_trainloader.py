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

def preprocess_image(img, img_size=(224, 224), augment=False):
    """
    Preprocesses a signature image through the full pipeline:

    Pipeline (in order):
        1. Convert to grayscale
        2. Otsu binarization
        3. Stroke inversion (strokes = white, background = black)
        4. [Train only] Morphological augmentation (erode/dilate)
        5. Tight crop with 10px margin
        6. Aspect-aware resize preserving stroke proportions
        7. Zero-padded centering onto 224x224 black canvas
        8. [Train only] Geometric augmentation (rotation, translation, scale)
           Applied AFTER canvas construction on the binary image.
           Fill = 0 (black) to match background — NOT 255.
        9. Convert grayscale canvas to 3-channel RGB (DenseNet compatibility)
       10. Normalize with ImageNet statistics

    Args:
        img: PIL.Image or np.ndarray (RGB or grayscale)
        img_size (tuple): Target canvas size. Default (224, 224).
        augment (bool): Apply morphological + geometric augmentation.
                        Must be False for validation and test.

    Returns:
        torch.Tensor: Normalized tensor of shape [3, H, W]
        or None if input is None.
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
    # Result: strokes = 255 (white), background = 0 (black)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv = cv2.bitwise_not(thresh)

    # --- 4. Morphological augmentation (training only) ---
    # Randomly dilates (thickens) strokes to simulate natural pen pressure
    # variation. Applied with p=0.5.
    # NOTE: Erosion is intentionally excluded. Erosion thins strokes and can
    # fragment fine pen strokes at small kernel sizes, destroying discriminative
    # structural details. Dilation is the safer augmentation — it simulates
    # heavier pen pressure without breaking strokes.
    if augment and random.random() < 0.5:
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img_inv = cv2.dilate(img_inv, kernel, iterations=1)

    # --- 5. Tight crop with margin ---
    coords = cv2.findNonZero(img_inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 10
        x_s = max(0, x - margin)
        y_s = max(0, y - margin)
        x_e = min(img_inv.shape[1], x + w + margin)
        y_e = min(img_inv.shape[0], y + h + margin)
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
    # Background = 0 (black), strokes = 255 (white)
    canvas = np.zeros(img_size, dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = img_resized

    # --- 8. Geometric augmentation on canvas (training only) ---
    # Applied AFTER canvas construction on the binary image.
    # fill=0 matches the black background — using fill=255 would
    # paint empty regions white (same as strokes), corrupting boundaries.
    #
    # Parameters chosen for realism:
    #   Rotation:    ±15°    (reduced from ±20° — extreme rotation is unrealistic)
    #   Translation: ±10%    (reduced from ±20% — large shifts push strokes off-canvas)
    #   Scale:       90-110% (tightened from 80-120% — extreme scale distorts proportions)
    #
    # NOTE: RandomHorizontalFlip is intentionally excluded.
    # A horizontally flipped signature is not a valid real-world sample —
    # no writer produces a mirror image of their own signature.
    if augment:
        h_canvas, w_canvas = canvas.shape
        center = (w_canvas // 2, h_canvas // 2)

        # Random rotation ±15°
        angle = random.uniform(-15, 15)

        # Random scale 90–110%
        scale_factor = random.uniform(0.90, 1.10)

        # Rotation + scale matrix
        M_rot = cv2.getRotationMatrix2D(center, angle, scale_factor)

        # Random translation ±10% of canvas size
        tx = random.uniform(-0.10, 0.10) * w_canvas
        ty = random.uniform(-0.10, 0.10) * h_canvas
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply with BORDER_CONSTANT fill=0 (black background)
        canvas = cv2.warpAffine(
            canvas, M_rot, (w_canvas, h_canvas),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # --- 9. Grayscale → RGB (DenseNet requires 3-channel input) ---
    # All three channels are identical since the source is binary grayscale.
    # This is necessary for ImageNet-pretrained DenseNet-121 compatibility.
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # --- 10. Float conversion + ImageNet normalization ---
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
        # Full signature preprocessing pipeline with optional augmentation.
        # All augmentation (morphological + geometric) is handled inside
        # preprocess_image() AFTER canvas construction — not before binarization.
        return transforms.Compose([
            transforms.Lambda(
                lambda img: preprocess_image(img, img_size=input_shape, augment=augment)
            )
        ])
    else:
        # Minimal pipeline for raw RGB input without binarization.
        # Augmentation is intentionally excluded here since this path
        # bypasses the binary canvas where augmentation is safe to apply.
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