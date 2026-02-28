import os
import random
import torch
import numpy as np
import cv2
import re
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# =============================================================================
# ULTRA PREPROCESSING (GRADUATED HYBRID STRATEGY)
# =============================================================================

def preprocess_image(img, img_size=(224, 224), augment=False):
    """
    Advanced preprocessing that preserves stroke pressure and removes background noise.
    Matches the 'Ultra' backend for seamless inference.
    """
    if img is None: return None

    # 1. Standardize Input
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 2. Contrast Normalization (CLAHE)
    # Evens out lighting and brings out faint stroke details
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Adaptive Signal Inversion (White Strokes on Black Background)
    # Models like tDCBAM perform better when 'Information' has higher pixel values
    if np.mean(enhanced) > 127:
        work_img = cv2.bitwise_not(enhanced)
    else:
        work_img = enhanced

    # 4. Soft Thresholding (Preserve Ink Gradients)
    # Cleans background but keeps pen pressure/texture
    _, work_img = cv2.threshold(work_img, 30, 255, cv2.THRESH_TOZERO)

    # 5. Morphological Augmentation (Stroke Thickness Variation)
    if augment and random.random() < 0.5:
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.5:
            work_img = cv2.erode(work_img, kernel, iterations=1)
        else:
            work_img = cv2.dilate(work_img, kernel, iterations=1)

    # 6. Tight Bounding Box with Margin
    coords = cv2.findNonZero(work_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 15
        x_s, y_s = max(0, x - margin), max(0, y - margin)
        x_e, y_e = min(work_img.shape[1], x + w + margin), min(work_img.shape[0], y + h + margin)
        img_crop = work_img[y_s:y_e, x_s:x_e]
    else:
        img_crop = work_img

    # 7. Aspect-Preserving Resize
    target_h, target_w = img_size
    h_c, w_c = img_crop.shape
    scale = min(target_w / w_c, target_h / h_c)
    nw, nh = max(1, int(w_c * scale)), max(1, int(h_c * scale))
    img_resized = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # 8. Canvas Placement with Spatial Jitter
    # Prevents the model from relying on perfect centering
    canvas = np.zeros(img_size, dtype=np.uint8)
    if augment:
        y_off = random.randint(0, target_h - nh)
        x_off = random.randint(0, target_w - nw)
    else:
        y_off, x_off = (target_h - nh) // 2, (target_w - nw) // 2
    
    canvas[y_off:y_off+nh, x_off:x_off+nw] = img_resized

    # 9. RGB Conversion and Normalization
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    img_float = img_rgb.astype("float32") / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

# =============================================================================
# REFINED TRANSFORMS
# =============================================================================

def get_pretraining_transforms(input_shape=(224, 224), preprocess=True, augment=True):
    transform_list = []

    if augment:
        transform_list.extend([
            # Degrees reduced to 15 to prevent clipping important stroke tails
            transforms.RandomRotation(degrees=15, fill=0), 
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4, fill=0),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])

    if preprocess:
        transform_list.append(transforms.Lambda(lambda img: preprocess_image(img, img_size=input_shape, augment=augment)))
    else:
        transform_list.extend([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transforms.Compose(transform_list)

# =============================================================================
# DATASET CLASS (ROBUST TRIPLETS)
# =============================================================================

class SignaturePretrainDataset(Dataset):
    def __init__(self, org_dir, forg_dir, transform=None, user_list=None):
        self.transform = transform
        self.org_images = []
        self.forg_images = []
        valid_exts = ('.png', '.tif', '.jpg', '.jpeg')
        
        for root, _, files in os.walk(org_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    self.org_images.append(os.path.join(root, f))
        
        for root, _, files in os.walk(forg_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    self.forg_images.append(os.path.join(root, f))

        if user_list is not None:
            user_list = set(str(u) for u in user_list)
            self.org_images = [x for x in self.org_images if self._get_user_id(os.path.basename(x)) in user_list]
            self.forg_images = [x for x in self.forg_images if self._get_user_id(os.path.basename(x)) in user_list]
        
        self.user_genuine_map = {}
        for path in self.org_images:
            uid = self._get_user_id(os.path.basename(path))
            self.user_genuine_map.setdefault(uid, []).append(path)
            
        self.triplets = []
        self.on_epoch_end()

    def _get_user_id(self, filename):
        match = re.search(r'\d+', filename)
        if match:
            num = str(int(match.group(0)))
            if 'H-' in filename: return f"H-{num}"
            if 'B-' in filename: return f"B-{num}"
            return num
        return "unknown"

    def on_epoch_end(self):
        """Regenerates Triplets with 70% Hard Negative (Skilled Forgery) focus."""
        self.triplets = []
        all_uids = list(self.user_genuine_map.keys())

        for anchor_path in self.org_images:
            uid = self._get_user_id(os.path.basename(anchor_path))
            positives = self.user_genuine_map.get(uid, [])
            if len(positives) < 2: continue
            
            pos_path = random.choice([p for p in positives if p != anchor_path])
            current_forgeries = [f for f in self.forg_images if self._get_user_id(os.path.basename(f)) == uid]
            
            # Hard Mining: Skilled forgeries are prioritized
            if random.random() < 0.7 and len(current_forgeries) > 0:
                neg_path = random.choice(current_forgeries)
            else:
                other_uid = random.choice([u for u in all_uids if u != uid])
                neg_path = random.choice(self.user_genuine_map[other_uid])
            
            self.triplets.append((anchor_path, pos_path, neg_path))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        a_img = Image.open(a_path).convert('RGB')
        p_img = Image.open(p_path).convert('RGB')
        n_img = Image.open(n_path).convert('RGB')

        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)
            
        return a_img, p_img, n_img, torch.tensor([1], dtype=torch.float32)