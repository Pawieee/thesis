import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# =============================================================================
# TRANSFORMATION UTILITIES (AUGMENTATION STRATEGY)
# =============================================================================


def preprocess_image(img, img_size=(224, 224), augment=False):
   """
   Preprocess a signature image using grayscale, Otsu binarization,
   inversion, optional morphological ops, tight cropping, aspect-aware resizing, and padding.
   """
   if img is None:
       return None


   if isinstance(img, Image.Image):
       img = img.convert("RGB")
       img = np.array(img)


   if img.ndim == 3:
       img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   else:
       img_gray = img


   _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   img_inv = cv2.bitwise_not(thresh)

   if augment and random.random() < 0.5:
        kernel = np.ones((2, 2), np.uint8)
        img_inv = cv2.dilate(img_inv, kernel, iterations=1)
  
#    if augment and random.random() < 0.5:
#     kernel = np.ones((2, 2), np.uint8)
    
#     if random.random() < 0.25:                 
#         img_inv = cv2.erode(img_inv, kernel, iterations=1)
#     else:
#         img_inv = cv2.dilate(img_inv, kernel, iterations=1)
          
   coords = cv2.findNonZero(img_inv)
   if coords is not None:
       x, y, w, h = cv2.boundingRect(coords)
       margin = 10
       x_s, y_s = max(0, x - margin), max(0, y - margin)
       w_e, h_e = min(img_inv.shape[1], x + w + margin), min(img_inv.shape[0], y + h + margin)
       img_crop = img_inv[y_s:h_e, x_s:w_e]
   else:
       img_crop = img_inv


   target_size = img_size[0]
   h_c, w_c = img_crop.shape
   scale = target_size / max(h_c, w_c)
   nw, nh = int(w_c * scale), int(h_c * scale)
  
   if nw == 0 or nh == 0:
       img_resized = cv2.resize(img_crop, img_size, interpolation=cv2.INTER_AREA)
       nw, nh = img_size
   else:
       img_resized = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_AREA)


   canvas = np.zeros(img_size, dtype=np.uint8)
   y_off, x_off = (target_size - nh) // 2, (target_size - nw) // 2
   canvas[y_off:y_off+nh, x_off:x_off+nw] = img_resized


   img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
   img_float = img_rgb.astype("float32") / 255.0
  
   tensor = torch.from_numpy(img_float).permute(2, 0, 1)
   norm_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
  
   return norm_tensor


def get_pretraining_transforms(input_shape=(224, 224), preprocess=False, augment=True):
   """
   Generates a pre-training transform pipeline with optional preprocessing and augmentation.
  
   Args:
       input_shape (tuple): Target input resolution (H, W). Default is (224, 224)
       preprocess (bool): Apply grayscale + Otsu + inversion + resize before tensor conversion
       augment (bool): Apply data augmentation (flip/rotation/affine)
      
   Returns:
       torchvision.transforms.Compose: The composition of transforms.
   """
   transform_list = []


   if augment:
       transform_list.extend([
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.RandomRotation(degrees=20, fill=255),
           transforms.RandomAffine(
               degrees=0,
               translate=(0.2, 0.2),
               scale=(0.8, 1.2),
               fill=255
           ),
       ])


   # --- UPDATED: Pass the augment flag into preprocess_image ---
   if preprocess:
       transform_list.append(transforms.Lambda(lambda img: preprocess_image(img, img_size=input_shape, augment=augment)))
   else:
       transform_list.append(transforms.Resize(input_shape))
       transform_list.append(transforms.ToTensor())
       transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


   return transforms.Compose(transform_list)


def get_baseline_transforms(input_shape=(224, 224), preprocess=False, augment=True):
   """
   Generates baseline (classification) transform pipeline with optional preprocessing and augmentation.
  
   Args:
       input_shape (tuple): Target input resolution (H, W). Default is (224, 224)
       preprocess (bool): Apply grayscale + Otsu + inversion + resize before tensor conversion
       augment (bool): Apply data augmentation (flip/rotation/affine)
      
   Returns:
       torchvision.transforms.Compose: Training transforms
       torchvision.transforms.Compose: Validation (no augment) transforms
   """
   train_list = []
   val_list = []


   if augment:
       train_list.extend([
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.RandomRotation(degrees=20, fill=255),
           transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), fill=255),
       ])


   if preprocess:
       train_list.append(transforms.Lambda(lambda img: preprocess_image(img, img_size=input_shape, augment=augment)))
       val_list.append(transforms.Lambda(lambda img: preprocess_image(img, img_size=input_shape, augment=False)))
   else:
       train_list.append(transforms.Resize(input_shape))
       train_list.append(transforms.ToTensor())
       train_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


       val_list.append(transforms.Resize(input_shape))
       val_list.append(transforms.ToTensor())
       val_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


   return transforms.Compose(train_list), transforms.Compose(val_list)


# =============================================================================
# DATASET CLASS WITH HARD MINING
# =============================================================================


class SignaturePretrainDataset(Dataset):
   """
   A PyTorch Dataset class for Triplet-based Pre-training with Online Hard Negative Mining.
  
   This dataset generates triplets (Anchor, Positive, Negative) dynamically.
   It prioritizes 'Hard Negatives' (skilled forgeries of the same user)
   over 'Easy Negatives' (random signatures from other users) to accelerate convergence.
   """
  
   def __init__(self, org_dir, forg_dir, transform=None, user_list=None):
       """
       Initializes the dataset by indexing all signature files.


       Args:
           org_dir (str): Path to the directory containing genuine signatures.
           forg_dir (str): Path to the directory containing forged signatures.
           transform (callable, optional): Transformations to apply to the images.
           user_list (list, optional): Filter specific user IDs (used for splitting Train/Val).
       """
       self.transform = transform
       self.org_images = []
       self.forg_images = []
      
       # --- File Indexing Strategy ---
       # Recursively search for image files to handle nested directory structures
       # (common in BHSig and CEDAR datasets).
       valid_exts = ('.png', '.tif', '.jpg', '.jpeg')
      
       for root, _, files in os.walk(org_dir):
           for file in files:
               if file.lower().endswith(valid_exts):
                    self.org_images.append(os.path.join(root, file))
      
       for root, _, files in os.walk(forg_dir):
           for file in files:
                if file.lower().endswith(valid_exts):
                    self.forg_images.append(os.path.join(root, file))


       # --- User Filtering ---
       if user_list is not None:
           user_list = set(str(u) for u in user_list)
           self.org_images = [x for x in self.org_images if self._get_user_id(os.path.basename(x)) in user_list]
           self.forg_images = [x for x in self.forg_images if self._get_user_id(os.path.basename(x)) in user_list]
      
       # Create a mapping: UserID -> List of Genuine Signature Paths
       self.user_genuine_map = {}
       for path in self.org_images:
           uid = self._get_user_id(os.path.basename(path))
           if uid not in self.user_genuine_map:
               self.user_genuine_map[uid] = []
           self.user_genuine_map[uid].append(path)
          
       self.users = list(self.user_genuine_map.keys())
       self.triplets = []
       self.on_epoch_end() # Initial triplet generation


   def _get_user_id(self, filename):
       """
       Extracts User ID from filename.
       Assumes format like 'original_1_1.png' or '001_01.png'.
       Standardizes extraction using Regex.
       """
       # Matches the first sequence of digits found in the filename
       import re
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
       """
       Regenerates triplets at the end of each epoch.
       This randomizes the pairings to prevent the model from overfitting to specific triplets.
       """
       self.triplets = []
       all_user_ids = list(self.user_genuine_map.keys())


       for anchor_path in self.org_images:
           anchor_uid = self._get_user_id(os.path.basename(anchor_path))
          
           # 1. Select Positive (Another genuine signature from the same user)
           positives = self.user_genuine_map.get(anchor_uid, [])
           # Need at least 2 genuine samples to form a pair
           if len(positives) < 2:
               continue
          
           # Ensure Positive is not the same file as Anchor
           possible_pos = [p for p in positives if p != anchor_path]
           if not possible_pos:
               continue
           positive_path = random.choice(possible_pos)


           # 2. Select Negative (Hard Mining Logic)
           # Strategy:
           # - Hard Negative: A skilled forgery of the SAME user.
           # - Easy Negative: A genuine signature of a DIFFERENT user.
          
           current_forgeries = [f for f in self.forg_images if self._get_user_id(os.path.basename(f)) == anchor_uid]
          
           # Probability threshold: 70% chance to pick a Hard Negative (if available)
           is_hard_mining = (random.random() < 0.7) and (len(current_forgeries) > 0)
          
           if is_hard_mining:
               negative_path = random.choice(current_forgeries)
           else:
               # Pick a random user that is NOT the anchor user
               other_uid = random.choice([u for u in all_user_ids if u != anchor_uid])
               negatives_from_other = self.user_genuine_map.get(other_uid, [])
               if not negatives_from_other: continue
               negative_path = random.choice(negatives_from_other)
          
           self.triplets.append((anchor_path, positive_path, negative_path))


   def __len__(self):
       return len(self.triplets)


   def __getitem__(self, idx):
       """
       Retrieves a triplet item.
       Crucial: Converts images to RGB to match ResNet backbone requirements.
       """
       anchor_path, pos_path, neg_path = self.triplets[idx]
      
       # Load images
       # CONVERT TO RGB: This is critical for ResNet (expects 3 channels)
       anchor_img = Image.open(anchor_path).convert('RGB')
       pos_img = Image.open(pos_path).convert('RGB')
       neg_img = Image.open(neg_path).convert('RGB')


       # Apply Transforms (Augmentation)
       if self.transform:
           anchor = self.transform(anchor_img)
           pos = self.transform(pos_img)
           neg = self.transform(neg_img)
          
       # Return Triplet and a dummy label (TripletLoss doesn't use explicit labels)
       return anchor, pos, neg, torch.tensor([1], dtype=torch.float32)
