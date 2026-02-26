import os
import sys
import io
import cv2
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from models.feature_extractor import DenseNetFeatureExtractor
from models.meta_learner import MetricGenerator

app = FastAPI(title="Signature Verification API - 3x K=1 Protocol")

app.add_middleware(
    CORSMiddleware,
    # Updated to include common dev ports and your Vercel frontend URL
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
FEATURE_DIM = 1024
EMBEDDING_DIM = 2048
BASELINE_NUM_CLASSES = 2
DEFAULT_THRESHOLD = 0.5

# Centralized Weights Directory
WEIGHTS_DIR = os.path.join(REPO_ROOT, 'weights')

model_cache = {}
baseline_cache = {}

def load_proposed_models(dataset_name: str, split_ratio: str):
    cache_key = f"{dataset_name}_{split_ratio}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    # Target: /backend/weights/proposed_splits/{dataset}_{split}/
    checkpoint_dir = os.path.join(WEIGHTS_DIR, 'proposed_splits', cache_key)
    backbone_path = os.path.join(checkpoint_dir, 'pretrained_backbone.pth')
    metric_path = os.path.join(checkpoint_dir, 'best_meta_model.pth')

    if not os.path.exists(backbone_path) or not os.path.exists(metric_path):
        raise FileNotFoundError(f"Proposed checkpoints not found in {checkpoint_dir}")

    feature_extractor = DenseNetFeatureExtractor('densenet121', FEATURE_DIM, pretrained=False, baseline=False)
    feature_extractor.load_state_dict(torch.load(backbone_path, map_location=DEVICE, weights_only=True))
    feature_extractor.to(DEVICE).eval()

    metric_generator = MetricGenerator(EMBEDDING_DIM)
    full_checkpoint = torch.load(metric_path, map_location=DEVICE, weights_only=False)

    learned_threshold = DEFAULT_THRESHOLD
    if isinstance(full_checkpoint, dict):
        metric_generator.load_state_dict(full_checkpoint.get('metric_generator', full_checkpoint))
        if 'metrics' in full_checkpoint and 'threshold' in full_checkpoint['metrics']:
            learned_threshold = float(full_checkpoint['metrics']['threshold'])
    else:
        metric_generator.load_state_dict(full_checkpoint)

    metric_generator.to(DEVICE).eval()
    model_cache[cache_key] = (feature_extractor, metric_generator, learned_threshold)
    return model_cache[cache_key]

def load_baseline_model(dataset_name: str, split_ratio: str):
    if dataset_name == 'combined':
        return None, DEFAULT_THRESHOLD
        
    cache_key = f"{dataset_name}_{split_ratio}"
    if cache_key in baseline_cache:
        return baseline_cache[cache_key]

    # Target: /backend/weights/baseline_splits/best_{dataset}_{split}.pth
    checkpoint_dir = os.path.join(WEIGHTS_DIR, 'baseline_splits')
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{dataset_name}_{split_ratio}.pth")

    if not os.path.exists(checkpoint_path):
        return None, DEFAULT_THRESHOLD

    baseline_model = DenseNetFeatureExtractor('densenet121', BASELINE_NUM_CLASSES, pretrained=False, baseline=True)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    baseline_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    baseline_model.to(DEVICE).eval()

    learned_threshold = DEFAULT_THRESHOLD
    if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
        learned_threshold = float(checkpoint['metrics'].get('threshold', DEFAULT_THRESHOLD))

    baseline_cache[cache_key] = (baseline_model, learned_threshold)
    return baseline_cache[cache_key]

def preprocess_image_with_steps(image_input) -> tuple[torch.Tensor, dict]:
    """Processes image and returns the final tensor plus all intermediate steps."""
    steps = {}
    
    # 1. Load
    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        image = image_input.convert("RGB")
    
    img = np.array(image)
    steps['1. Original'] = img
    
    # 2. Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    steps['2. Grayscale'] = img_gray

    # 3. Binarization
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv = cv2.bitwise_not(thresh)
    steps['3. Binarized'] = img_inv

    # 4. Tight Crop
    coords = cv2.findNonZero(img_inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 10
        x_s, y_s = max(0, x-margin), max(0, y-margin)
        w_e, h_e = min(img_inv.shape[1], x+w+margin), min(img_inv.shape[0], y+h+margin)
        img_crop = img_inv[y_s:h_e, x_s:w_e]
    else:
        img_crop = img_inv
    steps['4. Tight Crop'] = img_crop

    # 5. Aspect-Aware Resize
    h_c, w_c = img_crop.shape
    scale = IMAGE_SIZE / max(h_c, w_c)
    nw, nh = int(w_c * scale), int(h_c * scale)
    img_resized = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_AREA)
    steps['5. Resized'] = img_resized

    # 6. Padding
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y_off, x_off = (IMAGE_SIZE - nh) // 2, (IMAGE_SIZE - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = img_resized
    steps['6. Final Canvas'] = canvas

    # 7. Normalize
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    img_float = img_rgb.astype("float32") / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    norm_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    
    return norm_tensor.unsqueeze(0).to(DEVICE), steps


@app.post("/verify")
async def verify_signature(
    dataset: str = Form(...),
    split: str = Form(...),
    support_file_1: UploadFile = File(...),
    support_file_2: UploadFile = File(...),
    support_file_3: UploadFile = File(...),
    query_file: UploadFile = File(...)
):
    try:
        # Load Models
        f_extractor, m_generator, threshold = load_proposed_models(dataset, split)
        b_model, b_threshold = load_baseline_model(dataset, split)

        # Preprocess Tensors (We extract index [0] to get just the tensor, ignoring the steps dict)
        s_tensors = [
            preprocess_image_with_steps(await support_file_1.read())[0],
            preprocess_image_with_steps(await support_file_2.read())[0],
            preprocess_image_with_steps(await support_file_3.read())[0]
        ]
        q_tensor = preprocess_image_with_steps(await query_file.read())[0]

        # ================== PROPOSED MODEL (3x K=1) ==================
        t0_prop = time.time()
        with torch.no_grad():
            s_feats = [f_extractor(t).squeeze(0) for t in s_tensors]
            q_feat = f_extractor(q_tensor).squeeze(0)
            
            per_support = []
            genuine_votes = 0
            
            for idx, s_feat in enumerate(s_feats, 1):
                combined = torch.cat([s_feat, q_feat], dim=0).unsqueeze(0)
                p_genuine = torch.sigmoid(m_generator(combined)).item()
                p_forged = 1.0 - p_genuine
                pred = "GENUINE" if p_genuine >= threshold else "FORGED"
                if pred == "GENUINE": genuine_votes += 1
                
                per_support.append({
                    "support": f"Support {idx}",
                    "p_genuine": round(p_genuine, 4),
                    "p_forged": round(p_forged, 4),
                    "prediction": pred
                })

        processing_time_proposed = time.time() - t0_prop
        
        final_prediction = "GENUINE" if genuine_votes >= 2 else "FORGED"
        avg_p_gen = float(np.mean([item["p_genuine"] for item in per_support]))
        avg_p_forg = 1.0 - avg_p_gen
        vote_confidence = (max(genuine_votes, 3 - genuine_votes) / 3.0) * 100

        proposed_result = {
            "prediction": final_prediction,
            "vote_confidence": round(vote_confidence, 2),
            "avg_p_genuine": round(avg_p_gen, 4),
            "avg_p_forged": round(avg_p_forg, 4),
            "threshold": round(threshold, 4),
            "processing_time": round(processing_time_proposed, 4),
            "per_support": per_support
        }

        # ================== BASELINE MODEL ==================
        baseline_result = {"available": False}
        if b_model is not None:
            t0_base = time.time()
            with torch.no_grad():
                logits = b_model(q_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                b_p_gen, b_p_forg = probs[0].item(), probs[1].item()
            
            processing_time_base = time.time() - t0_base
            b_pred = "GENUINE" if b_p_gen >= b_threshold else "FORGED"
            b_conf = min(abs(b_p_gen - b_threshold) / 0.5 * 100, 100.0)

            baseline_result = {
                "available": True,
                "prediction": b_pred,
                "p_genuine": round(b_p_gen, 4),
                "p_forged": round(b_p_forg, 4),
                "threshold": round(b_threshold, 4),
                "confidence": round(b_conf, 2),
                "processing_time": round(processing_time_base, 4)
            }

        return {
            "proposed": proposed_result,
            "baseline": baseline_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))