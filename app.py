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

model_cache = {}
baseline_cache = {}

def load_proposed_models(dataset_name: str, split_ratio: str):
    cache_key = f"{dataset_name}_{split_ratio}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    checkpoint_dir = os.path.join(REPO_ROOT, 'checkpoints', 'proposed_splits', cache_key)
    backbone_path = os.path.join(checkpoint_dir, 'pretrained_backbone.pth')
    metric_path = os.path.join(checkpoint_dir, 'best_meta_model.pth')

    if not os.path.exists(backbone_path) or not os.path.exists(metric_path):
        raise FileNotFoundError(f"Proposed checkpoints not found at {checkpoint_dir}")

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

    checkpoint_dir = os.path.join(REPO_ROOT, 'checkpoints', 'baseline_splits')
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

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv = cv2.bitwise_not(thresh)
    img_resized = cv2.resize(img_inv, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    tensor = torch.from_numpy(img_rgb.astype("float32") / 255.0).permute(2, 0, 1).to(DEVICE)
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    return tensor.unsqueeze(0)

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

        # Preprocess Tensors
        s_tensors = [
            preprocess_image(await support_file_1.read()),
            preprocess_image(await support_file_2.read()),
            preprocess_image(await support_file_3.read())
        ]
        q_tensor = preprocess_image(await query_file.read())

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