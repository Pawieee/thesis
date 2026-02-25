import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sys
import cv2
import matplotlib.pyplot as plt
import time
import io


# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from models.feature_extractor import DenseNetFeatureExtractor
from models.meta_learner import MetricGenerator


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Signature Verification - Test Protocol",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✍️ Signature Verification — Test Set Protocol (K=1, 3 References)")
st.markdown("""
This application uses the **exact test set protocol** from the proposed tDCBAM notebook:
- Upload **3 support signatures** (genuine references from the test user)
- Upload **1 query signature** to verify (genuine or forged)
- The model computes **P(Genuine)** using the learned metric (use the threshold rule below)
- The proposed model performs **3 K=1 comparisons** and uses **majority vote** for the final decision
- Decision: **Genuine** if P(Genuine) ≥ threshold, else **Forged**

**Note:** This matches the K-shot episodic evaluation exactly as during training.
""")

# ==========================================
# DEVICE & MODEL CONFIGURATION
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configurations
DATASETS = {
    'BHSig-Bengali': 'bhsig_bengali',
    'BHSig-Hindi': 'bhsig_hindi',
    'CEDAR': 'cedar',
    'Combined (All Datasets)': 'combined'
}

SPLIT_RATIOS = {
    '60% Train / 20% Val / 20% Test': '60_20_20',
    '65% Train / 18% Val / 18% Test': '65_18_18',
    '70% Train / 15% Val / 15% Test': '70_15_15',
}

IMAGE_SIZE = 224
FEATURE_DIM = 1024
EMBEDDING_DIM = 2048
BASELINE_NUM_CLASSES = 2

# Default threshold (will be replaced by learned threshold from checkpoint)
DEFAULT_THRESHOLD = 0.5


# ==========================================
# MODEL LOADING (with caching)
# ==========================================
@st.cache_resource
def load_proposed_models(dataset_name, split_ratio):
    """Load feature extractor and metric generator models with learned threshold."""
    try:
        # Set up checkpoint path
        checkpoint_dir = os.path.join(REPO_ROOT, 'checkpoints', 'proposed_splits', 
                                      f'{dataset_name}_{split_ratio}')
        
        backbone_path = os.path.join(checkpoint_dir, 'pretrained_backbone.pth')
        metric_path = os.path.join(checkpoint_dir, 'best_meta_model.pth')
        
        if not os.path.exists(backbone_path) or not os.path.exists(metric_path):
            return None, None, DEFAULT_THRESHOLD, f"Checkpoints not found at {checkpoint_dir}"
        
        # Load feature extractor
        feature_extractor = DenseNetFeatureExtractor(
            backbone_name='densenet121',
            output_dim=FEATURE_DIM,
            pretrained=False,
            baseline=False
        )
        feature_extractor.load_state_dict(torch.load(backbone_path, map_location=DEVICE, weights_only=True))
        feature_extractor = feature_extractor.to(DEVICE)
        feature_extractor.eval()
        
        # Load metric generator (MLP)
        metric_generator = MetricGenerator(
            embedding_dim=EMBEDDING_DIM
        )
        
        # Load the full checkpoint
        full_checkpoint = torch.load(metric_path, map_location=DEVICE, weights_only=False)
        
        # Extract threshold from metrics if available
        learned_threshold = DEFAULT_THRESHOLD
        if isinstance(full_checkpoint, dict):
            if 'metric_generator' in full_checkpoint:
                metric_generator.load_state_dict(full_checkpoint['metric_generator'])
            else:
                metric_generator.load_state_dict(full_checkpoint)
            
            # Try to get the learned threshold from saved metrics
            if 'metrics' in full_checkpoint and 'threshold' in full_checkpoint['metrics']:
                learned_threshold = float(full_checkpoint['metrics']['threshold'])
        
        metric_generator = metric_generator.to(DEVICE)
        metric_generator.eval()
        
        return feature_extractor, metric_generator, learned_threshold, None
    
    except Exception as e:
        return None, None, DEFAULT_THRESHOLD, str(e)


@st.cache_resource
def load_baseline_model(dataset_name, split_ratio):
    """Load the baseline DenseNet classifier with its learned threshold."""
    try:
        checkpoint_dir = os.path.join(REPO_ROOT, 'checkpoints', 'baseline_splits')
        checkpoint_path = os.path.join(checkpoint_dir, f"best_{dataset_name}_{split_ratio}.pth")

        if not os.path.exists(checkpoint_path):
            return None, DEFAULT_THRESHOLD, f"Baseline checkpoint not found: {checkpoint_path}"

        baseline_model = DenseNetFeatureExtractor(
            backbone_name='densenet121',
            output_dim=BASELINE_NUM_CLASSES,
            pretrained=False,
            baseline=True
        )

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        baseline_model.load_state_dict(state)
        baseline_model = baseline_model.to(DEVICE)
        baseline_model.eval()

        learned_threshold = DEFAULT_THRESHOLD
        if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
            if 'threshold' in checkpoint['metrics']:
                learned_threshold = float(checkpoint['metrics']['threshold'])

        return baseline_model, learned_threshold, None

    except Exception as e:
        return None, DEFAULT_THRESHOLD, str(e)


# ==========================================
# IMAGE PROCESSING
# ==========================================
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


# ==========================================
# FEATURE EXTRACTION & METRIC COMPUTATION
# ==========================================
def extract_features(image: Image.Image, feature_extractor) -> tuple[torch.Tensor, dict]:
    """Extract feature embeddings from an image and return preprocessing steps."""
    tensor, steps = preprocess_image_with_steps(image)
    
    with torch.no_grad():
        features = feature_extractor(tensor)
    
    return features.squeeze(0), steps  # Remove batch dimension


def compute_similarity(support_feat: torch.Tensor, query_feat: torch.Tensor, 
                      metric_generator) -> float:
    """
    Compute forgery-likelihood between support and query using the metric generator.
    This matches the test protocol: concat(support_feat, query_feat) → MLP → sigmoid (interpreted as P(Genuine))
    
    Returns:
        P(Genuine): probability that query is genuine (same identity as support)
    """
    combined = torch.cat([support_feat, query_feat], dim=0).unsqueeze(0)
    
    with torch.no_grad():
        logit = metric_generator(combined)
        p_genuine = torch.sigmoid(logit).item()
    
    return p_genuine


def compute_baseline_probs(image: Image.Image, baseline_model) -> tuple[float, float]:
    """Return (P(Genuine), P(Forged)) from the baseline classifier."""
    tensor, _ = preprocess_image_with_steps(image)

    with torch.no_grad():
        logits = baseline_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    p_genuine = probs[0].item()
    p_forged = probs[1].item()
    return p_genuine, p_forged


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.markdown("### ⚙️ Configuration")

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=DATASETS.keys(),
    help="Choose the signature dataset for model selection"
)

selected_split = st.sidebar.selectbox(
    "Select Train/Test Split Ratio",
    options=SPLIT_RATIOS.keys(),
    help="Choose the data split ratio used for training"
)

# Clear cache button
if st.sidebar.button("🔄 Reload Model", help="Clear cache and reload model"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.markdown(f"""
- **Protocol**: K=1 Episodic (Test Set)
- **Backbone**: DenseNet-121 + CBAM
- **Feature Dimension**: 1024
- **Metric Learner**: MLP (2048 → 1)
- **Loss**: BCEWithLogitsLoss
- **Decision**: P(Genuine) ≥ Threshold → GENUINE
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Test Protocol")
st.sidebar.markdown("""
This app replicates the **exact test set evaluation**:
1. Extract features from 3 supports (K=1 each)
2. Extract features from query signature
3. For each support: concat [support_feat, query_feat]
4. Pass through MetricGenerator MLP
5. Apply sigmoid to get P(Genuine)
6. Majority vote across 3 comparisons for final decision
""")

# ==========================================
# MAIN APPLICATION
# ==========================================
st.markdown("---")

# Load models
dataset_name = DATASETS[selected_dataset]
split_ratio = SPLIT_RATIOS[selected_split]

with st.spinner(f"Loading models for {selected_dataset} ({selected_split})..."):
    feature_extractor, metric_generator, threshold, error = load_proposed_models(dataset_name, split_ratio)

baseline_model = None
baseline_threshold = DEFAULT_THRESHOLD
baseline_error = None
if dataset_name == 'combined':
    baseline_error = "Baseline checkpoint not available for the combined dataset"
else:
    with st.spinner(f"Loading baseline model for {selected_dataset} ({selected_split})..."):
        baseline_model, baseline_threshold, baseline_error = load_baseline_model(dataset_name, split_ratio)

if error:
    st.error(f"Failed to load proposed model: {error}")
    st.stop()

st.success(f"✅ Model loaded successfully! Learned threshold: **{threshold:.4f}**")

if baseline_error:
    st.info(f"ℹ️ Baseline model not loaded: {baseline_error}")
else:
    st.success(f"✅ Baseline loaded! Learned threshold: **{baseline_threshold:.4f}**")

# Create two columns for instructions
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Support Signatures (3x K=1)")
    st.markdown("Upload **3 genuine** reference signatures from the test user")

with col2:
    st.markdown("### Query Signature")
    st.markdown("Upload **1 signature** to verify (genuine or forged)")

# Create file upload columns
st.markdown("---")
col1, col2 = st.columns(2)

support_imgs = [None, None, None]
query_img = None

with col1:
    st.markdown("**Support 1 (Genuine Reference)**")
    support_upload_1 = st.file_uploader("Upload support signature 1", type=['jpg', 'png', 'jpeg'],
                                        key='support1', label_visibility='collapsed')
    if support_upload_1:
        support_imgs[0] = Image.open(support_upload_1)
        st.image(support_upload_1, caption="Support 1", use_container_width=True)

    st.markdown("**Support 2 (Genuine Reference)**")
    support_upload_2 = st.file_uploader("Upload support signature 2", type=['jpg', 'png', 'jpeg'],
                                        key='support2', label_visibility='collapsed')
    if support_upload_2:
        support_imgs[1] = Image.open(support_upload_2)
        st.image(support_upload_2, caption="Support 2", use_container_width=True)

    st.markdown("**Support 3 (Genuine Reference)**")
    support_upload_3 = st.file_uploader("Upload support signature 3", type=['jpg', 'png', 'jpeg'],
                                        key='support3', label_visibility='collapsed')
    if support_upload_3:
        support_imgs[2] = Image.open(support_upload_3)
        st.image(support_upload_3, caption="Support 3", use_container_width=True)

with col2:
    st.markdown("**Query (Test Signature)**")
    query_upload = st.file_uploader("Upload query signature", type=['jpg', 'png', 'jpeg'],
                                    key='query', label_visibility='collapsed')
    if query_upload:
        query_img = Image.open(query_upload)
        st.image(query_upload, caption="Query", use_container_width=True)

# ==========================================
# PREDICTION
# ==========================================
st.markdown("---")

if st.button("🔍 Verify Signature (K=1 Protocol)", type="primary", use_container_width=True):
    
    # Validation
    if any(img is None for img in support_imgs):
        missing = [str(i + 1) for i, img in enumerate(support_imgs) if img is None]
        st.error(f"❌ Please upload all support signatures (missing: {', '.join(missing)})")
    elif query_img is None:
        st.error("❌ Please upload a query signature")
    else:
        with st.spinner("Computing similarity using 3 K=1 comparisons..."):
            try:
                # Start timing for proposed model
                start_time_proposed = time.time()
                
                # Extract features AND keep steps for all support images
                support_features = []
                support_steps_list = []
                for img in support_imgs:
                    feat, steps = extract_features(img, feature_extractor)
                    support_features.append(feat)
                    support_steps_list.append(steps)

                # Extract features AND keep steps for the query image
                query_features, query_steps = extract_features(query_img, feature_extractor)  # [1024]

                # Compute P(Genuine) for each support and majority vote
                per_support = []
                for idx, support_feat in enumerate(support_features, start=1):
                    p_genuine = compute_similarity(support_feat, query_features, metric_generator)
                    p_forged = 1.0 - p_genuine
                    pred = "GENUINE" if p_genuine >= threshold else "FORGED"
                    per_support.append({
                        "support": f"Support {idx}",
                        "p_genuine": p_genuine,
                        "p_forged": p_forged,
                        "prediction": pred
                    })

                genuine_votes = sum(1 for item in per_support if item["prediction"] == "GENUINE")
                forged_votes = 3 - genuine_votes
                prediction = "GENUINE" if genuine_votes >= 2 else "FORGED"
                
                # End timing for proposed model
                end_time_proposed = time.time()
                processing_time_proposed = end_time_proposed - start_time_proposed

                avg_prob_genuine = float(np.mean([item["p_genuine"] for item in per_support]))
                avg_prob_forged = 1.0 - avg_prob_genuine

                vote_confidence_pct = (max(genuine_votes, forged_votes) / 3.0) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("## 📋 Verification Results (3x K=1 Protocol)")
                
                # Main prediction
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.markdown(f"""
                    ### Prediction
                    # {prediction}
                    """)
                
                with result_col2:
                    st.markdown(f"""
                    ### Vote Confidence
                    # {vote_confidence_pct:.2f}%
                    """)
                
                with result_col3:
                    decision_symbol = "✅ GENUINE" if prediction == "GENUINE" else "❌ FORGED"
                    st.markdown(f"""
                    ### Decision
                    # {decision_symbol}
                    """)
                
                with result_col4:
                    st.markdown(f"""
                    ### Processing Time
                    # {processing_time_proposed:.3f}s
                    """)
                
                # Detailed scores
                st.markdown("---")
                st.markdown("### Probability Scores")
                
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                
                with score_col1:
                    st.metric("P(Genuine)", f"{avg_prob_genuine:.4f}", 
                             delta=f"{(avg_prob_genuine - threshold)*100:.2f}% (P(Genuine) vs threshold)")
                
                with score_col2:
                    st.metric("P(Forged)", f"{avg_prob_forged:.4f}")
                
                with score_col3:
                    st.metric("Decision Threshold", f"{threshold:.4f}")
                
                with score_col4:
                    st.metric("Processing Time", f"{processing_time_proposed:.3f}s", 
                             help="Total time for feature extraction and 3 K=1 comparisons")
                
                # Score visualization
                st.markdown("---")
                st.markdown("### Score Visualization")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Bar chart for probabilities
                categories = ['P(Genuine)', 'P(Forged)']
                probabilities = [avg_prob_genuine, avg_prob_forged]
                colors = ['#2ca02c', '#d62728']
                
                ax1.bar(categories, probabilities, color=colors, alpha=0.7, edgecolor='black')
                ax1.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, 
                           label=f'Threshold ({threshold:.4f})')
                ax1.set_ylabel('Probability')
                ax1.set_title('Probability Distribution')
                ax1.set_ylim([0, 1])
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, (cat, prob) in enumerate(zip(categories, probabilities)):
                    ax1.text(i, prob + 0.02, f'{prob:.4f}', ha='center', fontweight='bold')
                
                # Gauge chart showing P(Genuine) vs threshold
                ax2.barh(['P(Genuine)'], [avg_prob_genuine], color='#2ca02c', height=0.5, alpha=0.7)
                ax2.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, 
                           label=f'Threshold ({threshold:.4f})')
                ax2.set_xlim([0, 1])
                ax2.set_xlabel('Probability')
                ax2.set_title('P(Genuine) vs Threshold')
                ax2.legend()
                ax2.grid(axis='x', alpha=0.3)
                
                # Add value label
                ax2.text(avg_prob_genuine + 0.02, 0, f'{avg_prob_genuine:.4f}', 
                        va='center', fontweight='bold')

                # Per-support breakdown
                st.markdown("---")
                st.markdown("### Per-Support Results")
                per_support_rows = [
                    {
                        "Support": item["support"],
                        "Prediction": item["prediction"],
                        "P(Genuine)": f"{item['p_genuine']:.4f}",
                        "P(Forged)": f"{item['p_forged']:.4f}"
                    }
                    for item in per_support
                ]
                st.table(per_support_rows)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Technical details
                st.markdown("---")
                st.markdown("### 🔬 Technical Details")
                
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    st.markdown(f"""
                    **Feature Extraction:**
                    - Support features shape: `{support_features[0].shape}` (x3)
                    - Query features shape: `{query_features.shape}`
                    - Combined input: `{support_features[0].shape[0] * 2}` dimensions
                    """)
                
                with tech_col2:
                    st.markdown(f"""
                    **Metric Computation:**
                    - Concatenated features: `[support, query]`
                    - MetricGenerator input: `2048` dimensions
                    - Output: Logit → Sigmoid → P(Genuine)
                    - Decision rule: P(Genuine) ≥ {threshold:.4f} → GENUINE
                    """)

                # Baseline prediction on query image
                if baseline_model is not None:
                    st.markdown("---")
                    st.markdown("## 🧭 Baseline Prediction (Single Image)")
                    st.markdown("This uses the baseline DenseNet classifier on the **query** image only.")

                    # Start timing for baseline model
                    start_time_baseline = time.time()
                    base_p_genuine, base_p_forged = compute_baseline_probs(query_img, baseline_model)
                    base_prediction = "GENUINE" if base_p_genuine >= baseline_threshold else "FORGED"
                    end_time_baseline = time.time()
                    processing_time_baseline = end_time_baseline - start_time_baseline

                    base_confidence = abs(base_p_genuine - baseline_threshold)
                    base_confidence_pct = min(base_confidence / 0.5, 1.0) * 100

                    base_col1, base_col2, base_col3, base_col4 = st.columns(4)

                    with base_col1:
                        st.markdown(f"""
                        ### Prediction
                        # {base_prediction}
                        """)

                    with base_col2:
                        st.markdown(f"""
                        ### Confidence
                        # {base_confidence_pct:.2f}%
                        """)

                    with base_col3:
                        base_symbol = "✅ GENUINE" if base_prediction == "GENUINE" else "❌ FORGED"
                        st.markdown(f"""
                        ### Decision
                        # {base_symbol}
                        """)
                    
                    with base_col4:
                        st.markdown(f"""
                        ### Processing Time
                        # {processing_time_baseline:.3f}s
                        """)

                    st.markdown("---")
                    st.markdown("### Baseline Probability Scores")

                    base_score_col1, base_score_col2, base_score_col3, base_score_col4 = st.columns(4)

                    with base_score_col1:
                        st.metric(
                            "P(Genuine)",
                            f"{base_p_genuine:.4f}",
                            delta=f"{(base_p_genuine - baseline_threshold) * 100:.2f}% (P(Genuine) vs threshold)"
                        )

                    with base_score_col2:
                        st.metric("P(Forged)", f"{base_p_forged:.4f}")

                    with base_score_col3:
                        st.metric("Decision Threshold", f"{baseline_threshold:.4f}")
                    
                    with base_score_col4:
                        st.metric("Processing Time", f"{processing_time_baseline:.3f}s",
                                 help="Total time for single image classification")

                    st.markdown("---")
                    st.markdown("### Baseline Score Visualization")

                    fig_base, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 4))

                    base_categories = ['P(Genuine)', 'P(Forged)']
                    base_probabilities = [base_p_genuine, base_p_forged]
                    base_colors = ['#2ca02c', '#d62728']

                    bx1.bar(base_categories, base_probabilities, color=base_colors, alpha=0.7, edgecolor='black')
                    bx1.axhline(y=baseline_threshold, color='blue', linestyle='--', linewidth=2,
                                label=f'Threshold ({baseline_threshold:.4f})')
                    bx1.set_ylabel('Probability')
                    bx1.set_title('Baseline Probability Distribution')
                    bx1.set_ylim([0, 1])
                    bx1.legend()
                    bx1.grid(axis='y', alpha=0.3)

                    for i, (cat, prob) in enumerate(zip(base_categories, base_probabilities)):
                        bx1.text(i, prob + 0.02, f'{prob:.4f}', ha='center', fontweight='bold')

                    bx2.barh(['P(Genuine)'], [base_p_genuine], color='#2ca02c', height=0.5, alpha=0.7)
                    bx2.axvline(x=baseline_threshold, color='blue', linestyle='--', linewidth=2,
                                label=f'Threshold ({baseline_threshold:.4f})')
                    bx2.set_xlim([0, 1])
                    bx2.set_xlabel('Probability')
                    bx2.set_title('Baseline P(Genuine) vs Threshold')
                    bx2.legend()
                    bx2.grid(axis='x', alpha=0.3)

                    bx2.text(base_p_genuine + 0.02, 0, f'{base_p_genuine:.4f}',
                             va='center', fontweight='bold')

                    plt.tight_layout()
                    st.pyplot(fig_base)

                    st.markdown("---")
                    st.markdown("## 🆚 Proposed vs Baseline Comparison")

                    comparison_rows = [
                        {
                            "Model": "Proposed (K=1 metric)",
                            "Prediction": prediction,
                            "P(Forged)": f"{avg_prob_forged:.4f}",
                            "P(Genuine)": f"{avg_prob_genuine:.4f}",
                            "Threshold": f"{threshold:.4f}",
                            "Confidence": f"{vote_confidence_pct:.2f}%",
                            "Processing Time": f"{processing_time_proposed:.3f}s"
                        },
                        {
                            "Model": "Baseline (single image)",
                            "Prediction": base_prediction,
                            "P(Genuine)": f"{base_p_genuine:.4f}",
                            "P(Forged)": f"{base_p_forged:.4f}",
                            "Threshold": f"{baseline_threshold:.4f}",
                            "Confidence": f"{base_confidence_pct:.2f}%",
                            "Processing Time": f"{processing_time_baseline:.3f}s"
                        }
                    ]

                    st.table(comparison_rows)
                    
                    # Add speedup metric
                    speedup = processing_time_proposed / processing_time_baseline if processing_time_baseline > 0 else 0
                    st.markdown(f"**Speed Comparison:** Baseline is **{speedup:.2f}x faster** than Proposed (Proposed requires 4 images: 3 supports + 1 query)")

                # ==========================================
                # PREPROCESSING PIPELINE VISUALIZATION
                # ==========================================
                st.markdown("---")
                st.markdown("## ⚙️ Preprocessing Pipeline Visualization")
                st.markdown("Visual breakdown of the steps applied to isolate the signature geometry, eliminate scaling distortions, and prepare the strokes for the CBAM layers.")
                
                # Function to render the 6 steps uniformly across tabs
                def render_preprocessing_steps(steps_dict):
                    vis_cols = st.columns(3)
                    step_items = list(steps_dict.items())
                    for i, (step_name, img_array) in enumerate(step_items):
                        with vis_cols[i % 3]:
                            st.markdown(f"**{step_name}**")
                            st.image(img_array, use_container_width=True, clamp=True)

                # Create tabs for all 4 images
                tab1, tab2, tab3, tab4 = st.tabs(["Support 1", "Support 2", "Support 3", "Query Signature"])

                with tab1:
                    render_preprocessing_steps(support_steps_list[0])
                with tab2:
                    render_preprocessing_steps(support_steps_list[1])
                with tab3:
                    render_preprocessing_steps(support_steps_list[2])
                with tab4:
                    render_preprocessing_steps(query_steps)

            except Exception as e:
                st.error(f"❌ Error during verification: {str(e)}")
                st.write("Please ensure images are valid signature images in JPG or PNG format")
                import traceback
                st.code(traceback.format_exc())

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
### ℹ️ About This Application

This signature verification system uses the **exact test set protocol** from the proposed tDCBAM notebook:

**K=1 Episodic Evaluation Protocol:**
1. **Support Set (3x K=1)**: Three genuine signatures from the test user (never seen during training)
2. **Query Set**: One signature to verify (genuine or forged)
3. **Feature Extraction**: Extract 1024-dimensional features using pretrained DenseNet-121 + CBAM backbone
4. **Metric Learning**: For each support, concat [support_feat, query_feat] → pass through MetricGenerator MLP
5. **Probability**: Apply sigmoid to get P(Genuine)
6. **Decision**: Majority vote across 3 comparisons; each comparison uses P(Genuine) ≥ threshold → Genuine

**Model Training:**
- **Step 1**: Pretrain backbone with Triplet Loss on train users
- **Step 2**: Meta-train MetricGenerator on train users (episodic protocol)
- **Validation**: Select best model based on validation EER
- **Test**: Evaluate on held-out test users (K=1 protocol)

**Key Differences from Previous App:**
- This app uses **K=1** (single support) instead of averaging 3 references
- Uses **learned threshold** from validation EER (not fixed 0.5)
- Matches the **exact test evaluation** from the notebook
- Output is **P(Genuine)** (not similarity score)

**Dataset Support:**
- BHSig-Bengali, BHSig-Hindi, CEDAR (evaluated independently)
- Combined model (trained on all datasets with 70:15:15 split)

**Decision Boundary:**
- The threshold is learned during validation to minimize EER (Equal Error Rate)
- This ensures optimal FAR/FRR tradeoff matching the test evaluation
""")

st.markdown("---")
st.markdown("*Test Protocol Verification Tool — Matches Proposed Notebook Evaluation*")