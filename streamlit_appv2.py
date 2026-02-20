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

st.title("✍️ Signature Verification — Test Set Protocol (K=1)")
st.markdown("""
This application uses the **exact test set protocol** from the proposed tDCBAM notebook:
- Upload **K=1 support signature** (genuine reference from the test user)
- Upload **1 query signature** to verify (genuine or forged)
- The model computes **P(Forged)** using the learned metric (lower means more genuine)
- Decision: **Forged** if P(Forged) ≥ threshold, else **Genuine**

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


# ==========================================
# IMAGE PROCESSING
# ==========================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocessing pipeline matching the notebook's get_pretraining_transforms.
    Steps: RGB → Grayscale → Otsu Thresholding → Inversion → Resize → RGB → Normalize
    """
    if image is None:
        raise ValueError("No image provided")

    # PIL → numpy (RGB)
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image

    # RGB → gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Otsu binarization
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # invert
    img_inv = cv2.bitwise_not(thresh)

    # resize
    img_resized = cv2.resize(img_inv, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # gray → rgb
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # to float32 [0,1]
    img_rgb = img_rgb.astype("float32") / 255.0

    # numpy HWC → torch CHW
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Normalize using ImageNet statistics (must match training)
    # tensor shape: [1,3,H,W]
    tensor = tensor.squeeze(0)  # [3,H,W]
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)
    tensor = tensor.unsqueeze(0)  # [1,3,H,W]

    return tensor


# ==========================================
# FEATURE EXTRACTION & METRIC COMPUTATION
# ==========================================
def extract_features(image: Image.Image, feature_extractor) -> torch.Tensor:
    """Extract feature embeddings from an image."""
    tensor = preprocess_image(image)
    
    with torch.no_grad():
        features = feature_extractor(tensor)
    
    return features.squeeze(0)  # Remove batch dimension


def compute_similarity(support_feat: torch.Tensor, query_feat: torch.Tensor, 
                      metric_generator) -> float:
    """
    Compute forgery-likelihood between support and query using the metric generator.
    This matches the test protocol: concat(support_feat, query_feat) → MLP → sigmoid (interpreted as P(Forged))
    
    Returns:
        P(Forged): probability that query is forged (different identity than support)
    """
    combined = torch.cat([support_feat, query_feat], dim=0).unsqueeze(0)
    
    with torch.no_grad():
        logit = metric_generator(combined)
        p_forged = torch.sigmoid(logit).item()
    
    return p_forged


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
- **Decision**: P(Forged) ≥ Threshold → FORGED
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Test Protocol")
st.sidebar.markdown("""
This app replicates the **exact test set evaluation**:
1. Extract features from support (K=1 genuine)
2. Extract features from query signature
3. Concatenate: [support_feat, query_feat]
4. Pass through MetricGenerator MLP
5. Apply sigmoid to get P(Forged)
6. Compare to learned threshold
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

if error:
    st.error(f"Failed to load proposed model: {error}")
    st.stop()

st.success(f"✅ Model loaded successfully! Learned threshold: **{threshold:.4f}**")

# Create two columns for instructions
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Support Signature (K=1)")
    st.markdown("Upload **1 genuine** reference signature from the test user")

with col2:
    st.markdown("### Query Signature")
    st.markdown("Upload **1 signature** to verify (genuine or forged)")

# Create file upload columns
st.markdown("---")
col1, col2 = st.columns(2)

support_img = None
query_img = None

with col1:
    st.markdown("**Support (Genuine Reference)**")
    support_upload = st.file_uploader("Upload support signature", type=['jpg', 'png', 'jpeg'],
                                      key='support', label_visibility='collapsed')
    if support_upload:
        support_img = Image.open(support_upload)
        st.image(support_upload, caption="Support", use_container_width=True)

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
    if support_img is None:
        st.error("❌ Please upload a support signature")
    elif query_img is None:
        st.error("❌ Please upload a query signature")
    else:
        with st.spinner("Computing similarity using K=1 episodic protocol..."):
            try:
                # Extract features
                support_features = extract_features(support_img, feature_extractor)  # [1024]
                query_features = extract_features(query_img, feature_extractor)      # [1024]
                
                # Compute P(Genuine) using metric generator
                prob_forged = compute_similarity(support_features, query_features, metric_generator)
                prob_genuine = 1.0 - prob_forged
                
                # Make decision using learned threshold
                # Note: In notebook, labels are 1=genuine, 0=forged
                # The metric outputs high score for genuine pairs
                prediction = "GENUINE" if prob_forged >= threshold else "FORGED"
                
                # Confidence: distance from threshold
                confidence = abs(prob_forged - threshold)
                confidence_pct = min(confidence / 0.5, 1.0) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("## 📋 Verification Results (K=1 Protocol)")
                
                # Main prediction
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown(f"""
                    ### Prediction
                    # {prediction}
                    """)
                
                with result_col2:
                    st.markdown(f"""
                    ### Confidence
                    # {confidence_pct:.2f}%
                    """)
                
                with result_col3:
                    decision_symbol = "✅ GENUINE" if prediction == "GENUINE" else "❌ FORGED"
                    st.markdown(f"""
                    ### Decision
                    # {decision_symbol}
                    """)
                
                # Detailed scores
                st.markdown("---")
                st.markdown("### Probability Scores")
                
                score_col1, score_col2, score_col3 = st.columns(3)
                
                with score_col1:
                    st.metric("P(Genuine)", f"{prob_genuine:.4f}", 
                             delta=f"{(prob_forged - threshold)*100:.2f}% (P(Forged) vs threshold)")
                
                with score_col2:
                    st.metric("P(Forged)", f"{prob_forged:.4f}")
                
                with score_col3:
                    st.metric("Decision Threshold", f"{threshold:.4f}")
                
                # Score visualization
                st.markdown("---")
                st.markdown("### Score Visualization")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Bar chart for probabilities
                categories = ['P(Genuine)', 'P(Forged)']
                probabilities = [prob_genuine, prob_forged]
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
                ax2.barh(['P(Forged)'], [prob_forged], color='#d62728', height=0.5, alpha=0.7)
                ax2.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, 
                           label=f'Threshold ({threshold:.4f})')
                ax2.set_xlim([0, 1])
                ax2.set_xlabel('Probability')
                ax2.set_title('P(Genuine) vs Threshold')
                ax2.legend()
                ax2.grid(axis='x', alpha=0.3)
                
                # Add value label
                ax2.text(prob_genuine + 0.02, 0, f'{prob_genuine:.4f}', 
                        va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Technical details
                st.markdown("---")
                st.markdown("### 🔬 Technical Details")
                
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    st.markdown(f"""
                    **Feature Extraction:**
                    - Support features shape: `{support_features.shape}`
                    - Query features shape: `{query_features.shape}`
                    - Combined input: `{support_features.shape[0] * 2}` dimensions
                    """)
                
                with tech_col2:
                    st.markdown(f"""
                    **Metric Computation:**
                    - Concatenated features: `[support, query]`
                    - MetricGenerator input: `2048` dimensions
                    - Output: Logit → Sigmoid → P(Genuine)
                    - Decision rule: P(Genuine) ≥ {threshold:.4f}
                    """)
                
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
1. **Support Set (K=1)**: One genuine signature from the test user (never seen during training)
2. **Query Set**: One signature to verify (genuine or forged)
3. **Feature Extraction**: Extract 1024-dimensional features using pretrained DenseNet-121 + CBAM backbone
4. **Metric Learning**: Concatenate [support_feat, query_feat] → pass through MetricGenerator MLP
5. **Probability**: Apply sigmoid to get P(Genuine)
6. **Decision**: Compare P(Genuine) to learned threshold (from validation EER)

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
