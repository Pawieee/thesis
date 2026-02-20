import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sys
import cv2


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
    page_title="Signature Verification",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✍️ Offline Signature Verification System")
st.markdown("""
This application uses a **Triplet Loss + MLP-based** deep learning model to verify the authenticity 
of handwritten signatures. Upload **3 genuine reference signatures** and **1 test signature**. The app 
computes a similarity score for each reference and uses the **average** for the final decision.
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
THRESHOLD = 0.54
BASELINE_THRESHOLD = 0.5


# ==========================================
# MODEL LOADING (with caching)
# ==========================================
@st.cache_resource
def load_proposed_models(dataset_name, split_ratio):
    """Load feature extractor and metric generator models (proposed)."""
    try:
        # Set up checkpoint path
        checkpoint_dir = os.path.join(REPO_ROOT, 'checkpoints', 'proposed_splits', 
                                      f'{dataset_name}_{split_ratio}')
        
        backbone_path = os.path.join(checkpoint_dir, 'pretrained_backbone.pth')
        metric_path = os.path.join(checkpoint_dir, 'best_meta_model.pth')
        
        if not os.path.exists(backbone_path) or not os.path.exists(metric_path):
            return None, None, f"Checkpoints not found at {checkpoint_dir}"
        
        # Load feature extractor
        feature_extractor = DenseNetFeatureExtractor(
            backbone_name='densenet121',
            output_dim=FEATURE_DIM,
            pretrained=False,
            baseline=False
        )
        feature_extractor.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
        feature_extractor = feature_extractor.to(DEVICE)
        feature_extractor.eval()
        
        # Load metric generator (MLP)
        metric_generator = MetricGenerator(
            embedding_dim=EMBEDDING_DIM
        )
        
        # Load the full checkpoint
        full_checkpoint = torch.load(metric_path, map_location=DEVICE)
        
        # Try different loading strategies based on checkpoint structure
        if isinstance(full_checkpoint, dict):
            # Check if this is a wrapper model or direct state dict
            if 'metric_generator' in full_checkpoint:
                # It's a wrapper model, extract only the metric_generator part
                metric_state = full_checkpoint['metric_generator']
                metric_generator.load_state_dict(metric_state)
            elif 'relation_module' in str(full_checkpoint.keys()):
                # Direct state dict for MetricGenerator
                metric_generator.load_state_dict(full_checkpoint)
            else:
                # Try to extract metric_generator component from wrapper
                # Look for keys starting with "metric_generator."
                metric_keys = {k.replace('metric_generator.', ''): v 
                             for k, v in full_checkpoint.items() 
                             if k.startswith('metric_generator.')}
                
                if metric_keys:
                    metric_generator.load_state_dict(metric_keys)
                else:
                    # If still not found, just try loading directly
                    metric_generator.load_state_dict(full_checkpoint)
        
        metric_generator = metric_generator.to(DEVICE)
        metric_generator.eval()
        
        return feature_extractor, metric_generator, None
    
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_baseline_model(dataset_name, split_ratio):
    """Load baseline DenseNet classifier model."""
    try:
        checkpoint_name = f"best_{dataset_name}_{split_ratio}.pth"
        checkpoint_path = os.path.join(REPO_ROOT, 'checkpoints', 'baseline_splits', checkpoint_name)

        if not os.path.exists(checkpoint_path):
            return None, f"Checkpoint not found at {checkpoint_path}"

        baseline_model = DenseNetFeatureExtractor(
            backbone_name='densenet121',
            output_dim=2,
            pretrained=True,
            baseline=True
        )

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        try:
            baseline_model.load_state_dict(state, strict=True)
        except RuntimeError:
            model_state = baseline_model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_state and model_state[k].shape == v.shape}
            model_state.update(filtered)
            baseline_model.load_state_dict(model_state)

        baseline_model = baseline_model.to(DEVICE)
        baseline_model.eval()

        return baseline_model, None

    except Exception as e:
        return None, str(e)


# ==========================================
# IMAGE PROCESSING
# ==========================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Use RGB conversion to match .convert('RGB') in your TripletDataset
    if image is None:
            raise ValueError("No image provided")

    # PIL -> numpy (RGB)
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image

    # RGB -> gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Otsu binarization
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # invert
    img_inv = cv2.bitwise_not(thresh)

    # resize
    img_resized = cv2.resize(img_inv, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # gray -> rgb
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # to float32 [0,1]
    img_rgb = img_rgb.astype("float32") / 255.0

    # numpy HWC -> torch CHW
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    return tensor


# ==========================================
# FEATURE EXTRACTION & SIMILARITY SCORING
# ==========================================
def extract_features(image: Image.Image, feature_extractor) -> torch.Tensor:
    """Extract feature embeddings from an image."""
    tensor = preprocess_image(image)
    
    with torch.no_grad():
        features = feature_extractor(tensor)
    
    return features.squeeze(0)  # Remove batch dimension


def predict_baseline_probability(image: Image.Image, baseline_model) -> float:
    """Predict genuine probability using the baseline classifier."""
    tensor = preprocess_image(image)

    with torch.no_grad():
        logits = baseline_model(tensor)
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()



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
- **Backbone**: DenseNet-121
- **Feature Dimension**: 1024
- **Attention**: CBAM (Channel & Spatial)
- **Metric Learner**: MLP with LayerNorm
- **Loss Function**: Triplet Loss + Relation Loss
- **Embedding Dimension**: 2048
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📘 Baseline Model")
st.sidebar.markdown("""
- **Backbone**: DenseNet-121 (no CBAM)
- **Head**: 2-class classifier
- **Score**: Softmax probability (genuine)
""")

# ==========================================
# MAIN APPLICATION
# ==========================================
st.markdown("---")

# Load models
dataset_name = DATASETS[selected_dataset]
split_ratio = SPLIT_RATIOS[selected_split]

with st.spinner(f"Loading models for {selected_dataset} ({selected_split})..."):
    feature_extractor, metric_generator, error = load_proposed_models(dataset_name, split_ratio)
    baseline_model, baseline_error = load_baseline_model(dataset_name, split_ratio)

if error:
    st.error(f"Failed to load proposed model: {error}")
    st.stop()

if baseline_error:
    st.warning(f"Baseline model unavailable: {baseline_error}")

st.success("Model loaded successfully!")

# Create two columns for instructions
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Reference Signatures")
    st.markdown("Upload 3 **genuine** reference signatures from the same person")

with col2:
    st.markdown("### Test Signature")
    st.markdown("Upload 1 signature to verify (genuine or forged)")

# Create file upload columns
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

ref_images = []
with col1:
    st.markdown("**Reference 1**")
    ref1 = st.file_uploader("Upload reference signature 1", type=['jpg', 'png', 'jpeg'],
                            key='ref1', label_visibility='collapsed')
    if ref1:
        ref_images.append(Image.open(ref1))
        st.image(ref1, width='content')

with col2:
    st.markdown("**Reference 2**")
    ref2 = st.file_uploader("Upload reference signature 2", type=['jpg', 'png', 'jpeg'],
                            key='ref2', label_visibility='collapsed')
    if ref2:
        ref_images.append(Image.open(ref2))
        st.image(ref2, width='content')

with col3:
    st.markdown("**Reference 3**")
    ref3 = st.file_uploader("Upload reference signature 3", type=['jpg', 'png', 'jpeg'],
                            key='ref3', label_visibility='collapsed')
    if ref3:
        ref_images.append(Image.open(ref3))
        st.image(ref3, width='content')

with col4:
    st.markdown("**Test Signature**")
    test_img = st.file_uploader("Upload test signature", type=['jpg', 'png', 'jpeg'],
                               key='test', label_visibility='collapsed')
    if test_img:
        st.image(test_img, width='content')

# ==========================================
# PREDICTION
# ==========================================
st.markdown("---")

if st.button("Verify Signature", type="primary", width='stretch'):
    
    # Validation
    if len(ref_images) < 3:
        st.error("Please upload all 3 reference signatures")
    elif test_img is None:
        st.error("Please upload a test signature")
    else:
        with st.spinner("Processing signatures and computing individual scores..."):
            try:
                # 1. Extract features for Test Image
                test_image = Image.open(test_img)
                test_features = extract_features(test_image, feature_extractor) # [1024]
                
                # 2. Extract features for each Reference Image
                reference_features_tensors = []
                for ref_img in ref_images:
                    feat = extract_features(ref_img, feature_extractor)
                    reference_features_tensors.append(feat)
                
                # 3. Compute per-reference forged probabilities (K=1 trials)
                logits = []
                prob_genuine = []
                prob_forged = []

                for ref_feat in reference_features_tensors:
                    combined = torch.cat([ref_feat, test_features], dim=0).unsqueeze(0)

                    with torch.no_grad():
                        logit = metric_generator(combined)
                        logits.append(logit.item())

                        p_genuine = torch.sigmoid(logit).item()
                        p_forged = 1.0 - p_genuine
                        prob_genuine.append(p_genuine)
                        prob_forged.append(p_forged)

                # Per-reference decisions using the same threshold as test phase
                ref_decisions = ["FORGED" if p >= THRESHOLD else "GENUINE" for p in prob_forged]
                forged_votes = sum(1 for d in ref_decisions if d == "FORGED")
                genuine_votes = len(ref_decisions) - forged_votes

                # Majority vote (2 out of 3)
                prediction = "FORGED" if forged_votes >= 2 else "GENUINE"

                # Optional mean forged score for display
                mean_forged_score = float(np.mean(prob_forged))

                # Confidence: distance of mean forged score from threshold
                confidence = abs(mean_forged_score - THRESHOLD)
                confidence = min(confidence / 0.5, 1.0) * 100

                
                baseline_score = None
                if baseline_model is not None:
                    baseline_score = predict_baseline_probability(test_image, baseline_model)
                    baseline_prediction = "GENUINE" if baseline_score >= BASELINE_THRESHOLD else "FORGED"
                    baseline_confidence = abs(baseline_score - BASELINE_THRESHOLD)
                    baseline_confidence = min(baseline_confidence / 0.5, 1.0) * 100

                # Display results
                st.markdown("---")
                st.markdown("## 📋 Verification Results")
                
                # Main prediction with large display
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown(f"""
                    ### Prediction
                    # {prediction}
                    """)
                
                with result_col2:
                    st.markdown(f"""
                    ### Confidence
                    # {confidence:.2f}%
                    """)

                if baseline_score is not None:
                    st.markdown("---")
                    st.markdown("### Baseline Model Result")
                    st.caption("Baseline score is computed from the test signature only.")
                    base_col1, base_col2 = st.columns(2)

                    with base_col1:
                        st.markdown(f"""
                        ### Prediction
                        # {baseline_prediction}
                        """)

                    with base_col2:
                        st.markdown(f"""
                        ### Confidence
                        # {baseline_confidence:.2f}%
                        """)
                
                # Detailed scores
                st.markdown("### Per-Reference Forged Scores (K=1)")
                score_data = {
                    'Reference': ['Reference 1', 'Reference 2', 'Reference 3'],
                    'P(Forged)': [f"{s:.4f}" for s in prob_forged],
                    'Decision': ref_decisions,
                    'P(Forged) %': [f"{s*100:.2f}%" for s in prob_forged]
                }
                
                import pandas as pd
                df = pd.DataFrame(score_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Vote summary and mean score
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean P(Forged)", f"{mean_forged_score:.4f}", 
                             delta=f"{(mean_forged_score - THRESHOLD)*100:.2f}%")
                
                with col2:
                    st.metric("Decision Threshold", f"{THRESHOLD:.4f}")
                
                with col3:
                    vote_summary = f"{forged_votes} Forged / {genuine_votes} Genuine"
                    decision = "❌ NO MATCH" if prediction == "FORGED" else "✅ MATCH"
                    st.metric("Final Decision", decision, delta=vote_summary)

                if baseline_score is not None:
                    st.markdown("---")
                    base_metric_col1, base_metric_col2, base_metric_col3 = st.columns(3)

                    with base_metric_col1:
                        st.metric("Baseline Genuine Prob", f"{baseline_score:.4f}",
                                  delta=f"{(baseline_score - BASELINE_THRESHOLD)*100:.2f}%")

                    with base_metric_col2:
                        st.metric("Baseline Threshold", f"{BASELINE_THRESHOLD:.4f}")

                    with base_metric_col3:
                        base_decision = "✅ MATCH" if baseline_score >= BASELINE_THRESHOLD else "❌ NO MATCH"
                        st.metric("Baseline Decision", base_decision)
                
                # Score visualization
                st.markdown("---")
                st.markdown("### Score Distribution")
                
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Bar chart for per-reference forged scores
                ax1.bar(range(1, 4), prob_forged, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax1.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD:.4f})')
                ax1.set_xlabel('Reference Signature')
                ax1.set_ylabel('P(Forged)')
                ax1.set_title('Per-Reference Forged Scores')
                ax1.set_ylim([0, 1])
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # Gauge chart showing mean forged score
                ax2.barh(['Mean\nP(Forged)'], [mean_forged_score], color='#1f77b4', height=0.4)
                ax2.axvline(x=THRESHOLD, color='r', linestyle='--', linewidth=2, label='Threshold')
                ax2.set_xlim([0, 1])
                ax2.set_xlabel('P(Forged)')
                ax2.set_title('Mean Forged Score')
                ax2.legend()
                ax2.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error during verification: {str(e)}")
                st.write("Please ensure images are valid signature images in JPG or PNG format")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
### ℹ️ About This Application
This signature verification system uses a state-of-the-art deep learning approach combining:
- **Triplet Loss** for feature learning during pretraining
- **Meta-Learning (Episodic)** for few-shot adaptation
- **Learnable Metric** (MLP) for similarity computation via Relation Network
- **CBAM Attention** for feature refinement

**Verification Process:**
1. Extract feature embeddings from reference signatures using DenseNet-121 backbone
2. Extract feature embedding from test signature
3. Compute individual similarity scores: each reference vs test signature
4. Average scores to determine final verdict (genuine or forged)

**Model Training:**
- Pretraining: Triplet loss on combined training data
- Meta-Training: Episodic learning with relation network on validation data
- Evaluation: Tested on held-out test set

The model is trained on offline signature datasets (BHSig-Bengali, BHSig-Hindi, CEDAR) 
and achieves high accuracy in distinguishing genuine signatures from skilled forgeries.

**Model Details:**
- Feature Extractor: DenseNet-121 with CBAM attention modules
- Metric Generator: 2-layer MLP with LayerNorm (episodic relation network)
- Input: 224×224 RGB signature images
- Feature Dimension: 1024 per signature
- Metric Embedding: 2048 (concatenated pair)
- Decision Threshold: Adaptive per dataset/split ratio
""")
st.markdown("---")
st.markdown("*Developed for Research in Offline Signature Verification*")
