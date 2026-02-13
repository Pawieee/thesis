import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sys

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
    'CEDAR': 'cedar'
}

SPLIT_RATIOS = {
    '60% Train / 20% Val / 20% Test': '60_20_20',
    '65% Train / 18% Val / 18% Test': '65_18_18',
    '70% Train / 15% Val / 15% Test': '70_15_15',
    '70% Train / 30% Test': '70_30',
    '80% Train / 20% Test': '80_20',
    '90% Train / 10% Test': '90_10'
}

IMAGE_SIZE = 224
FEATURE_DIM = 1024
EMBEDDING_DIM = 2048
THRESHOLD = 0.0094


# ==========================================
# MODEL LOADING (with caching)
# ==========================================
@st.cache_resource
def load_models(dataset_name, split_ratio):
    """Load feature extractor and metric generator models."""
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
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=256,
            dropout=0.3
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


# ==========================================
# IMAGE PROCESSING
# ==========================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Use RGB conversion to match .convert('RGB') in your TripletDataset
    image = image.convert('RGB') 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# ==========================================
# FEATURE EXTRACTION & SIMILARITY SCORING
# ==========================================
def extract_features(image: Image.Image, feature_extractor) -> torch.Tensor:
    """Extract feature embeddings from an image."""
    tensor = preprocess_image(image)
    
    with torch.no_grad():
        features = feature_extractor(tensor)
    
    return features.squeeze(0)  # Remove batch dimension


def compute_similarity(feat1: torch.Tensor, feat2: torch.Tensor, 
                      metric_generator) -> float:
    """Compute similarity score between two feature vectors using the metric generator."""
    # Concatenate features
    combined = torch.cat([feat1, feat2], dim=0).unsqueeze(0)  # [1, 2048]
    
    with torch.no_grad():
        logit = metric_generator(combined)
    
    # Apply sigmoid to convert logit to probability
    score = torch.sigmoid(logit).item()
    
    return score


def verify_signature_prototype(test_features, prototype_features, metric_generator):
    """
    Implements the Relation Network logic: 
    Score = g(Concat(Prototype, Query))
    """
    # Concatenate according to embedding_dim=2048 (1024 + 1024)
    # Ensure shape is [1, 2048] for the Linear layer in MetricGenerator
    combined = torch.cat([prototype_features, test_features], dim=0).unsqueeze(0)
    
    with torch.no_grad():
        logit = metric_generator(combined)
        # Use sigmoid as done in your notebook's meta_validate function
        score = torch.sigmoid(logit).item() 
    
    prediction = "🟢 GENUINE" if score >= THRESHOLD else "🔴 FORGED"
    
    # Simple confidence metric relative to your decision threshold
    conf = abs(score - THRESHOLD) / max(THRESHOLD, 1 - THRESHOLD)
    conf = min(conf, 1.0) * 100
    
    return score, prediction, conf


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

# ==========================================
# MAIN APPLICATION
# ==========================================
st.markdown("---")

# Load models
dataset_name = DATASETS[selected_dataset]
split_ratio = SPLIT_RATIOS[selected_split]

with st.spinner(f"Loading model for {selected_dataset} ({selected_split})..."):
    feature_extractor, metric_generator, error = load_models(dataset_name, split_ratio)

if error:
    st.error(f"❌ Failed to load model: {error}")
    st.stop()

st.success("✅ Model loaded successfully!")

# Create two columns for instructions
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 👤 Reference Signatures")
    st.markdown("Upload 3 **genuine** reference signatures from the same person")

with col2:
    st.markdown("### 🔍 Test Signature")
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
        st.image(ref1, use_column_width=True)

with col2:
    st.markdown("**Reference 2**")
    ref2 = st.file_uploader("Upload reference signature 2", type=['jpg', 'png', 'jpeg'],
                            key='ref2', label_visibility='collapsed')
    if ref2:
        ref_images.append(Image.open(ref2))
        st.image(ref2, use_column_width=True)

with col3:
    st.markdown("**Reference 3**")
    ref3 = st.file_uploader("Upload reference signature 3", type=['jpg', 'png', 'jpeg'],
                            key='ref3', label_visibility='collapsed')
    if ref3:
        ref_images.append(Image.open(ref3))
        st.image(ref3, use_column_width=True)

with col4:
    st.markdown("**Test Signature**")
    test_img = st.file_uploader("Upload test signature", type=['jpg', 'png', 'jpeg'],
                               key='test', label_visibility='collapsed')
    if test_img:
        st.image(test_img, use_column_width=True)

# ==========================================
# PREDICTION
# ==========================================
st.markdown("---")

if st.button("🚀 Verify Signature", type="primary", use_container_width=True):
    
    # Validation
    if len(ref_images) < 3:
        st.error("❌ Please upload all 3 reference signatures")
    elif test_img is None:
        st.error("❌ Please upload a test signature")
    else:
        with st.spinner("Processing signatures and creating prototype..."):
            try:
                # 1. Extract features for Test Image
                test_image = Image.open(test_img)
                test_features = extract_features(test_image, feature_extractor) # [1024]
                
                # 2. Extract and Aggregate features for References
                reference_features_tensors = []
                for ref_img in ref_images:
                    feat = extract_features(ref_img, feature_extractor)
                    reference_features_tensors.append(feat)
                
                # RESEARCH MATCH: Create a single 'Prototype' by averaging reference embeddings
                # This matches how K-shot (K=3) is conceptually handled in Relation Networks
                all_refs_stack = torch.stack(reference_features_tensors) # [3, 1024]
                prototype_features = torch.mean(all_refs_stack, dim=0)   # [1024]
                
                # 3. Verify signature using the Prototype vs Test
                # We now unpack 3 values to match the updated function below
                final_score, prediction, confidence = verify_signature_prototype(
                    test_features, prototype_features, metric_generator
                )
                
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
                
                # Detailed scores
                st.markdown("### Individual Similarity Scores")
                score_data = {
                    'Reference': ['Reference 1', 'Reference 2', 'Reference 3'],
                    'Similarity Score': [f"{s:.4f}" for s in ind_scores],
                    'Match %': [f"{s*100:.2f}%" for s in ind_scores]
                }
                
                import pandas as pd
                df = pd.DataFrame(score_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Final average score
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Similarity", f"{final_score:.4f}", 
                             delta=f"{(final_score - THRESHOLD)*100:.2f}%")
                
                with col2:
                    st.metric("Decision Threshold", f"{THRESHOLD:.4f}")
                
                with col3:
                    decision = "✅ MATCH" if final_score >= THRESHOLD else "❌ NO MATCH"
                    st.metric("Final Decision", decision)
                
                # Score visualization
                st.markdown("---")
                st.markdown("### Score Distribution")
                
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Bar chart for individual scores
                ax1.bar(range(1, 4), ind_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax1.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD:.4f})')
                ax1.set_xlabel('Reference Signature')
                ax1.set_ylabel('Similarity Score')
                ax1.set_title('Similarity Scores vs References')
                ax1.set_ylim([0, 1])
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # Gauge chart showing final score
                ax2.barh(['Final\nScore'], [final_score], color='#1f77b4', height=0.4)
                ax2.axvline(x=THRESHOLD, color='r', linestyle='--', linewidth=2, label='Threshold')
                ax2.set_xlim([0, 1])
                ax2.set_xlabel('Similarity Score')
                ax2.set_title('Final Verification Score')
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
- **Triplet Loss** for feature learning
- **Meta-Learning** for few-shot adaptation
- **Learnable Metric** (MLP) for similarity computation
- **CBAM Attention** for feature refinement

The model is trained on offline signature datasets (BHSig-Bengali, BHSig-Hindi, CEDAR) 
and achieves high accuracy in distinguishing genuine signatures from skilled forgeries.

**Model Details:**
- Feature Extractor: DenseNet-121 with CBAM
- Metric Generator: 2-layer MLP with LayerNorm
- Training: 2-stage (pretraining with triplet loss + meta-learning with relation loss)
""")
st.markdown("---")
st.markdown("*Developed for Research in Offline Signature Verification*")
