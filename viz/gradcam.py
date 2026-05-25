import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
# ⭐ Use the 'Agg' backend to suppress GUI pop-ups
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from models.feature_extractor import DenseNetFeatureExtractor
from models.Triplet_Siamese_Similarity_Network import tDCBAM
from dataloader.tDCBAM_trainloader import get_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths for your BHSig Hindi TIFF image and weights
IMG_PATH = os.path.join(REPO_ROOT, 'viz', 'B-S-45-F-27.tif') 
BASELINE_PTH = os.path.join(REPO_ROOT, 'checkpoints', 'baseline_splits', 
                            'bhsig_hindi_65_18_18', 'best_baseline_model.pth')
PROPOSED_PTH = os.path.join(REPO_ROOT, 'checkpoints', 'proposed_splits', 
                            'bhsig_hindi_65_18_18', 'best_triplet_model.pth')

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)
        if output.shape[1] > 2:
            score = output.norm() 
        else:
            class_idx = class_idx or output.argmax(dim=1).item()
            score = output[0, class_idx]
        
        score.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-10)
        return heatmap.detach().cpu().numpy()

def visualize_comparison(img_path, baseline_path, proposed_path):
    """Generates the Baseline vs Proposed comparison silently."""
    raw_img = Image.open(img_path)
    if getattr(raw_img, "is_animated", False): raw_img.seek(0)
    raw_img = raw_img.convert('RGB')
    
    transform = get_transforms(mode='val', input_shape=(224, 224))
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    baseline = DenseNetFeatureExtractor(backbone_name='densenet121', output_dim=2, baseline=True).to(DEVICE)
    baseline.load_state_dict(torch.load(baseline_path, map_location=DEVICE)['model_state_dict'])
    baseline.eval()

    proposed = tDCBAM(backbone_name='densenet121', output_dim=1024).to(DEVICE)
    proposed.feature_extractor.load_state_dict(torch.load(proposed_path, map_location=DEVICE)['feature_extractor'])
    proposed.eval()
    
    heatmap_base = GradCAM(baseline, baseline.backbone.denseblock4).generate_heatmap(input_tensor)
    heatmap_prop = GradCAM(proposed.feature_extractor, proposed.feature_extractor.cbam4).generate_heatmap(input_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    display_img = np.array(raw_img.resize((224, 224)))
    axes[0].imshow(display_img); axes[0].set_title("Input Signature"); axes[0].axis('off')
    
    for ax, hm, title in zip([axes[1], axes[2]], [heatmap_base, heatmap_prop], 
                             ["Baseline (Final Dense Block)", "Proposed (Final CBAM Block)"]):
        hm_resized = cv2.resize(hm, (224, 224))
        heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(display_img, 0.6, heatmap_rgb, 0.4, 0)
        ax.imshow(superimposed); ax.set_title(title, fontsize=12, fontweight='bold'); ax.axis('off')

    plt.tight_layout()
    output_filename = os.path.join(REPO_ROOT, 'viz', 'gradcam_comparison.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f" > Comparison plot saved → {output_filename}")
    plt.close(fig) # ⭐ Close figure instead of show()

def visualize_progression(img_path, proposed_path):
    """Generates the 8-stage progression plot silently."""
    raw_img = Image.open(img_path)
    if getattr(raw_img, "is_animated", False): raw_img.seek(0)
    raw_img = raw_img.convert('RGB')
    
    transform = get_transforms(mode='val', input_shape=(224, 224))
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    display_img = np.array(raw_img.resize((224, 224)))

    model = tDCBAM(backbone_name='densenet121', output_dim=1024).to(DEVICE)
    model.feature_extractor.load_state_dict(torch.load(proposed_path, map_location=DEVICE)['feature_extractor'])
    model.eval()

    fe = model.feature_extractor
    target_layers = [
        (fe.block1, "Dense Block 1"), (fe.cbam1, "CBAM Block 1"),
        (fe.block2, "Dense Block 2"), (fe.cbam2, "CBAM Block 2"),
        (fe.block3, "Dense Block 3"), (fe.cbam3, "CBAM Block 3"),
        (fe.block4, "Dense Block 4"), (fe.cbam4, "CBAM Block 4")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Proposed Model Focus Progression: DenseBlock 1 to CBAM 4", fontsize=16, fontweight='bold')

    for i, (layer, name) in enumerate(target_layers):
        row, col = i // 4, i % 4
        heatmap = GradCAM(fe, layer).generate_heatmap(input_tensor)
        hm_resized = cv2.resize(heatmap, (224, 224))
        heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(display_img, 0.6, heatmap_rgb, 0.4, 0)
        
        axes[row, col].imshow(superimposed)
        axes[row, col].set_title(name, fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(REPO_ROOT, 'viz', 'gradcam_progression.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f" > Progression plot saved → {output_filename}")
    plt.close(fig) # ⭐ Close figure instead of show()

# ── Execute both visualizations ───────────────────────────────────────────
visualize_comparison(IMG_PATH, BASELINE_PTH, PROPOSED_PTH)
visualize_progression(IMG_PATH, PROPOSED_PTH)