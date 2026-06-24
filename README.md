# TDCBAM: Enhanced DenseNet-121 with Triplet Network-Based Metric Learning and Convolutional Block Attention for Offline Signature Verification

> Triplet-trained Deep Convolutional network with CBAM attention (tDCBAM) for writer-independent offline handwritten signature verification across CEDAR, BHSig-Bengali, and BHSig-Hindi benchmark datasets.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Backbone: DenseNet-121](#backbone-densenet-121)
  - [CBAM Integration](#cbam-integration)
  - [Projection Head](#projection-head)
  - [L2 Normalization](#l2-normalization)
- [Metric Learning Paradigm](#metric-learning-paradigm)
  - [Triplet Network](#triplet-network)
  - [Triplet Loss](#triplet-loss)
  - [Embedding Space Verification](#embedding-space-verification)
- [Training Protocol](#training-protocol)
  - [Two-Phase Training Strategy](#two-phase-training-strategy)
  - [Hard Negative Mining](#hard-negative-mining)
  - [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Datasets](#datasets)
- [Experimental Results](#experimental-results)
- [Baseline Comparison](#baseline-comparison)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration Reference](#configuration-reference)
- [References](#references)

---

## Overview

Offline handwritten signature verification is a biometric authentication task where a system determines whether a query signature is genuine or forged relative to a set of reference signatures from a known writer. The core challenge is **writer-independent generalization** — the system must discriminate genuine from skilled forgeries for writers never encountered during training.

This repository presents **TDCBAM** (Triplet Deep Convolutional with Block Attention Module), a metric learning framework that addresses this challenge through three integrated contributions:

1. **CBAM-augmented DenseNet-121 backbone** — Convolutional Block Attention Modules (CBAM) are inserted after each Dense Block and before each Transition layer, allowing the network to recalibrate feature responses along both channel and spatial dimensions before feature compression. This directs the backbone's representational capacity toward discriminative stroke-level features relevant to signature identity.

2. **Triplet Network-Based Metric Learning** — Rather than training a closed-set classifier, the model is trained with triplet loss to produce a metric embedding space where genuine signature pairs are pulled together and genuine-forgery pairs are pushed apart. This formulation enables open-set verification: at inference, verification is performed by computing the distance between two embeddings and comparing against a threshold derived from the Equal Error Rate (EER) operating point.

3. **Offline Hard Negative Mining** — Training triplets are constructed with a curriculum that mixes 70% skilled forgeries (hard negatives) from the same writer with 30% genuine signatures from other writers (easy negatives), providing a structured gradient signal that scales with the difficulty of the forgery discrimination task.

The enhanced system is evaluated against a DenseNet-121 classification baseline (Kandeil et al., 2023) across three benchmark datasets using a strict writer-disjoint 70:15:15 train/validation/test split protocol.

---

## Architecture

### Backbone: DenseNet-121

DenseNet-121 (Huang et al., 2017) is a densely connected convolutional network where each layer receives feature maps from all preceding layers within a Dense Block via concatenated skip connections. This dense connectivity pattern has two properties that make it well-suited for signature verification:

- **Feature reuse**: every layer has direct access to gradients from all preceding layers, promoting efficient parameter utilization and reducing vanishing gradient effects in deep networks.
- **Compact feature representation**: the concatenation of multi-scale feature maps produces rich representations that capture both fine-grained stroke details and coarse spatial structure simultaneously.

The DenseNet-121 architecture consists of four Dense Blocks interleaved with three Transition layers, preceded by a stem (Conv7×7, BN, ReLU, MaxPool) and followed by a final BatchNorm and global average pooling:

```
Input (3 × 224 × 224)
    → Stem: Conv7×7(64) + BN + ReLU + MaxPool
    → Dense Block 1 (6 layers, growth rate 32)  → 256 channels
    → Transition 1 (compression 0.5)            → 128 channels
    → Dense Block 2 (12 layers, growth rate 32) → 512 channels
    → Transition 2 (compression 0.5)            → 256 channels
    → Dense Block 3 (24 layers, growth rate 32) → 1024 channels
    → Transition 3 (compression 0.5)            → 512 channels
    → Dense Block 4 (16 layers, growth rate 32) → 1024 channels
    → BN + ReLU + GlobalAvgPool
    → Projection Head
    → L2 Normalization
```

All backbone weights are initialized from ImageNet-pretrained weights (`DenseNet121_Weights.IMAGENET1K_V1`).

### CBAM Integration

CBAM (Woo et al., 2018) is a lightweight attention mechanism that sequentially applies channel attention followed by spatial attention as multiplicative feature recalibration:

```
F_out = F_in × M_c(F_in) × M_s(F_in × M_c(F_in))
```

where `M_c` is the channel attention map and `M_s` is the spatial attention map.

**Channel Attention** aggregates spatial information via global average pooling and global max pooling, passes both through a shared MLP with bottleneck ratio 8, sums the outputs, and applies sigmoid activation:

```
M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
```

**Spatial Attention** aggregates channel information by computing channel-wise average and max pooling, concatenating the results into a 2-channel feature map, and applying a 7×7 convolution followed by sigmoid:

```
M_s(F) = σ( Conv7×7([AvgPool_c(F); MaxPool_c(F)]) )
```

**Placement — architecturally faithful to Woo et al. (2018):** CBAM is inserted **after each Dense Block and before each Transition layer**. This placement is deliberate: the Dense Block produces fully-enriched features at maximum channel resolution via its concatenated skip connections. Applying attention at this point allows the model to suppress irrelevant channels and spatial regions before the Transition's 1×1 convolution and average pooling irreversibly compress the feature maps. For Dense Block 4 (which has no subsequent Transition), CBAM is placed before the final BatchNorm.

| CBAM Position | Channels | Placed Before |
|---|---|---|
| After Dense Block 1 | 256 | Transition 1 |
| After Dense Block 2 | 512 | Transition 2 |
| After Dense Block 3 | 1024 | Transition 3 |
| After Dense Block 4 | 1024 | Final BN (norm5) |

Channel counts are fixed architectural constants derived from the DenseNet-121 growth rate (k=32) and compression factor (θ=0.5) and should not be inferred at runtime.

### Projection Head

Following global average pooling, a regularized projection head maps the 1024-dimensional pooled feature vector to the final embedding space:

```
Head: BatchNorm1d(1024) → Dropout(p=0.5) → Linear(1024, 1024)
```

The BatchNorm1d normalizes the pooled feature distribution before projection. The Dropout layer (p=0.5) provides regularization during training. The Linear layer learns a projection into the embedding space — the model learns directions in this space rather than magnitudes, since L2 normalization is applied subsequently.

### L2 Normalization

The output of the projection head is L2-normalized to constrain all embeddings to the unit hypersphere:

```
z = f(x) / ||f(x)||₂
```

This normalization has two important consequences. First, the Squared Euclidean Distance (SED) between two unit-norm embeddings is bounded to [0, 4] and is monotonically related to cosine similarity:

```
SED(a, b) = ||a - b||² = 2 - 2·cos(a, b)
```

This makes the margin in the triplet loss geometrically interpretable as an angular separation requirement on the unit sphere. Second, the similarity score used for verification can be computed as:

```
similarity(a, b) = 1 - SED(a, b) / 4  ∈ [0, 1]
```

where higher values indicate more similar (more likely genuine) pairs.

---

## Metric Learning Paradigm

### Triplet Network

The tDCBAM model follows the Siamese/triplet network architecture where a single shared-weight feature extractor processes three inputs simultaneously — an Anchor signature, a Positive signature (same writer as anchor), and a Negative signature (different writer or skilled forgery). Weight sharing enforces that the same transformation is applied to all three inputs, producing embeddings in a common metric space.

```
anchor  ─┐
          ├─► shared DenseNetFeatureExtractor ─► (z_a, z_p, z_n)
positive ─┤
negative ─┘
```

At inference time, the triplet structure is collapsed to a pairwise comparison: a support (reference) signature and a query signature are each independently encoded, and their similarity score determines the verification decision.

### Triplet Loss

The triplet loss is the hinge-based margin loss formulated as:

```
L(a, p, n) = max(0,  ||z_a - z_p||² - ||z_a - z_n||² + α)
```

where `α` is the margin hyperparameter. Only **active triplets** — those where the loss is positive (i.e., the margin constraint is violated) — contribute to the gradient. This is critical: averaging over satisfied triplets (loss = 0) would dilute the gradient signal and slow convergence. The mean is computed over active triplets only:

```
L_batch = (1 / |A|) Σ_{(a,p,n) ∈ A} L(a, p, n)
```

where `A` is the set of active triplets in the batch.

**Margin configuration:** With L2-normalized embeddings and SED ∈ [0, 4], the margin `α = 1.0` requires the positive to be at least 1.0 SED units closer to the anchor than the negative. In cosine terms, this corresponds to requiring `cos(a,p) - cos(a,n) ≥ 0.5`, which is a meaningful angular separation constraint on the unit sphere.

### Embedding Space Verification

At inference, verification proceeds as follows:

1. Encode the support (reference) signature: `z_s = f(x_s)`
2. Encode the query signature: `z_q = f(x_q)`
3. Compute similarity: `score = 1 - SED(z_s, z_q) / 4`
4. Compare against threshold: `genuine if score ≥ τ, else forged`

The threshold `τ` is determined from the validation set at the **Equal Error Rate (EER)** operating point — the threshold where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). This threshold is then applied unchanged to the test set for final evaluation.

---

## Training Protocol

### Two-Phase Training Strategy

Training is divided into two phases to balance the stability of pretrained ImageNet features with the need to adapt the backbone to signature-specific representations:

**Phase 1 — Head Warm-Up (epochs 1–20, backbone frozen):**
The DenseNet-121 backbone parameters are frozen. Only the CBAM modules and the projection head are trained. This prevents the randomly initialized head from corrupting the pretrained backbone features with noisy gradients during the initial epochs. The optimizer trains only head parameters:

```
Optimizer: AdamW(head_params, lr=1e-4, weight_decay=1e-4)
```

**Phase 2 — Full Fine-Tuning (epochs 21–100, backbone unfrozen):**
All parameters are unfrozen and trained with differential learning rates — the backbone receives a conservative learning rate to preserve pretrained representations while the head adapts faster:

```
Optimizer: AdamW([
    {backbone_params, lr=1e-5},   # lr × 0.1 — conservative fine-tuning
    {head_params,     lr=1e-4}    # original LR — continued head training
], weight_decay=1e-4)

Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=3, min_lr=1e-6)
```

The scheduler monitors validation EER and halves the learning rate if no improvement is observed for 3 consecutive validation checkpoints (validation runs every 3 epochs).

**Checkpoint saving:** The best model checkpoint is saved based on minimum validation EER. At the end of training, the best checkpoint is reloaded for final test evaluation, ensuring the reported results correspond to the model with the best generalization to unseen validation writers.

### Hard Negative Mining

Training triplets are generated offline at the start of each epoch via `SplitTripletDataset._generate_triplets()`. For each genuine signature serving as an anchor, a positive is sampled uniformly from the remaining genuine signatures of the same writer, and a negative is selected according to the following curriculum:

```
P(skilled forgery from same writer) = 0.70   ← hard negative
P(genuine from different writer)    = 0.30   ← easy negative
```

Skilled forgeries are domain-specific hard negatives — they are designed to closely mimic the genuine writer's stroke patterns and represent the primary challenge in offline signature verification. Including 70% skilled forgeries ensures the model receives strong gradient signal from the most informative negative examples. The 30% easy negatives (other writers' genuines) prevent premature embedding collapse by ensuring some trivially separable pairs are always present in the triplet set.

Triplets are regenerated each epoch to prevent the model from memorizing fixed pairings, effectively providing a different set of training examples at each epoch without requiring additional data.

### Data Preprocessing and Augmentation

All signature images undergo the following deterministic preprocessing pipeline before augmentation:

1. **Grayscale conversion** — RGB to grayscale via OpenCV `COLOR_RGB2GRAY`
2. **Otsu binarization** — adaptive thresholding produces white background (255) and black strokes (0)
3. **Tight crop with margin** — bounding box of stroke pixels expanded by 10px on each side
4. **Aspect-aware resize** — longest dimension scaled to 224px, shorter dimension scaled proportionally
5. **Canvas placement** — resized image placed on a 224×224 white canvas

During training, the following geometric augmentation parameters are sampled **independently per image** within each triplet:

| Parameter | Range | Rationale |
|---|---|---|
| Rotation | [-15°, +15°] | Natural slant variation in handwriting |
| Scale | [0.85, 1.15] | Natural size/pressure variation |
| Y jitter | [0, slack] | Canvas placement variation |
| X jitter | [0, slack] | Canvas placement variation |

Independent augmentation (rather than shared parameters across anchor, positive, negative) is intentional: it forces the model to learn stroke-level similarity invariant to spatial transformation, rather than relying on co-registration between images. Horizontal flipping is explicitly excluded to preserve stroke directionality — a discriminative feature in handwriting.

A small amount of Gaussian noise (σ=5.0) is added to the canvas during training augmentation to break pure white/zero-padding artifacts.

All images are normalized using ImageNet statistics:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## Datasets

| Dataset | Writers | Genuine/Writer | Forged/Writer | Script |
|---|---|---|---|---|
| CEDAR | 55 | 24 | 24 | English |
| BHSig-Bengali | 100 | 24 | 30 | Bengali |
| BHSig-Hindi | 160 | 24 | 30 | Hindi |

All experiments use a **writer-disjoint 70:15:15 split** — training, validation, and test sets contain completely non-overlapping writer identities. A writer whose signatures appear in the training set cannot appear in validation or test under any circumstances. This is enforced at the data preparation stage by `prepare_split_ratios.py` and verified at runtime by explicit set-intersection assertions before each training run.

The evaluation protocol uses **exhaustive pairwise comparison** on the validation and test sets: all genuine-genuine pairs and all genuine-forged pairs from each split are evaluated, producing the complete score distribution from which EER, AUC, and classification metrics are computed.

---

## Experimental Results

All results are reported on the held-out test set using the EER threshold derived from the validation set.

| Dataset | EER  | AUC  | Accuracy  | Precision  | Recall  | F1-score  |
|---|---|---|---|---|---|---|
| CEDAR | 16.03% | 91.94% | 83.97% | 71.51% | 83.98% | 77.24% |
| BHSig-Bengali | 8.01% | 97.12% | 91.99% | 81.49% | 92.00% | 86.43% |
| BHSig-Hindi | 13.69% | 93.63% | 86.31% | 70.74% | 86.31% | 77.75% |



---

## Baseline Comparison

The baseline model is a faithful replication of the DenseNet-121 classification system described in Kandeil et al. (2023), using the exact hyperparameters reported in Table 1 of that paper:

| Configuration | Baseline (Kandeil et al., 2023) | TDCBAM |
|---|---|---|
| Architecture | DenseNet-121 | DenseNet-121 + CBAM |
| Training paradigm | Binary classification | Metric learning (triplet) |
| Loss function | CrossEntropyLoss | TripletLoss (SED, margin=1.0) |
| Optimizer | Adam (lr=0.001, β₁=0.99) | AdamW (lr=1e-4, differential) |
| Backbone freeze | None (full fine-tuning) | Two-phase (Phase 1 frozen) |
| Inference | Softmax probability | L2 distance on unit sphere |
| Output space | 2-class logits | 1024-dim unit hypersphere |

The key architectural difference is the training paradigm. The baseline learns a closed-set classifier that maps signatures to a binary genuine/forged decision for the specific writers seen during training. The TDCBAM model learns an open-set metric space where verification generalizes to unseen writers by distance comparison — this is the fundamental advantage of metric learning over classification for biometric verification tasks.

---

## Repository Structure

```
├── config/
│   ├── configs.json                # Key configurations per dataset
├── dataloader/
│   └── tDCBAM_trainloader.py       # Preprocessing, augmentation, SplitTripletDataset,
│                                   # SplitPairDataset, SplitImageDataset,
│                                   # BalancedBatchSampler
├── losses/
│   ├── triplet_loss.py             # Offline element-wise TripletLoss (SED)
│   └── online_triplet_loss.py      # Online batch hard mining OnlineTripletLoss
├── models/
│   ├── feature_extractor.py        # DenseNetFeatureExtractor (baseline + TDCBAM)
│   │                               # ChannelAttention, SpatialAttention, CBAMBlock
│   └── Triplet_Siamese_Similarity_Network.py  # tDCBAM triplet wrapper

├── notebooks/
│   ├── tDCBAM_cedar.ipynb          # TDCBAM model — CEDAR
│   ├── tDCBAM_bengali.ipynb        # TDCBAM model — BHSig-Bengali
│   ├── tDCBAM_hindi.ipynb          # TDCBAM model — BHSig-Hindi
│   ├── baseline_cedar.ipynb        # Baseline — CEDAR
│   ├── baseline_bengali.ipynb      # Baseline — BHSig-Bengali
│   └── baseline_hindi.ipynb        # Baseline — BHSig-Hindi
├── scripts/
│   └── prepare_split_ratios.py     # Writer-disjoint train/val/test split generator
├── utils/
│   └── model_evaluation.py         # compute_metrics, EER, AUC, plot utilities
├── data/
│   └── ratio_splits/               # Generated split JSON files (gitignored)
├── checkpoints/
│   ├── proposed_splits/            # TDCBAM model checkpoints (gitignored)
│   └── baseline_splits/            # Baseline model checkpoints (gitignored)
├── model_evals/                    # Evaluation plots and metrics (gitignored)
├── main.py                         # Initializes datasets
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

**Requirements:** Python 3.10+, CUDA 11.8+, PyTorch 2.4+, uv for package management

```bash
# Clone the repository
git clone https://github.com/Pawieee/thesis.git
cd thesis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

```
torch>=2.4.0
torchvision>=0.19.0
numpy>=1.26.0
opencv-python>=4.9.0
Pillow>=10.3.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
```

---

## Usage

### 1. Prepare Dataset Splits

Generate writer-disjoint train/val/test splits for all datasets. The script automatically discovers datasets under `--data_root` and outputs split JSON files compatible with both the baseline and TDCBAM pipelines:


**ensure kaggle.json is set**

```bash
uv run main.py
```
Expected data directory structure:

```
data/
├── cedardataset/
│   └── signatures/
│       ├── full_org/         # original_<uid>_<sid>.png
│       └── full_forg/        # forgeries_<uid>_<sid>.png
└── bhsig260-hindi-bengali/
    ├── BHSig100_Bengali/     # B-S-<uid>-G-<sid>.tif / B-S-<uid>-F-<sid>.tif
    └── BHSig160_Hindi/       # H-S-<uid>-G-<sid>.tif / H-S-<uid>-F-<sid>.tif
```
### 2. Train the TDCBAM Model

Open and run the appropriate notebook:

```
notebooks/tDCBAM_bengali.ipynb    # BHSig-Bengali
notebooks/tDCBAM_cedar.ipynb      # CEDAR
notebooks/tDCBAM_hindi.ipynb      # BHSig-Hindi
```

Key configuration parameters:

*Sample values, refer to config.json*
```python
TRAIN_EPOCHS        = 100    # total training epochs
TRAIN_PHASE1_EPOCHS = 20     # epochs with backbone frozen (Phase 1)
TRAIN_LR            = 1e-4   # base learning rate
TRAIN_MARGIN        = 1.0    # triplet loss margin
TRAIN_WEIGHT_DECAY  = 1e-4   # AdamW weight decay
TRAIN_BATCH_SIZE    = 32     # P=8 writers × K=4 samples
FEATURE_DIM         = 1024   # embedding dimension
```

### 3. Train the Baseline Model

```
notebooks/baseline_cedar.ipynb      # CEDAR
notebooks/baseline_bengali.ipynb    # BHSig-Bengali
notebooks/baseline_hindi.ipynb      # BHSig-Hindi
```

The baseline strictly replicates Kandeil et al. (2023) with the following fixed configuration (not user-modifiable without deviating from the paper):

```python
EPOCHS     = 100     # paper: 100
BATCH_SIZE = 30      # paper: 30
LR         = 1e-3    # paper: 0.001
MOMENTUM   = 0.99    # paper: momentum = 0.99 → Adam beta1
```
---

## Configuration Reference

### TDCBAM Model

| Parameter | Value | Description |
|---|---|---|
| `TRAIN_EPOCHS` | 100 | Total training epochs |
| `TRAIN_PHASE1_EPOCHS` | 20 | Backbone frozen epochs (Phase 1) |
| `TRAIN_LR` | 1e-4 | Base learning rate (head) |
| `TRAIN_MARGIN` | 1.0 | Triplet loss margin (SED on unit sphere) |
| `TRAIN_WEIGHT_DECAY` | 1e-4 | AdamW L2 penalty |
| `TRAIN_BATCH_SIZE` | 32 | P×K = 8 writers × 4 samples |
| `FEATURE_DIM` | 1024 | Embedding dimensionality |
| Backbone LR | 1e-5 | Phase 2 backbone LR = LR × 0.1 |
| CBAM ratio | 8 | Channel reduction ratio in MLP bottleneck |
| CBAM kernel | 7 | Spatial attention convolution kernel size |
| Dropout | 0.5 | Projection head dropout probability |
| Hard negative ratio | 0.70 | Fraction of skilled forgery negatives |
| Scheduler factor | 0.5 | LR reduction factor on plateau |
| Scheduler patience | 3 | Validation checkpoints before LR reduction |
| Validation frequency | 3 | Epochs between validation runs |

### Baseline (Kandeil et al., 2023)

| Parameter | Value | Source |
|---|---|---|
| `EPOCHS` | 100 | Table 1 |
| `BATCH_SIZE` | 30 | Table 1 |
| `LR` | 1e-3 | Table 1 |
| `MOMENTUM` | 0.99 | Table 1 (Adam beta1) |
| `WEIGHT_DECAY` | 0.0 | Not specified in paper |
| Freezing | None | Not specified in paper |
| Optimizer | Adam | Table 1 |
| Loss | CrossEntropyLoss | Table 1 (categorical_crossentropy) |

---

## References

```
Kandeil, S. A., Mostafa, E.-S. M. E., & Salama, W. M. (2023).
Signature Verification Based on Deep Learning.
Alexandria Journal of Science and Technology, 1(2), 55–63.
https://doi.org/10.21608/AJST.2023.236375.1016

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017).
Densely Connected Convolutional Networks.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
CBAM: Convolutional Block Attention Module.
Proceedings of the European Conference on Computer Vision (ECCV).

Schroff, F., Kalenichenko, D., & Philbin, J. (2015).
FaceNet: A Unified Embedding for Face Recognition and Clustering.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Hermans, A., Beyer, L., & Leibe, B. (2017).
In Defense of the Triplet Loss for Person Re-Identification.
arXiv preprint arXiv:1703.07737.

Loshchilov, I., & Hutter, F. (2019).
Decoupled Weight Decay Regularization.
International Conference on Learning Representations (ICLR).

Dey, S., Dutta, A., Toledo, J. I., Ghosh, S. K., Lladós, J., & Pal, U. (2017).
SigNet: Convolutional Siamese Network for Writer Independent Offline
Signature Verification. arXiv preprint arXiv:1707.02131.
```
