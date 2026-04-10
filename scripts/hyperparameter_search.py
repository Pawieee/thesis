# =============================================================================
# Bayesian Hyperparameter Optimization for tDCBAM using Optuna
#
# Strategy:
#   Trials are given a full epoch budget (e.g., 100). Termination is handled
#   dynamically by early stopping when the TripletLoss active fraction drops 
#   near zero (margin satisfied). Unpromising trials are aggressively pruned 
#   by Optuna's MedianPruner based on intermediate validation EER.
# =============================================================================

import os, sys, json, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import optuna
from optuna.samplers import TPESampler
import multiprocessing

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from models.feature_extractor                  import DenseNetFeatureExtractor
from models.Triplet_Siamese_Similarity_Network import tDCBAM
from losses.triplet_loss                       import TripletLoss
from utils.model_evaluation                    import compute_metrics
from dataloader.tDCBAM_trainloader             import (get_transforms,
                                                        preprocess_image,
                                                        sample_augment_params)

multiprocessing.set_start_method('fork', force=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# DATASET
# =============================================================================

class SplitTripletDataset(Dataset):
    def __init__(self, user_dict, input_shape, val_transform,
                 hard_neg_ratio=0.7):
        self.input_shape    = input_shape
        self.val_transform  = val_transform
        self.hard_neg_ratio = hard_neg_ratio

        self.user_genuine_map  = {}
        self.user_forged_map   = {}
        self.all_genuine_paths = []

        for uid, data in user_dict.items():
            gen_key  = next((k for k in data if k.lower() in ('genuine', 'gen')), None)
            forg_key = next((k for k in data if k.lower() in ('forged', 'forgeries', 'forg')), None)
            gen_paths  = data.get(gen_key,  []) if gen_key  else []
            forg_paths = data.get(forg_key, []) if forg_key else []
            if len(gen_paths) >= 2:
                self.user_genuine_map[uid] = gen_paths
                self.user_forged_map[uid]  = forg_paths
                self.all_genuine_paths.extend((p, uid) for p in gen_paths)

        self.users = list(self.user_genuine_map.keys())
        self._generate_triplets()

    def _generate_triplets(self):
        self.triplets = []
        for anchor_path, uid in self.all_genuine_paths:
            positives = [p for p in self.user_genuine_map[uid] if p != anchor_path]
            if not positives:
                continue
            pos_path  = random.choice(positives)
            forgeries = self.user_forged_map.get(uid, [])
            
            if random.random() < self.hard_neg_ratio and forgeries:
                neg_path = random.choice(forgeries)
            else:
                other_uid = random.choice([u for u in self.users if u != uid])
                neg_path  = random.choice(self.user_genuine_map[other_uid])
            self.triplets.append((anchor_path, pos_path, neg_path))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]

        shared_flip = random.random() < 0.5
        params_a    = sample_augment_params(shared_flip=shared_flip)
        params_p    = sample_augment_params(shared_flip=shared_flip)
        params_n    = sample_augment_params(shared_flip=shared_flip)

        return (
            preprocess_image(Image.open(a_path).convert('RGB'), img_size=self.input_shape, augment_params=params_a),
            preprocess_image(Image.open(p_path).convert('RGB'), img_size=self.input_shape, augment_params=params_p),
            preprocess_image(Image.open(n_path).convert('RGB'), img_size=self.input_shape, augment_params=params_n),
            torch.tensor([1], dtype=torch.float32)
        )


class SplitPairDataset(Dataset):
    def __init__(self, user_dict, input_shape, transform):
        self.input_shape = input_shape
        self.transform   = transform
        self.pairs       = []

        for uid, data in user_dict.items():
            gen_key  = next((k for k in data if k.lower() in ('genuine', 'gen')), None)
            forg_key = next((k for k in data if k.lower() in ('forged', 'forgeries', 'forg')), None)
            gen_paths  = data.get(gen_key,  []) if gen_key  else []
            forg_paths = data.get(forg_key, []) if forg_key else []
            for i in range(len(gen_paths)):
                for j in range(i + 1, len(gen_paths)):
                    self.pairs.append((gen_paths[i], gen_paths[j], 1))
            for g in gen_paths:
                for f in forg_paths:
                    self.pairs.append((g, f, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, q, label = self.pairs[idx]
        return (self._load(s), self._load(q), torch.tensor(label, dtype=torch.float32))

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform:
            return self.transform(img)
        return preprocess_image(img, img_size=self.input_shape, augment=False)


# =============================================================================
# VALIDATION
# =============================================================================

def validate(fe, loader, device):
    fe.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for sup, qry, labels in loader:
            sup  = sup.to(device, non_blocking=True)
            qry  = qry.to(device, non_blocking=True)
            dist = torch.sum((fe(sup) - fe(qry)) ** 2, dim=1)
            scores = 1.0 - (dist / 4.0)
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    return compute_metrics(all_labels, all_scores, return_curve_data=False)


# =============================================================================
# OPTUNA TRIAL TRAINING LOOP
# =============================================================================

def run_training_for_optuna(trial, train_user_dict, val_user_dict, device, input_shape, num_workers,
                            lr, margin, weight_decay, phase1_epochs,
                            backbone_lr_ratio, hard_neg_ratio,
                            scheduler_patience, epochs, zero_active_patience):
    
    seed_everything(42)
    VAL_EVERY = 3
    zero_active_counter = 0
    best_eer = float('inf')

    # ── Transforms & Datasets ─────────────────────────────────────────────────
    val_transform = get_transforms(mode='val', input_shape=input_shape)

    train_dataset = SplitTripletDataset(
        train_user_dict, input_shape=input_shape,
        val_transform=val_transform, hard_neg_ratio=hard_neg_ratio
    )
    val_dataset = SplitPairDataset(
        val_user_dict, input_shape=input_shape, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    # ── Model Initialization ──────────────────────────────────────────────────
    model     = tDCBAM(backbone_name='densenet121', output_dim=1024, pretrained=True).to(device)
    criterion = TripletLoss(margin=margin, mode='euclidean')
    scaler    = torch.amp.GradScaler('cuda')

    # ── Phase 1 Setup ─────────────────────────────────────────────────────────
    for p in model.feature_extractor.get_backbone_params():
        p.requires_grad = False

    optimizer = optim.AdamW(model.get_head_params(), lr=lr, weight_decay=weight_decay)
    scheduler = None

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):

        # Phase transition
        if epoch == phase1_epochs:
            for p in model.feature_extractor.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.get_backbone_params(), 'lr': lr * backbone_lr_ratio},
                {'params': model.get_head_params(), 'lr': lr}
            ], weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=scheduler_patience, min_lr=1e-6
            )

        model.train()
        for anchor, pos, neg, _ in train_loader:
            anchor = anchor.to(device, non_blocking=True)
            pos    = pos.to(device,    non_blocking=True)
            neg    = neg.to(device,    non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = criterion(*model(anchor, pos, neg))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_dataset._generate_triplets()

        # ── Periodic Validation & Optuna Logic ────────────────────────────────
        should_validate = ((epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == epochs)
        
        if should_validate:
            val_metrics = validate(model.feature_extractor, val_loader, device)
            val_eer     = val_metrics['eer']

            if scheduler is not None:
                scheduler.step(val_eer)

            # Report to Optuna for Median Pruning
            trial.report(val_eer, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_eer < best_eer:
                best_eer = val_eer

            # ── Active fraction early stopping ────────────────────────────────
            if epoch >= phase1_epochs:
                if criterion.last_fraction_active < 0.01:
                    zero_active_counter += 1
                    if zero_active_counter >= zero_active_patience:
                        break
                else:
                    zero_active_counter = 0

    return best_eer


# =============================================================================
# OPTUNA STUDY
# =============================================================================

def run_search(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split_file) as f:
        split_data = json.load(f)

    train_dict = split_data['train']
    val_dict   = split_data['val']

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (224, 224)

    print(f"\n{'='*60}")
    print(f"  Hyperparameter Search — {args.dataset.upper()}")
    print(f"  Trials: {args.n_trials} | Max Epochs/trial: {args.epochs_per_trial}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    sampler = TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=3
    )

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'tDCBAM_{args.dataset}'
    )

    def objective(trial):
        # ── Sample hyperparameters ────────────────────────────────────────────
        lr                 = trial.suggest_float('lr',                1e-5, 1e-3, log=True)
        margin             = trial.suggest_float('margin',            0.5,  2.0)
        weight_decay       = trial.suggest_float('weight_decay',      1e-5, 1e-3, log=True)
        phase1_epochs      = trial.suggest_int(  'phase1_epochs',     5,    15)
        backbone_lr_ratio  = trial.suggest_float('backbone_lr_ratio', 0.05, 0.2)
        hard_neg_ratio     = trial.suggest_float('hard_neg_ratio',    0.5,  0.9)
        scheduler_patience = trial.suggest_int(  'scheduler_patience',2,    6)

        # ── Run full training with early stopping ─────────────────────────────
        # No fixed epoch cap per trial — early stopping handles termination.
        # zero_active_patience=5 stops when the model has no learning signal.
        # This guarantees Optuna sees the true best val EER for each config.
        best_val_eer = run_training_for_optuna(
            trial=trial,
            train_user_dict=train_dict,
            val_user_dict=val_dict,
            device=device,
            input_shape=input_shape,
            num_workers=args.num_workers,
            lr=lr,
            margin=margin,
            weight_decay=weight_decay,
            phase1_epochs=phase1_epochs,
            backbone_lr_ratio=backbone_lr_ratio,
            hard_neg_ratio=hard_neg_ratio,
            scheduler_patience=scheduler_patience,
            epochs=args.epochs_per_trial,   # full budget — early stopping cuts it short
            zero_active_patience=5,         # stops when no learning signal remains
        )
        return best_val_eer

    t0 = time.time()
    
    study.enqueue_trial({
        'lr':                 7.18e-4,
        'margin':             0.52,
        'weight_decay':       1.78e-4,
        'phase1_epochs':      13,
        'backbone_lr_ratio':  0.12,
        'hard_neg_ratio':     0.51,
        'scheduler_patience': 5
    })
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    elapsed = time.time() - t0

    # ── Report results ────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Search complete | {args.n_trials} trials | {elapsed:.0f}s")
    print(f"  Best Val EER : {best.value:.4f}")
    print(f"  Best params  :")
    for k, v in best.params.items():
        print(f"    {k:<25} : {v}")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'dataset':          args.dataset,
        'n_trials':         args.n_trials,
        'epochs_per_trial': args.epochs_per_trial,
        'best_val_eer':     best.value,
        'best_params':      best.params,
        'all_trials': [
            {
                'number': t.number,
                'value':  t.value,
                'params': t.params,
                'state':  str(t.state)
            }
            for t in study.trials
        ]
    }
    out_path = os.path.join(
        args.output_dir, f'hparam_search_{args.dataset}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f" > Results saved → {out_path}")

    # ── Optuna visualization ──────────────────────────────────────────────────
    try:
        import optuna.visualization as vis

        fig1 = vis.plot_optimization_history(study)
        fig1.write_image(os.path.join(
            args.output_dir, f'optuna_history_{args.dataset}.png'
        ))

        fig2 = vis.plot_param_importances(study)
        fig2.write_image(os.path.join(
            args.output_dir, f'optuna_importance_{args.dataset}.png'
        ))

        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_image(os.path.join(
            args.output_dir, f'optuna_parallel_{args.dataset}.png'
        ))

        print(" > Optuna plots saved")
    except ImportError:
        print(" > Install plotly + kaleido for Optuna visualization: "
              "pip install plotly kaleido")

    return best.params


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bayesian hyperparameter search for tDCBAM'
    )
    parser.add_argument('--dataset',          type=str, required=True,
                        help='Dataset name (cedar / bhsig_bengali / bhsig_hindi)')
    parser.add_argument('--split_file',       type=str, required=True,
                        help='Path to split JSON file')
    parser.add_argument('--n_trials',         type=int, default=40,
                        help='Number of Optuna trials (default: 40)')
    parser.add_argument('--epochs_per_trial', type=int, default=100,
                        help='Maximum training epochs per trial (default: 100)')
    parser.add_argument('--output_dir',       type=str,
                        default='checkpoints/hparam_search',
                        help='Directory to save search results')
    parser.add_argument('--num_workers',      type=int, default=4)
    args = parser.parse_args()

    best_params = run_search(args)