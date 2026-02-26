## Thesis
```
thesis
├─ README.md
├─ backend
│  ├─ app.py
│  ├─ models
│  │  ├─ Triplet_Siamese_Similarity_Network.py
│  │  ├─ __init__.py
│  │  ├─ feature_extractor.py
│  │  └─ meta_learner.py
│  ├─ requirements.txt
│  └─ weights
│     ├─ baseline_splits
│     │  ├─ best_bhsig_bengali_60_20_20.pth
│     │  ├─ best_bhsig_bengali_65_18_18.pth
│     │  ├─ best_bhsig_bengali_70_15_15.pth
│     │  ├─ best_bhsig_hindi_60_20_20.pth
│     │  ├─ best_bhsig_hindi_65_18_18.pth
│     │  ├─ best_bhsig_hindi_70_15_15.pth
│     │  ├─ best_cedar_60_20_20.pth
│     │  ├─ best_cedar_65_18_18.pth
│     │  └─ best_cedar_70_15_15.pth
│     └─ proposed_splits
│        ├─ bhsig_bengali_60_20_20
│        ├─ bhsig_bengali_65_18_18
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ bhsig_bengali_70_15_15
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ bhsig_hindi_60_20_20
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ bhsig_hindi_65_18_18
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ bhsig_hindi_70_15_15
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ cedar_60_20_20
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ cedar_65_18_18
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        ├─ cedar_70_15_15
│        │  ├─ best_meta_model.pth
│        │  └─ pretrained_backbone.pth
│        └─ combined_70_15_15
│           ├─ best_meta_model.pth
│           └─ pretrained_backbone.pth
├─ frontend
│  ├─ README.md
│  ├─ eslint.config.js
│  ├─ index.html
│  ├─ package-lock.json
│  ├─ package.json
│  ├─ public
│  │  └─ vite.svg
│  ├─ src
│  │  ├─ App.css
│  │  ├─ App.jsx
│  │  ├─ components
│  │  │  ├─ charts
│  │  │  │  ├─ PGenuineGauge.jsx
│  │  │  │  ├─ PerSupportChart.jsx
│  │  │  │  ├─ ProbDistChart.jsx
│  │  │  │  └─ chartUtils.jsx
│  │  │  ├─ icons
│  │  │  │  └─ Icons.jsx
│  │  │  ├─ panels
│  │  │  │  ├─ BaselinePanel.jsx
│  │  │  │  ├─ ModelComparisonPanel.jsx
│  │  │  │  ├─ ProposedPanel.jsx
│  │  │  │  └─ VoteDots.jsx
│  │  │  └─ ui
│  │  │     └─ Primitives.jsx
│  │  ├─ constants
│  │  │  ├─ config.js
│  │  │  └─ theme.js
│  │  ├─ hooks
│  │  │  └─ useVerification.js
│  │  ├─ index.css
│  │  └─ main.jsx
│  └─ vite.config.js
└─ model-training
   ├─ .python-version
   ├─ checkpoints
   │  ├─ CBAM_splits
   │  │  ├─ CBAM_bhsig_bengali_results.json
   │  │  ├─ CBAM_bhsig_hindi_results.json
   │  │  ├─ CBAM_cedar_results.json
   │  │  ├─ baseline_bhsig_bengali_comparison.png
   │  │  ├─ baseline_bhsig_bengali_results.json
   │  │  ├─ baseline_bhsig_hindi_comparison.png
   │  │  ├─ baseline_bhsig_hindi_results.json
   │  │  ├─ best_bhsig_bengali_60_20_20.pth
   │  │  ├─ best_bhsig_bengali_65_18_18.pth
   │  │  ├─ best_bhsig_bengali_70_15_15.pth
   │  │  ├─ best_bhsig_hindi_60_20_20.pth
   │  │  ├─ best_bhsig_hindi_65_18_18.pth
   │  │  ├─ best_bhsig_hindi_70_15_15.pth
   │  │  ├─ best_cedar_60_20_20.pth
   │  │  ├─ best_cedar_65_18_18.pth
   │  │  ├─ best_cedar_70_15_15.pth
   │  │  ├─ cbam_bhsig_bengali_comparison.png
   │  │  ├─ cbam_bhsig_hindi_comparison.png
   │  │  └─ cbam_cedar_comparison.png
   │  ├─ baseline_splits
   │  │  ├─ baseline_bhsig_bengali_comparison.png
   │  │  ├─ baseline_bhsig_bengali_results.json
   │  │  ├─ baseline_bhsig_hindi_comparison.png
   │  │  ├─ baseline_bhsig_hindi_results.json
   │  │  ├─ baseline_cedar_comparison.png
   │  │  ├─ baseline_cedar_results.json
   │  │  ├─ best_bhsig_bengali_60_20_20.pth
   │  │  ├─ best_bhsig_bengali_65_18_18.pth
   │  │  ├─ best_bhsig_bengali_70_15_15.pth
   │  │  ├─ best_bhsig_bengali_70_30.pth
   │  │  ├─ best_bhsig_bengali_80_20.pth
   │  │  ├─ best_bhsig_bengali_90_10.pth
   │  │  ├─ best_bhsig_hindi_60_20_20.pth
   │  │  ├─ best_bhsig_hindi_65_18_18.pth
   │  │  ├─ best_bhsig_hindi_70_15_15.pth
   │  │  ├─ best_bhsig_hindi_70_30.pth
   │  │  ├─ best_bhsig_hindi_80_20.pth
   │  │  ├─ best_bhsig_hindi_90_10.pth
   │  │  ├─ best_cedar_60_20_20.pth
   │  │  ├─ best_cedar_65_18_18.pth
   │  │  ├─ best_cedar_70_15_15.pth
   │  │  ├─ best_cedar_70_30.pth
   │  │  ├─ best_cedar_80_20.pth
   │  │  └─ best_cedar_90_10.pth
   │  ├─ combined_prototype
   │  │  ├─ combined_70_15_15
   │  │  │  ├─ best_meta_model.pth
   │  │  │  └─ pretrained_backbone.pth
   │  │  └─ combined_prototype_results.json
   │  └─ proposed_splits
   │     ├─ bhsig_bengali_60_20_20
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ bhsig_bengali_65_18_18
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ bhsig_bengali_70_15_15
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ bhsig_hindi_60_20_20
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ bhsig_hindi_65_18_18
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ bhsig_hindi_70_15_15
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ cedar_60_20_20
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ cedar_65_18_18
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ cedar_70_15_15
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ combined_70_15_15
   │     │  ├─ best_meta_model.pth
   │     │  └─ pretrained_backbone.pth
   │     ├─ proposed_bhsig_bengali_comparison.png
   │     ├─ proposed_bhsig_bengali_results.json
   │     ├─ proposed_bhsig_hindi_comparison.png
   │     ├─ proposed_bhsig_hindi_results.json
   │     ├─ proposed_cedar_comparison.png
   │     ├─ proposed_cedar_results.json
   │     ├─ triplet_bhsig_bengali_comparison.png
   │     ├─ triplet_bhsig_bengali_results.json
   │     ├─ triplet_bhsig_hindi_comparison.png
   │     ├─ triplet_bhsig_hindi_results.json
   │     ├─ triplet_cedar_comparison.png
   │     └─ triplet_cedar_results.json
   ├─ configs
   │  ├─ __init__.py
   │  └─ config_tDCBAM.yaml
   ├─ dataloader
   │  ├─ __init__.py
   │  ├─ meta_dataloader.py
   │  └─ tDCBAM_trainloader.py
   ├─ losses
   │  ├─ __init__.py
   │  └─ triplet_loss.py
   ├─ main.py
   ├─ model_evals
   │  ├─ CBAM_bengali
   │  │  ├─ CBAM_bengali_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ CBAM_bengali_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ CBAM_bengali_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ CBAM_cedar
   │  │  ├─ CBAM_cedar_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ CBAM_cedar_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ CBAM_cedar_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ CBAM_hindi
   │  │  ├─ CBAM_hindi_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ CBAM_hindi_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ CBAM_hindi_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ baseline_bengali
   │  │  ├─ baseline_bengali_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ baseline_bengali_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ baseline_bengali_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ baseline_cedar
   │  │  ├─ baseline_cedar_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ baseline_cedar_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ baseline_cedar_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ baseline_hindi
   │  │  ├─ baseline_hindi_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ baseline_hindi_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ baseline_hindi_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ combined_prototype
   │  │  ├─ confusion_matrix.png
   │  │  ├─ det_curve.png
   │  │  ├─ far_frr.png
   │  │  ├─ roc_curve.png
   │  │  └─ score_distribution.png
   │  ├─ proposed_bengali
   │  │  ├─ proposed_bengali_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ proposed_bengali_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ proposed_bengali_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ proposed_cedar
   │  │  ├─ proposed_cedar_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ proposed_cedar_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ proposed_cedar_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ proposed_hindi
   │  │  ├─ proposed_hindi_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ proposed_hindi_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ proposed_hindi_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ triplet_bengali
   │  │  ├─ triplet_bengali_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ triplet_bengali_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ triplet_bengali_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  ├─ triplet_cedar
   │  │  ├─ triplet_cedar_60-20-20
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  ├─ triplet_cedar_65-18-18
   │  │  │  ├─ confusion_matrix.png
   │  │  │  ├─ det_curve.png
   │  │  │  ├─ far_frr.png
   │  │  │  ├─ roc_curve.png
   │  │  │  └─ score_distribution.png
   │  │  └─ triplet_cedar_70-15-15
   │  │     ├─ confusion_matrix.png
   │  │     ├─ det_curve.png
   │  │     ├─ far_frr.png
   │  │     ├─ roc_curve.png
   │  │     └─ score_distribution.png
   │  └─ triplet_hindi
   │     ├─ triplet_hindi_60-20-20
   │     │  ├─ confusion_matrix.png
   │     │  ├─ det_curve.png
   │     │  ├─ far_frr.png
   │     │  ├─ roc_curve.png
   │     │  └─ score_distribution.png
   │     ├─ triplet_hindi_65-18-18
   │     │  ├─ confusion_matrix.png
   │     │  ├─ det_curve.png
   │     │  ├─ far_frr.png
   │     │  ├─ roc_curve.png
   │     │  └─ score_distribution.png
   │     └─ triplet_hindi_70-15-15
   │        ├─ confusion_matrix.png
   │        ├─ det_curve.png
   │        ├─ far_frr.png
   │        ├─ roc_curve.png
   │        └─ score_distribution.png
   ├─ models
   │  ├─ Triplet_Siamese_Similarity_Network.py
   │  ├─ __init__.py
   │  ├─ feature_extractor.py
   │  └─ meta_learner.py
   ├─ notebooks
   │  ├─ CBAM_bengali.ipynb
   │  ├─ CBAM_cedar.ipynb
   │  ├─ CBAM_hindi.ipynb
   │  ├─ baseline_bengali.ipynb
   │  ├─ baseline_cedar.ipynb
   │  ├─ baseline_hindi.ipynb
   │  ├─ combined_prototype.ipynb
   │  ├─ proposed_bengali.ipynb
   │  ├─ proposed_cedar.ipynb
   │  ├─ proposed_hindi.ipynb
   │  ├─ triplet_bengali.ipynb
   │  ├─ triplet_cedar.ipynb
   │  └─ triplet_hindi.ipynb
   ├─ pyproject.toml
   ├─ requirements.txt
   ├─ scripts
   │  ├─ __init__.py
   │  ├─ prepare_kfold_splits.py
   │  ├─ prepare_split_ratios.py
   │  └─ restructure_bhsig.py
   ├─ streamlit_new_app.py
   ├─ streamlit_old_app.py
   ├─ utils
   │  ├─ __init__.py
   │  ├─ helpers.py
   │  └─ model_evaluation.py
   └─ uv.lock

```