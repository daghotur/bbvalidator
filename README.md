# ProteinScoreModel

A deep learning pipeline for scoring the **foldability of generated protein backbones**. Given raw backbone coordinates `[N, Cα, C]`, the model predicts whether a structure will fold successfully, estimates RMSD from native, flags steric clashes and hydrogen-bond patterns, and classifies the dominant failure mode — all in a single forward pass.

---

## Architecture overview

```
Backbone coords [B, L, 3, 3]
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  BiophysicalFrontend          (frozen, no_grad)             │
│                                                             │
│  BackboneGeometry   FoldabilityProxies   PairFeatureBuilder │
│  φ ψ ω · clashes   packing · burial ·   RBF16 · kNN16 ·   │
│  H-bonds → 10 f.   PCA-frag  → 21 f.    seq-sep → 20 f.   │
│                                                             │
│  node feats [B,L,31]      pair feats [B,L,L,20] + kNN edges│
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
                ▼                         │
┌──────────────────────────────────────┐  │
│  HybridProteinEncoder                │  │
│                                      │  │
│  PyGGraphMessageLayer ×2  ◄──────────┘  │
│  (GRU-gate MPNN + grad-ckpt)            │
│            │                            │
│            ▼                            │
│  TransformerEncoder ×4                  │
│  Pre-LN · bfloat16 · d_model=192        │
└───────────────┬──────────────────────┘
                │  [B, L, 192]
                ▼
┌──────────────────────────────────────┐
│  MultiHeadAttentionPooling  (H=4)    │
│  softmax over L  →  [B, 192]         │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  ProteinMultiTaskHeads               │
│                                      │
│  fold_logit  rmsd  steric  hbond     │
│  failure_mode (4-class)              │
└───────────────┬──────────────────────┘
                │
                ▼
   DynamicMultiTaskLoss (Kendall & Gal 2018)
   Σ 0.5·exp(−sᵢ)·Lᵢ + 0.5·sᵢ   (sᵢ learnable)
```

The frontend computes physics-based features **once** and caches them. During inference the encoder is run `mc_runs` times with dropout enabled to produce calibrated uncertainty estimates (MC-Dropout).

---

## Repository structure

```
.
├── dataset/
│   └── dataloader.py          # ProteinManifestDataset, protein_collate_fn, get_dataloaders
├── preprocess/
│   ├── biophys_frontend.py    # BiophysicalFrontend — orchestrates all feature extractors
│   ├── geometry_features.py   # BackboneGeometryExtractor (φ ψ ω, clashes, H-bonds)
│   ├── foldability_features.py# FoldabilityProxies (packing, burial, PCA fragment similarity)
│   └── pair_features.py       # PairFeatureBuilder (RBF distances, kNN graph)
├── model/
│   ├── encoder.py             # HybridProteinEncoder (MPNN + Transformer)
│   └── heads_loss.py          # Pooler, heads, DynamicMultiTaskLoss, ProteinScoreModel
├── train_model.py             # Training loop with AMP, GradScaler, TensorBoard
├── inference.py               # CLI inference with MC-Dropout uncertainty
├── checkpoints/               # Saved model weights (created at training time)
└── runs/                      # TensorBoard logs (created at training time)
```

---

## Installation

```bash
pip install torch torchvision torch-geometric biotite scipy pandas h5py tensorboard
```
---

## Data format

The dataset is described by a **manifest CSV** with the following required columns:

| Column | Type | Description |
|---|---|---|
| `split` | str | `train` / `val` / `test` |
| `source_h5` | str | Path to the HDF5 file containing this sample |
| `h5_group_key` | str | Key of the group within the HDF5 file |
| `label` | float | 1.0 = foldable, 0.0 = decoy |

Each HDF5 group must contain a `coords` dataset of shape `[L, 3, 3]` (residues × atoms `{N, Cα, C}` × xyz) stored as `float32`. Optional group attributes:

| Attribute | Default | Description |
|---|---|---|
| `rmsd_target` | 0.0 | Backbone RMSD to native (Å) |
| `steric_target` | 0.0 | Normalised steric clash count |
| `hbond_target` | 0.0 | Normalised H-bond count |
| `failure_mode_label` | 0 | 0=Ok · 1=Clash · 2=Core · 3=Loop |

---

## Training

```bash
python train_model.py
```

Key hyperparameters are set inside `main()`:

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 192 | Encoder hidden dimension |
| `num_graph_layers` | 2 | MPNN layers |
| `num_transformer_layers` | 4 | Transformer encoder layers |
| `num_heads` | 8 | Attention heads |
| `dropout` | 0.15 | Dropout rate (encoder + heads) |
| `batch_size` | 16 | Samples per batch |
| `num_epochs` | 15 | Training epochs |
| `lr` (model) | 3e-4 | AdamW learning rate for network weights |
| `lr` (loss) | 1e-3 | AdamW learning rate for task uncertainty weights |

Training uses **bfloat16 automatic mixed precision** and **gradient checkpointing** inside MPNN layers. The best checkpoint (by validation accuracy) is written to `checkpoints/best_model.pth`. TensorBoard logs are written to `runs/ProteinScoreModel`.

```bash
tensorboard --logdir runs/
```

---

## Inference API

### Command-line

```bash
# Single PDB file
python inference.py -i path/to/structure.pdb -c checkpoints/best_model.pth

# Directory of PDB files
python inference.py -i path/to/pdb_dir/ -c checkpoints/best_model.pth

# Force CPU, increase MC-Dropout passes
python inference.py -i structure.pdb --cpu -m 32
```

Output columns:

| Column | Description |
|---|---|
| `P(Fold)` | Mean probability of successful folding (0–1) |
| `Uncert.` | Variance across MC-Dropout passes (epistemic uncertainty) |
| `Pred RMSD` | Predicted backbone RMSD to native (Å) |
| `Len` | Sequence length (residues) |

Visual indicator: ✅ `P > 0.8` · ⚠️ `P > 0.4` · ❌ `P ≤ 0.4`

---
