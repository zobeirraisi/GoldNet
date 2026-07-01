# GoldNet

GoldNet is a public benchmark dataset and code release for visual authentication
of genuine versus counterfeit gold items from ordinary smartphone photographs. It
accompanies the paper:

> **GoldFormer: A Texture-Aware Vision Transformer-Based Algorithm for Detecting
> Near-Identical Images**, Z. Raisi, *Algorithms*, 2026, 19(7), 530.
> DOI: [10.3390/a19070530](https://doi.org/10.3390/a19070530). Open access
> (CC BY 4.0).

The task is fine-grained: high-quality counterfeits closely replicate the surface
texture, hallmark engravings, color, and geometry of genuine pieces, so the two
classes are near-identical to the eye. On a blind subset, trained gold-trading
experts reached 89.80% accuracy, which sets the human baseline the models are
compared against.

## Dataset

- **2,127 images** of physical gold items, one image per item (no item is
  photographed more than once).
- **1,044 authentic** (`real`) and **1,083 counterfeit** (`fake`), a near-balanced
  split (49.1% / 50.9%).
- Captured with several consumer smartphones under varied real-world conditions
  (daylight, indoor, and low-light; a range of angles, distances, and
  backgrounds), with no specialist imaging hardware.
- Items originate primarily from Iran and the wider Persian Gulf market.

### Layout

```
gold/
  real/    # authentic items   (r_img_001.jpg ...)   1,044 images
  fake/    # counterfeit items (f_img_001.jpg ...)   1,083 images
  pairs/   # matched authentic/counterfeit examples used in the paper figures
```

Because each image is a distinct physical item, an image-level train/validation
split is also an item-level split: no item can appear in more than one fold, so
the cross-validation results carry no item-level leakage.

## Code

```
code/        # training and evaluation pipeline (PyTorch)
weights/     # pretrained backbone checkpoints and trained model weights
```

The canonical evaluation uses 5-fold stratified cross-validation, AdamW, AMP
(bfloat16), and a freeze-then-unfreeze fine-tuning schedule.

```bash
# environment (CUDA 11.8 build of PyTorch)
python -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# run the full benchmark (ResNet / VGG / ViT / Swin / GoldFormer ...)
python run_experiments.py

# evaluate only the 2025 backbones (iFormer-S, OverLoCK-XT)
python run_new_models.py

# classical SVM baselines (LBP, Haralick)
python run_svm_only.py
```

`models.py` includes the GoldFormer/TAAG architecture. To reproduce the
published model from the released checkpoint:

```python
import torch
from models import build_model

model = build_model("goldformer")
state = torch.load("weights/GoldFormer_best.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)   # strict — exact match with the released checkpoint
model.eval()
logits, gamma = model(images)  # gamma = TAAG gate activations, for interpretability
```

## Results (5-fold cross-validation, matched 224×224 resolution)

| Model | Accuracy (%) | F1 |
|---|---|---|
| Human experts (baseline) | 89.80 | -- |
| ResNet-101 | 92.29 ± 1.01 | 0.9228 |
| Swin-T | 93.65 ± 0.67 | 0.9365 |
| ViT-B/16 | 94.31 ± 0.94 | 0.9431 |
| Soft-voting ensemble | 94.92 | 0.9492 |
| **GoldFormer (ours)** | **95.02 ± 0.75** | **0.9502** |

GoldFormer is the best single model and beats the ensemble; it is statistically
tied with the strongest individual backbone, ViT-B/16 (paired McNemar p = 0.228),
and significantly beats its own Swin-T backbone (p = 0.014) while using half
ViT-B/16's FLOPs (8.6 vs 16.9 GFLOPs) and fewer parameters (54.3M vs 86.6M). Its
contribution is competitive accuracy together with built-in, attribution-free
texture-gate interpretability.

## Citation

```bibtex
@article{raisi2026goldformer,
  title   = {GoldFormer: A Texture-Aware Vision Transformer-Based Algorithm
             for Detecting Near-Identical Images},
  author  = {Raisi, Zobeir},
  journal = {Algorithms},
  volume  = {19},
  number  = {7},
  pages   = {530},
  year    = {2026},
  doi     = {10.3390/a19070530}
}
```

## License

The code and scripts in this repository are released under the
[MIT License](LICENSE).

The dataset (`gold/` directory) is released under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-DATA).
You are free to use, share, and adapt the data for any purpose, provided you
give appropriate credit and cite the paper above.

## Contact

Zobeir Raisi, Chabahar Maritime University, zobeir.raisi@cmu.ac.ir
