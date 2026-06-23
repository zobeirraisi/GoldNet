# GoldNet

GoldNet is a public benchmark dataset and code release for visual authentication
of genuine versus counterfeit gold items from ordinary smartphone photographs. It
accompanies the paper:

> **GoldFormer: A Texture-Aware Vision Transformer-based Algorithm for Detecting
> Near-Identical Images**, Z. Raisi, *Algorithms* (MDPI), under review.

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

> Note: adjust the names below to match the scripts you upload.

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

## Results (5-fold cross-validation)

| Model | Accuracy (%) | F1 |
|---|---|---|
| Human experts (baseline) | 89.80 | -- |
| ResNet-101 | 92.29 ± 1.01 | 0.9228 |
| ViT-B/16 | 94.31 ± 0.94 | 0.9431 |
| Swin-T | 94.31 ± 0.78 | 0.9431 |
| **GoldFormer (ours)** | **94.69 ± 0.79** | **0.9469** |
| Soft-voting ensemble (best overall) | **95.39** | **0.9539** |

GoldFormer is statistically tied with the strongest individual backbone, Swin-T
(paired McNemar p = 0.48); its contribution is built-in, attribution-free texture
interpretability rather than higher raw accuracy. The training-free ensemble is
the best overall configuration.

## Citation

```bibtex
@article{raisi2026goldformer,
  title   = {GoldFormer: A Texture-Aware Vision Transformer-based Algorithm
             for Detecting Near-Identical Images},
  author  = {Raisi, Zobeir},
  journal = {Algorithms},
  year    = {2026},
  note    = {Under review}
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
