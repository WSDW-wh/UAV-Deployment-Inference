# UAV Position Prediction

This project predicts two UAV positions from JAM situation images. It keeps the full workflow needed to generate/load data, train the model, save checkpoints, and evaluate saved models.

## Project Structure

```text
.
|-- JAM_data/                         # dataset images, CSV files, and data utilities
|-- Generate_JAM_dataset.py           # synthetic JAM dataset generation
|-- JAM_model.py                      # JAM image generation model
|-- utils.py                          # physical model and helper functions
|-- data_load_double.py               # paired image regression dataloader
|-- EAA_tea_train.py                  # main root training script, saves best_feat_model.pth
|-- test_all.py                       # evaluate best_feat_model.pth on the test set
|-- viz_gt_pred_maps.py               # visualize input / GT / prediction maps
|-- best_feat_model.pth               # saved root model checkpoint
`-- dual_backbone_experiment/         # ablation experiment: single/dual, ResNet18, attention
```

## Environment

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if GPU acceleration is needed.

## Dataset

The default dataset path is `JAM_data/`. The main scripts expect these paired CSV files:

```text
JAM_data/re_train_paired.csv
JAM_data/re_test_paired.csv
```

Each paired CSV should contain:

```text
re_image,image,x1,y1,x2,y2
```

To generate synthetic images/labels:

```bash
python Generate_JAM_dataset.py
```

If raw CSV files are regenerated, make sure the paired CSV files still point to both `re_images` and `images`.

Useful data utilities:

```bash
python JAM_data/update_re_image_paths.py
python JAM_data/normalize_bbox_order.py
python JAM_data/recolor_blue_regions.py
```

## Train Root Model

```bash
python EAA_tea_train.py
```

The best checkpoint is saved as:

```text
best_feat_model.pth
```

## Evaluate Saved Root Model

```bash
python test_all.py
```

This loads `best_feat_model.pth`, evaluates the test set, prints MAE/RMSE and distance error metrics, and writes `test_predictions.npy`.

## Visualize Predictions

```bash
python viz_gt_pred_maps.py
```

The script loads `best_feat_model.pth` and saves visual comparison images to `viz_out/`.

## Dual Backbone Experiments

Run from the experiment directory:

```bash
cd dual_backbone_experiment
```

Single-input ResNet18:

```bash
python train_dual_backbone.py --backbone resnet18 --save_dir runs_exp/single_resnet18
```

Single-input ResNet18 + attention:

```bash
python train_dual_backbone.py --backbone resnet18 --use_attention --save_dir runs_exp/single_resnet18_attn
```

Dual-input ResNet18:

```bash
python train_dual_backbone.py --backbone resnet18 --use_dual --save_dir runs_exp/dual_resnet18
```

Dual-input ResNet18 + attention:

```bash
python train_dual_backbone.py --backbone resnet18 --use_dual --use_attention --save_dir runs_exp/dual_resnet18_attn
```

Evaluate all saved dual-backbone checkpoints:

```bash
python eval_situation_iou.py
```

Existing saved checkpoints are kept under `dual_backbone_experiment/runs_exp/*/best_model.pth`.

## Notes for GitHub

Model files (`*.pth`) and raw dataset images can be large. They are ignored by default in `.gitignore`; use Git LFS or GitHub release assets if you need to publish checkpoints.
