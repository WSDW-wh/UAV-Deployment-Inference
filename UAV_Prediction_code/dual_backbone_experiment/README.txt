This folder contains a self-contained dual-input experiment setup.

Files:
- data_load_ablation.py         paired CSV loader
- train_utils.py               dual-input trainer
- models_dual_backbone.py      plain CNN / ResNet18 + optional attention
- train_dual_backbone.py       training entry script

Recommended runs:
1) Plain CNN + attention
python train_dual_backbone.py --backbone plaincnn --use_attention --save_dir runs_dual_backbone/plaincnn_attn_dual

2) Plain CNN without attention
python train_dual_backbone.py --backbone plaincnn --save_dir runs_dual_backbone/plaincnn_dual

3) ResNet18 + attention
python train_dual_backbone.py --backbone resnet18 --use_attention --save_dir runs_dual_backbone/resnet18_attn_dual

4) ResNet18 without attention
python train_dual_backbone.py --backbone resnet18 --save_dir runs_dual_backbone/resnet18_dual

Notes:
- Default is NOT pretrained.
- If you want pretrained ResNet18, add: --pretrained
- The dual branch is kept: model takes inputs1, teacher takes inputs2.
- Feature alignment uses cosine loss on pooled feature vectors.
