# Code for "Resource-efficient integration of multi-modal clinical data via a divide-and-conquer paradigm"

This repository provides the MIMIC-IV-MM implementation of PPO-based collaboration
between large and small models for multimodal clinical prediction. The model
combines image, text, and time-series representations through a seven-expert
mixture-of-experts network, with PPO learning the routing policy and a
distillation reward connecting large- and small-model features.

## Repository structure

```text
train.py                         Training and validation
test.py                          Test evaluation
common.py                        Shared data and model interfaces
models/PPO_MultiModal.py          Multimodal fusion network
models/SimplifiedPPO_MoE_Agent.py  PPO trainer and reward computation
utils/                           Loss functions and logging
dataset/MIMIC_IV_MM_npy_dataset.py  Feature dataset
requirements.txt                 Python dependencies
```

## Installation

Use Python 3.9 or newer and install PyTorch for your CPU or CUDA environment.

```bash
python -m pip install -r requirements.txt
```

Run the following commands from the repository root.

## Data preparation

Prepare split-level NPY arrays for image, text, time-series, and small-model
features. Set `--large_data_path` to the large-model feature directory and
`--small_data_path` to the small-model feature directory.

The dataset follows the original MIMIC-IV-MM file layout. Large-model features
are read from `<large_data_path>/<task>/<image_model>_<text_model>_<time_model>/`
as `<split>_img.npy`, `<split>_text.npy`, and `<split>_time.npy`; labels are read
from `<large_data_path>/<task>/<split>_label.npy`. Small-model features and labels
are read from `<small_data_path>/<task>/<split>_feat.npy` and
`<small_data_path>/<task>/<split>_target.npy`.

All arrays for a split share the same sample order. The dataset checks that
large- and small-model labels match. See [dataset/README.md](dataset/README.md)
for the full directory structure and array shapes.

| Representation | Model | Dimension |
| --- | --- | --- |
| Image | dinov2 | 1536 |
| Image | rad-dino | 768 |
| Text | MeLLaMA-13B | 5120 |
| Text | BioMistral-7B | 4960 |
| Time series | moirai-1.1-R-large | 1024 |
| Small model | Joint multimodal features | 1536 |

Training uses a task-specific small-model checkpoint to initialize the
`Linear(1536, 2)` classification head. Supported checkpoint keys are
`cls_head.weight` / `cls_head.bias`, `large_classifier.weight` /
`large_classifier.bias`, or a head state dictionary containing `weight` and `bias`.

## Training

```bash
python train.py \
  --task readmission \
  --large_data_path /path/to/large_embed \
  --small_data_path /path/to/MultimodalPred_input \
  --small_model_path ./checkpoints/readmission_small.pth \
  --save_path ./outputs/readmission \
  --seed 42
```

Supported tasks are `mortality`, `longstay`, and `readmission`. Select the feature
extractors with `--large_image_model_name`, `--large_text_model_name`, and
`--large_time_model_name`. Supply the corresponding feature files and small-model
checkpoint for the selected task.

Training uses inverse-frequency sampling, task-weighted cross entropy, PPO
updates, and cosine learning-rate scheduling. The default settings are:

| Parameter | Default |
| --- | --- |
| Epochs | 100 |
| Batch size | 64 |
| Task learning rate | 1e-4 |
| PPO learning rate | 3e-4 |
| Hidden dimension | 512 |
| Distillation weight | 0.2 |
| Reward type | exp |
| Inference gate temperature | 0.5 |

The training entry point reads the training and validation splits. The model
with the highest validation AUROC is saved as `best_model.pth`, together with
its configuration, epoch, validation score, and positive-label ratio.
`config.json` records the run settings, and `history.json` records epoch metrics.
Use a separate `--save_path` for each run.

## Evaluation

```bash
python test.py \
  --checkpoint ./outputs/readmission/best_model.pth \
  --large_data_path /path/to/large_embed \
  --small_data_path /path/to/MultimodalPred_input \
  --output ./outputs/readmission/test_metrics.json
```

Evaluation restores the model configuration and weights from the checkpoint.
Use `--large_data_path` and `--small_data_path` to set data locations on the
evaluation machine.

The default `--threshold_mode ratio` determines a threshold from validation
predictions using the positive-label ratio recorded during training.
`--threshold_mode argmax` selects the class with the highest probability and
loads only the test split.

The result JSON contains AUROC, average precision, trapezoidal AUPRC, the
classification report, and mean expert probabilities. Select the execution
device with `--device auto`, `--device cpu`, or `--device cuda`.
