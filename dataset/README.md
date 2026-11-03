# MIMIC-IV-MM data loading

`MIMIC_IV_MM_npy_dataset.py` loads split-level NPY arrays using the layout in the
original MIMIC-IV-MM dataset implementation. Large- and small-model features
are stored in separate directories, selected with `--large_data_path` and
`--small_data_path`.

## Directory structure

For task `readmission` and model combination
`dinov2_MeLLaMA-13B_moirai-1.1-R-large`:

```text
large_embed/
└── readmission/
    ├── train_label.npy
    ├── val_label.npy
    ├── test_label.npy
    └── dinov2_MeLLaMA-13B_moirai-1.1-R-large/
        ├── train_img.npy
        ├── train_text.npy
        ├── train_time.npy
        ├── val_img.npy
        ├── val_text.npy
        ├── val_time.npy
        ├── test_img.npy
        ├── test_text.npy
        └── test_time.npy

MultimodalPred_input/
└── readmission/
    ├── train_feat.npy
    ├── train_target.npy
    ├── val_feat.npy
    ├── val_target.npy
    ├── test_feat.npy
    └── test_target.npy
```

Replace `readmission` with `mortality` or `longstay` for the other tasks. The
model subdirectory is formed by joining the image, text, and time-series model
names with underscores.

## Files loaded for each split

| Array | Location | Shape |
| --- | --- | --- |
| Image features | `<large_data_path>/<task>/<models>/<split>_img.npy` | `[N, image_dim]` |
| Text features | `<large_data_path>/<task>/<models>/<split>_text.npy` | `[N, text_dim]` |
| Time-series features | `<large_data_path>/<task>/<models>/<split>_time.npy` | `[N, 1024]` |
| Large-model labels | `<large_data_path>/<task>/<split>_label.npy` | `[N]` |
| Small-model features | `<small_data_path>/<task>/<split>_feat.npy` | `[N, 1536]` |
| Small-model labels | `<small_data_path>/<task>/<split>_target.npy` | `[N]` |

Here `<models>` is `<image_model_name>_<text_model_name>_<timeseries_model_name>`,
and `<split>` is `train`, `val`, or `test`. The default image and text dimensions
are 1536 and 5120. `rad-dino` uses image dimension 768; `BioMistral-7B` uses text
dimension 4960.

All arrays within a split share the same sample order and sample count. The
loader compares the two label arrays element by element. Labels are binary 0/1.
Feature tensors are float32; label tensors are int64. Samples retain their
stored order and split assignment.

## Usage

```bash
python train.py \
  --task readmission \
  --large_data_path /path/to/large_embed \
  --small_data_path /path/to/MultimodalPred_input \
  --small_model_path ./checkpoints/readmission_small.pth \
  --save_path ./outputs/readmission
```

The data paths are saved in the model checkpoint. Set either argument during
evaluation to use another data location:

```bash
python test.py \
  --checkpoint ./outputs/readmission/best_model.pth \
  --large_data_path /path/to/large_embed \
  --small_data_path /path/to/MultimodalPred_input
```

Paths are relative to the working directory unless absolute paths are supplied.
The default roots are `./data/large_embed` and `./data/MultimodalPred_input`.

## Python interface

```python
dataset = NPY_Large_MultimodalPred_Dataset(
    split='train',
    task='readmission',
    image_model_name='dinov2',
    text_model_name='MeLLaMA-13B',
    timeseries_model_name='moirai-1.1-R-large',
    large_data_path='/path/to/large_embed',
    small_data_path='/path/to/MultimodalPred_input',
)
```

Each sample returns `(large_image, large_text, large_time, small_features, label)`.
The `.label` tensor supplies labels to the training sampler.
