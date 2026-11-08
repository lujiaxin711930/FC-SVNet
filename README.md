# FC-SVNet: Lithofacies Identification from Well-Log Images

Research source code for the FC-SVNet model for lithofacies identification in deep tight sandstone using spatial morphological and vertical sequence features with facies-sequence constraints.

## Repository Structure

```
FC-SVNet/
├── FC-SVNet_train.py          # Model training
├── FC-SVNet_test.py           # Model evaluation and Viterbi decoding
├── ECA.py                     # Efficient Channel Attention module
├── ECAs.py                    # Enhanced channel attention module
├── MGC.py                     # Multi-scale Group Convolution module
├── SP.py                      # Strip Pooling module
├── SMGC.py                    # Strip Pooling and Multi-scale Group Convolution
├── resnet50.py                # ResNet50 backbone implementation
├── preprocessing/            # Well-log image generation and preprocessing
│   ├── compute_curve_ranges.py
│   ├── plot_well_log_images.py
│   ├── crop_images.py
│   ├── resize_images.py
│   ├── merge_images.py
│   └── RUN_ORDER.md
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Environment

The environment used Ubuntu 22.04.5 LTS, Python 3.12.13, an NVIDIA Tesla P100 (16 GB), an Intel Xeon CPU @ 2.00 GHz (four reported cores), PyTorch 2.3.1+cu121, and CUDA 12.1.

### Installation

Create a Python 3.12 environment, then install the matching CUDA build of PyTorch.

## Data directory structure

The scripts expect a `No_deep_Data_hua` folder in the working directory with numeric class subdirectories:

```text
No_deep_Data_hua/
  train/
    0/ ... 6/
  val/
    0/ ... 6/
  test/
    0/ ... 6/
```

**Data availability:** Data will be made available on request.

## Run

```bash
python FC-SVNet_train.py
python FC-SVNet_test.py
```

The training code writes models and checkpoints to `outputs/`, and the testing code reads `outputs/best_model_FC-SVNet.pth` or `outputs/best_model_params.pth`. Testing therefore requires a trained weight file. Do not claim that testing is runnable without those files. Only load checkpoint files from trusted sources: model deserialization can execute arbitrary code.

## Notes for reproducibility

- Training uses an AdamW optimizer, batch size 64, a nominal 100-epoch loop, and progressive backbone freezing. Inspect the training script for exact hyperparameters.

## Lithofacies label mapping

The class indices used by the preprocessing code and model are:

| Class index | Code | Lithofacies |
|---:|---|---|
| 0 | St | Weakly dissolved trough cross-bedded fine sandstone |
| 1 | Sa | Strongly dissolved tabular cross-bedded fine sandstone |
| 2 | Sp | Weakly dissolved planar-laminated fine sandstone |
| 3 | Sw | Strongly cemented wavy-laminated fine sandstone |
| 4 | Fw | Strongly cemented wavy-laminated argillaceous siltstone |
| 5 | Fh | Strongly cemented horizontally-laminated siltstone |
| 6 | O | Other mudstone lithofacies |

The Excel-based preprocessing scripts require `openpyxl` for `.xlsx` input (its original version is unknown). The original Excel sheets use the column header `岩相` for lithofacies labels. The plotting script maps `St: 0`, `Sa: 1`, `Sp: 2`, `Sw: 3`, `Fw: 4`, `Fh: 5`, and `O: 6`.

## Well-log image generation

The `preprocessing/` directory contains the original five-stage image-generation workflow, with English-language comments and portable relative paths. See [`preprocessing/RUN_ORDER.md`](preprocessing/RUN_ORDER.md) for step-by-step commands and constraints. The stages are:

1. `compute_curve_ranges.py` — inspect minima and maxima for the six logging curves.
2. `plot_well_log_images.py` — generate lithofacies-labeled images from six-sample well-log windows.
3. `crop_images.py` — crop the generated images.
4. `resize_images.py` — resize within each well/class folder.
5. `merge_images.py` — merge images into class-specific folders.

For reproducibility, the original fixed curve offsets, the crop rectangle, and window/filename conventions are preserved in code. Some choices are source-data-specific and may need adjustment for a different authorized dataset.
