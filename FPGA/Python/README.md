# Python Model Training

This folder contains Python scripts for preparing data, training and fine-tuning the CNN-LSTM model, and exporting reference data for HLS verification.

## Contents

| File | Description |
|---|---|
| `model.py` | Defines the CNN-LSTM model architecture. |
| `dataset.py` | Loads and prepares data for model training and evaluation. |
| `extract_dataset_sliding.py` | Extracts sliding-window samples from the source dataset. |
| `extract_own_data.py` | Prepares personally collected sensor recordings. |
| `train.py` | Trains the model. |
| `finetune.py` | Fine-tunes a pretrained model using personal sensor data. |
| `sweep.py` | Runs experiments with different training settings. |
| `export_golden.py` | Exports reference inputs and outputs for checking the HLS implementation. |
| `best_model_enabl3s.pt` | Saved model checkpoint from training on ENABL3S. |
| `best_model.pt` | Saved model checkpoint for the final model. |

## Workflow

1. Prepare the training data using the extraction scripts.
2. Train the model using `train.py`.
3. Fine-tune it on personal recordings using `finetune.py`.
4. Use `export_golden.py` to generate reference data for HLS verification.

Update dataset and checkpoint paths to match your local setup before running the scripts.

The HLS implementation is documented in [`../hls/`](../hls/), and the FPGA design is documented in [`../vivado/`](../vivado/).
