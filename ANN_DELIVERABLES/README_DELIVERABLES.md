# ANN Modelling Deliverables

## Main Notebook

### `5SC28_group1.ipynb`

Main notebook for the ANN modelling work.

Contains:

- dataset loading and preprocessing;
- chronological train/validation/test split;
- NARX model explanation and results;
- LSTM model explanation and results;
- tuning tables used in the report;
- final comparison between NARX, standard LSTM, and residual LSTM;
- notes on how the final submission files are generated.

The notebook is the main readable story of the experiments.

## Final Python Scripts

### `narx_selected_export.py`

Final reproducible script for the selected NARX model.

It:

- trains the selected NARX model;
- evaluates validation and held-out test performance;
- saves the selected NARX checkpoint and metrics;
- creates NARX prediction and simulation `.npz` submission files.

Run:

```bash
python3 narx_selected_export.py
```

### `lstm_selected_export.py`

Final reproducible script for the selected standard LSTM model.

It:

- trains the selected standard LSTM model;
- evaluates validation and held-out test performance;
- saves the selected LSTM checkpoint and metrics;
- creates LSTM prediction and simulation `.npz` submission files.

Run:

```bash
python3 lstm_selected_export.py --device mps
```

If `mps` is not available:

```bash
python3 lstm_selected_export.py --device cpu
```

## Final Output Files

### NARX

- `selected_narx_best.pt`: selected NARX checkpoint.
- `selected_narx_metrics.json`: NARX validation and held-out test metrics.
- `narx_prediction_submission.npz`: NARX one-step prediction submission.
- `narx_simulation_submission.npz`: NARX simulation submission.

### LSTM

- `selected_lstm_best.pt`: selected standard LSTM checkpoint.
- `selected_lstm_metrics.json`: LSTM validation and held-out test metrics.
- `lstm_prediction_submission.npz`: LSTM one-step prediction submission.
- `lstm_simulation_submission.npz`: LSTM simulation submission.

## Required Provided Files

These files are needed to run the scripts:

- `training-val-test-data.npz`
- `hidden-test-prediction-submission-file.npz`
- `hidden-test-simulation-submission-file.npz`
- `submission-file-checker.py`

The hidden submission templates are only used to create files with the correct submission format. They are not used for training, validation, test evaluation, or hyperparameter selection.

## Checking Submission Format

```bash
python3 submission-file-checker.py narx_prediction_submission.npz hidden-test-prediction-submission-file.npz
python3 submission-file-checker.py narx_simulation_submission.npz hidden-test-simulation-submission-file.npz

python3 submission-file-checker.py lstm_prediction_submission.npz hidden-test-prediction-submission-file.npz
python3 submission-file-checker.py lstm_simulation_submission.npz hidden-test-simulation-submission-file.npz
```

The checker only confirms that the file format is valid. The printed RMSE is not the real hidden-test score because the hidden target values are not provided.

## Files Not Included In The Final Delivery

The old tuning scripts and logs are not needed in the final delivery package. Their results are summarized in the notebook and report tables.

Examples of files not needed unless specifically requested:

- `hp_tuning.py`
- `hp_tuning_seq.py`
- `hp_tuning_res.py`
- `hp_tuning_res_advanced.py`
- `hp_tuning_ultimate.py`
- `test_lstm.py`
- `test_lstm_opt.py`
- `test_res_lstm.py`
- `best_lstm_results.txt`
- `seq_len_results.txt`
- `res_lstm_tuning.txt`
- `res_lstm_advanced_tuning.txt`
- `ultimate_results.txt`

## Minimal Final Package

- `README_DELIVERABLES.md`
- `5SC28_group1.ipynb`
- `narx_selected_export.py`
- `lstm_selected_export.py`
- `selected_narx_best.pt`
- `selected_narx_metrics.json`
- `selected_lstm_best.pt`
- `selected_lstm_metrics.json`
- `narx_prediction_submission.npz`
- `narx_simulation_submission.npz`
- `lstm_prediction_submission.npz`
- `lstm_simulation_submission.npz`
- required provided `.npz` files and `submission-file-checker.py`, if the code must be runnable from the delivered folder.
