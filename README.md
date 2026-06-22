# ML for Systems and Control — Unbalanced Disk Assignment

This repository contains the code, notebooks, trained artifacts, and submission files for the 5SC28 unbalanced disk assignment. The work is split into two main parts:

1. system identification / model estimation;
2. reinforcement-learning policy learning.

The most important files are listed below so that each assignment part can be found quickly.

## 1. System identification / model estimation

### NARX and LSTM models

Main location:

- `ANN_DELIVERABLES/5SC28_group1.ipynb`

This notebook contains the NARX, standard LSTM, and residual LSTM experiments, including preprocessing, model selection, validation/test evaluation, and final discussion.

Reproducible scripts:

- `ANN_DELIVERABLES/narx_selected_export.py`
- `ANN_DELIVERABLES/lstm_selected_export.py`

Saved models and metrics:

- `ANN_DELIVERABLES/selected_narx_best.pt`
- `ANN_DELIVERABLES/selected_narx_metrics.json`
- `ANN_DELIVERABLES/selected_lstm_best.pt`
- `ANN_DELIVERABLES/selected_lstm_metrics.json`

### NOE and Encoder RNN models

Main location:

- `Assignment_NOE_RNN.ipynb`

This notebook contains the neural output-error (NOE) model and the encoder RNN experiments.

### Gaussian Process model

Main location:

- `GaussianProcess.ipynb`

This notebook contains the sparse GP-NARX model, lag-order screening, kernel choices, inducing-point selection, validation/test evaluation, and submission-file generation.

Environment file for GP dependencies:

- `gpy_env.yaml`

## 2. Model-estimation submission `.npz` files

The final system-identification submission files are collected in:

- `submission_files/`

The professor can check the final prediction and simulation `.npz` files here:

| Model | One-step prediction file | Free-run simulation file |
|---|---|---|
| NARX | `submission_files/narx_prediction_submission.npz` | `submission_files/narx_simulation_submission.npz` |
| LSTM | `submission_files/lstm_prediction_submission.npz` | `submission_files/lstm_simulation_submission.npz` |
| NOE | `submission_files/NOE__prediction-submission.npz` | `submission_files/NOE_simulation-submission.npz` |
| Encoder RNN | `submission_files/EncoderRNN_prediction-submission.npz` | `submission_files/EncoderRNN_simulation-submission.npz` |
| Gaussian Process | `submission_files/gp-prediction-submission.npz` | `submission_files/gp-simulation-submission.npz` |

Additional ANN-specific copies/templates are also available in:

- `ANN_DELIVERABLES/`

The original assignment data and hidden-test templates are available in:

- `training-val-test-data.npz`
- `training-val-test-data.csv`
- `gym-unbalanced-disk-master/disc-benchmark-files/`

The hidden-test templates in `gym-unbalanced-disk-master/disc-benchmark-files/` show the required submission format. The final `.npz` submission files above are the files intended for checking.

## 3. Q-learning and DQN

Main location:

- `Q_Learning_Deliverable_Vanilla_and_DQN_V3.ipynb`

This notebook contains the tabular Q-learning and DQN experiments for swing-up and stabilization.

Saved Q-learning/DQN artifacts:

- `Q_Learning_Final_Artifacts/vanilla_q_results.npz`
- `Q_Learning_Final_Artifacts/dqn_v3_1500_results/dqn_v3_1500_results.npz`

## 4. Actor-Critic swing-up control

Main location:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/actor_critic_rl.ipynb`

This notebook contains the A2C swing-up controller, including the simple reward baseline, shaped reward policy, deterministic rollout evaluation, policy map, and robust/noisy policy used for the lab setup.

Supporting scripts:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/a2c_unbalanced_disk_train.py`
- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/a2c_policy_search.ipynb`

Saved A2C policies are stored mainly in:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/a2c_models/`

## 5. Actor-Critic reference tracking

Main location:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/actor_critic_rl_reference.ipynb`

This notebook compares A2C and PPO for reference tracking around the upright position. It includes learning curves, tracking-error tables, final achieved offset plots, voltage plots, and policy maps.

Supporting script:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/a2c_reference_tracking_train.py`

Policy-search / backup notebook:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/reference_tracking_policy_search.ipynb`

Saved reference-tracking policies are stored mainly in:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/ref_models/`
- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/reference_tracking_zip_models/`

## 6. Real hardware / lab experiment notebooks

The real-setup and calibration work is located in:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/omega_calibration_and_real_actor_critic.ipynb`
- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/real_reference_tracking.ipynb`

Deployment logs and real-setup checkpoints are stored in:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/omega_calibration_logs/`
- `gym-unbalanced-disk-master/gym_unbalanced_disk/examples-connect-to-exp/real_setup_checkpoints_omega_corrected/`

## 7. Gym environment and simulator code

The unbalanced disk environment is located in:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/`

Important files:

- `gym-unbalanced-disk-master/gym_unbalanced_disk/envs/UnbalancedDisk.py`
- `gym-unbalanced-disk-master/gym_unbalanced_disk/envs/UnbalancedDiskExp.py`
- `gym-unbalanced-disk-master/gym_unbalanced_disk/sim_src.py`

The benchmark data supplied with the assignment is located in:

- `gym-unbalanced-disk-master/disc-benchmark-files/`

## Notes

- The final model-estimation `.npz` files are in `submission_files/`.
- The ANN-specific deliverable folder has its own detailed README: `ANN_DELIVERABLES/README_DELIVERABLES.md`.
- No folders were renamed in this cleanup; this README is intended as the main navigation file for the repository.
