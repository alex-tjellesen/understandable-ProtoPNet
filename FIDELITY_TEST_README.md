# Fidelity Test for ProtoPNet

## Overview

The fidelity test measures how well the prototype activation vectors (explanations) can reconstruct the preceding convolutional feature map from a trained PPNet model. This provides a quantitative measure of how much information is preserved when the high-dimensional feature maps are compressed into the prototype activation vectors.

## Implementation

The implementation consists of:

1. **Reconstructor Network**: A simple neural network that learns to map prototype activation vectors back to the convolutional feature maps.
   - Input: Prototype activation vector (num_prototypes,)
   - Output: Reconstructed feature map (D, H, W)
   - Architecture: Linear layer + ReLU activation

2. **Training Process**: The reconstructor is trained to minimize MSE between:
   - **Target (Z_actual)**: The actual feature maps from PPNet's `conv_features`
   - **Reconstruction (Z_recon)**: The feature maps reconstructed from prototype activations

3. **Fidelity Score**: The final average MSE on the test set, measuring reconstruction quality.

## Usage

### Basic Usage

```bash
python fidelity_test.py \
  --model saved_models_backup/resnet34/as_is_run/300nopush78.08.pth \
  --dataset datasets/cub200
```

### Advanced Usage

```bash
python fidelity_test.py \
  --model saved_models_backup/resnet34/as_is_run/300nopush78.08.pth \
  --dataset datasets/cub200 \
  --num_epochs 15 \
  --batch_size 100 \
  --learning_rate 0.001 \
  --use_train_for_reconstruction \
  --save_results \
  --out fidelity_analysis \
  --gpus 0
```

### Arguments

- `--model` (required): Path to the trained PPNet model (.pth file)
- `--dataset` (required): Path to the dataset directory (must contain train/ and test/ subdirectories)
- `--gpus`: GPU devices to use (default: '0', use '-1' for CPU)
- `--batch_size`: Batch size for data loading (default: 100)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--num_epochs`: Number of epochs to train the reconstructor (default: 15)
- `--learning_rate`: Learning rate for the reconstructor optimizer (default: 0.001)
- `--use_train_for_reconstruction`: Use training set to train the reconstructor (default: use test set)
- `--save_results`: Save results to a text file
- `--out`: Output directory for saving results (default: 'fidelity_analysis')

## Methodology

### 1. Setup Phase
- Load the trained PPNet model and freeze all its parameters
- Set the model to evaluation mode
- Determine the feature map dimensions (D, H, W) by passing a dummy input

### 2. Training Phase
For each batch in the training set (or test set if `--use_train_for_reconstruction` is not set):
1. Pass images through frozen PPNet to get feature maps: `z_actual = ppnet.conv_features(x)`
2. Compute prototype activations from feature maps:
   - Calculate L2 distances: `distances = ppnet._l2_convolution(z_actual)`
   - Apply global min pooling: `min_distances = -max_pool2d(-distances, ...)`
   - Convert to similarities: `activations = ppnet.distance_2_similarity(min_distances)`
3. Reconstruct feature maps: `z_recon = reconstructor(activations)`
4. Compute MSE loss and backpropagate through the reconstructor only

### 3. Evaluation Phase
- Run the same process on the test set with `torch.no_grad()`
- Compute average MSE across all test batches
- Report the final fidelity score (lower is better)

## Interpreting Results

- **Lower MSE** = Higher fidelity = More information preserved in prototype activations
- **Higher MSE** = Lower fidelity = More information lost during compression

The fidelity score provides insight into:
- How well the prototype-based explanation captures the original feature representations
- The trade-off between interpretability (using a small number of prototypes) and information preservation
- Whether the prototype activation layer is a bottleneck in the model's information flow

## Example Output

```
Using device: cpu
Loading model from: saved_models_backup/resnet34/as_is_run/300nopush78.08.pth
Model loaded and frozen
Number of prototypes: 2000
Feature map shape: (512, 7, 7)
Test dataset size: 5794
Using test set for training the reconstructor
Reconstructor initialized
Reconstructor parameters: 50176000

Training Reconstructor for 10 epochs...
Epoch 1/10: 100%|████████████| 116/116 [00:45<00:00, 2.55it/s, loss=0.123456]
Epoch 1/10 - Average Loss: 0.145678
...

Evaluating Fidelity on test set...
100%|████████████████████████| 116/116 [00:20<00:00, 5.67it/s]

============================================================
FINAL FIDELITY SCORE (Average MSE): 0.098765
============================================================

Results saved to: fidelity_analysis/fidelity_results.txt
```

## Notes

- By default, the test set is used for both training the reconstructor and evaluating fidelity
- Use `--use_train_for_reconstruction` to train on the train set and evaluate on test set
- The reconstructor is intentionally simple to focus on measuring information content
- GPU usage is recommended for faster training on large datasets
