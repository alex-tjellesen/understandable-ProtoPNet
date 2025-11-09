"""
Fidelity Test for ProtoPNet

This script measures how well the prototype activation vectors (explanations)
can reconstruct the preceding convolutional feature map from a trained PPNet model.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm


class Reconstructor(nn.Module):
    """
    A simple neural network that reconstructs the convolutional feature map
    from the prototype activation vector.
    """
    
    def __init__(self, num_prototypes, D, H, W):
        """
        Args:
            num_prototypes: Number of prototypes in the PPNet model
            D: Number of channels in the feature map
            H: Height of the feature map
            W: Width of the feature map
        """
        super(Reconstructor, self).__init__()
        self.num_prototypes = num_prototypes
        self.D = D
        self.H = H
        self.W = W
        self.output_dim = D * H * W
        
        # Simple linear layer followed by ReLU
        self.linear = nn.Linear(num_prototypes, self.output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_prototypes)
        
        Returns:
            Tensor of shape (batch_size, D, H, W)
        """
        # Pass through linear layer
        x = self.linear(x)
        x = self.activation(x)
        
        # Reshape to feature map dimensions
        x = x.view(-1, self.D, self.H, self.W)
        
        return x


def get_prototype_activations(ppnet, z_actual):
    """
    Compute prototype activation vector from feature map.
    
    Args:
        ppnet: The frozen PPNet model
        z_actual: Feature map tensor of shape (batch_size, D, H, W)
    
    Returns:
        Prototype activation vector of shape (batch_size, num_prototypes)
    """
    # Compute L2 distances
    distances = ppnet._l2_convolution(z_actual)
    
    # Global min pooling
    min_distances = -F.max_pool2d(-distances,
                                  kernel_size=(distances.size()[2],
                                             distances.size()[3]))
    min_distances = min_distances.view(-1, ppnet.num_prototypes)
    
    # Convert distances to similarities (prototype activations)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    
    return prototype_activations


def train_reconstructor(ppnet, reconstructor, train_loader, criterion, optimizer, 
                       num_epochs, device):
    """
    Train the reconstructor network to reconstruct feature maps from activation vectors.
    
    Args:
        ppnet: Frozen PPNet model
        reconstructor: Reconstructor network to train
        train_loader: DataLoader for training data
        criterion: Loss function (MSE)
        optimizer: Optimizer for reconstructor
        num_epochs: Number of training epochs
        device: Device to use (cpu or cuda)
    """
    reconstructor.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            # Get target feature map (Z_actual) from frozen PPNet
            with torch.no_grad():
                z_actual = ppnet.conv_features(images)
                # Get prototype activations (A_vec)
                prototype_activations = get_prototype_activations(ppnet, z_actual)
            
            # Get reconstruction (Z_recon) from Reconstructor
            z_recon = reconstructor(prototype_activations)
            
            # Calculate MSE loss
            loss = criterion(z_recon, z_actual)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}')


def evaluate_fidelity(ppnet, reconstructor, test_loader, criterion, device):
    """
    Evaluate the fidelity of the reconstructor on the test set.
    
    Args:
        ppnet: Frozen PPNet model
        reconstructor: Trained Reconstructor network
        test_loader: DataLoader for test data
        criterion: Loss function (MSE)
        device: Device to use (cpu or cuda)
    
    Returns:
        Average MSE loss (fidelity score)
    """
    reconstructor.eval()
    
    total_mse_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Evaluating Fidelity'):
            images = images.to(device)
            
            # Get target feature map (Z_actual)
            z_actual = ppnet.conv_features(images)
            
            # Get prototype activations (A_vec)
            prototype_activations = get_prototype_activations(ppnet, z_actual)
            
            # Get reconstruction (Z_recon)
            z_recon = reconstructor(prototype_activations)
            
            # Calculate MSE loss
            loss = criterion(z_recon, z_actual)
            total_mse_loss += loss.item()
            num_batches += 1
    
    avg_mse = total_mse_loss / num_batches
    return avg_mse


def run_fidelity_test(args):
    """
    Main function to run the fidelity test.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus != '-1' else 'cpu')
    if args.gpus != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    print(f'Using device: {device}')
    print(f'Loading model from: {args.model}')
    
    # Load the trained PPNet model
    ppnet = torch.load(args.model, map_location=device)
    ppnet = ppnet.to(device)
    
    # Freeze the PPNet model
    ppnet.eval()
    for param in ppnet.parameters():
        param.requires_grad = False
    
    print(f'Model loaded and frozen')
    print(f'Number of prototypes: {ppnet.num_prototypes}')
    
    # Get feature map dimensions
    # Create a dummy input to determine the output shape
    img_size = ppnet.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        dummy_features = ppnet.conv_features(dummy_input)
    
    D = dummy_features.shape[1]  # Number of channels
    H = dummy_features.shape[2]  # Height
    W = dummy_features.shape[3]  # Width
    
    print(f'Feature map shape: ({D}, {H}, {W})')
    
    # Create data loaders
    test_dir = os.path.join(args.dataset, 'test')
    
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=False)
    
    print(f'Test dataset size: {len(test_dataset)}')
    
    # If using test set for training, create a separate loader
    # Otherwise, use train set for training the reconstructor
    if args.use_train_for_reconstruction:
        train_dir = os.path.join(args.dataset, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False)
        print(f'Train dataset size: {len(train_dataset)}')
    else:
        # Use test set for training the reconstructor
        train_loader = test_loader
        print('Using test set for training the reconstructor')
    
    # Initialize the Reconstructor
    reconstructor = Reconstructor(ppnet.num_prototypes, D, H, W).to(device)
    print(f'Reconstructor initialized')
    print(f'Reconstructor parameters: {sum(p.numel() for p in reconstructor.parameters())}')
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=args.learning_rate)
    
    # Train the Reconstructor
    print(f'\nTraining Reconstructor for {args.num_epochs} epochs...')
    train_reconstructor(ppnet, reconstructor, train_loader, criterion, optimizer, 
                       args.num_epochs, device)
    
    # Evaluate Fidelity
    print(f'\nEvaluating Fidelity on test set...')
    fidelity_score = evaluate_fidelity(ppnet, reconstructor, test_loader, criterion, device)
    
    print(f'\n{"="*60}')
    print(f'FINAL FIDELITY SCORE (Average MSE): {fidelity_score:.6f}')
    print(f'{"="*60}')
    
    # Optionally save the results
    if args.save_results:
        results_file = os.path.join(args.out, 'fidelity_results.txt')
        os.makedirs(args.out, exist_ok=True)
        with open(results_file, 'w') as f:
            f.write(f'Model: {args.model}\n')
            f.write(f'Dataset: {args.dataset}\n')
            f.write(f'Number of prototypes: {ppnet.num_prototypes}\n')
            f.write(f'Feature map shape: ({D}, {H}, {W})\n')
            f.write(f'Training epochs: {args.num_epochs}\n')
            f.write(f'Batch size: {args.batch_size}\n')
            f.write(f'Learning rate: {args.learning_rate}\n')
            f.write(f'Fidelity Score (Average MSE): {fidelity_score:.6f}\n')
        print(f'\nResults saved to: {results_file}')
    
    return fidelity_score


def main():
    parser = argparse.ArgumentParser(
        description='Fidelity Test for ProtoPNet',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='path to the trained PPNet model (.pth file)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='path to the dataset directory (should contain train/ and test/ subdirectories)')
    
    # Optional arguments
    parser.add_argument('--gpus', type=str, default='0',
                       help='GPU devices to use, e.g., "0,1,2" or "-1" for CPU (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='batch size for data loading (default: %(default)s)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='number of workers for data loading (default: %(default)s)')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='number of epochs to train the reconstructor (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='learning rate for the reconstructor optimizer (default: %(default)s)')
    parser.add_argument('--use_train_for_reconstruction', action='store_true',
                       help='use training set to train the reconstructor (default: use test set)')
    parser.add_argument('--save_results', action='store_true',
                       help='save results to a text file')
    parser.add_argument('--out', type=str, default='fidelity_analysis',
                       help='output directory for saving results (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Run the fidelity test
    run_fidelity_test(args)


if __name__ == '__main__':
    main()
