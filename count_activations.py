"""
Script to count the number of activated prototypes for each sample in a test set.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from ppnet.preprocess import mean, std


def count_activated_prototypes(model, dataloader, activation_threshold=0.0, device='cpu', log=print):
    """
    Count the number of prototypes that provide positive evidence for the predicted class
    and analyze the distribution/concentration of evidence in top prototypes.
    
    A prototype is considered 'activated' or 'relevant' if it contributes positively to the 
    predicted class's logit. This is determined by:
    1. Computing prototype activations (similarities)
    2. Getting the predicted class
    3. Extracting the weight vector for the predicted class from the final layer
    4. Computing element-wise product: evidence = activations * class_weights
    5. Counting prototypes where evidence > threshold
    6. Analyzing what % of total evidence comes from top-1, top-3, top-5 prototypes
    
    Args:
        model: Trained ProtoPNet model
        dataloader: DataLoader for the test set
        activation_threshold: Minimum evidence score to consider a prototype as "activated"
        device: Device to run the model on
        log: Logging function
    
    Returns:
        results: Dictionary containing:
            - activation_counts: List of number of activated prototypes per sample
            - max_similarities: List of max similarity values per sample
            - min_distances: List of min distance values per sample
            - max_evidence: List of max evidence scores per sample
            - top_1_pct: List of % of evidence from top-1 prototype per sample
            - top_3_pct: List of % of evidence from top-3 prototypes per sample
            - top_5_pct: List of % of evidence from top-5 prototypes per sample
            - predictions: List of predicted class labels
            - true_labels: List of true class labels
            - sample_indices: List of sample indices
    """
    model.eval()
    model.to(device)
    
    activation_counts = []
    max_similarities_per_sample = []
    min_distances_per_sample = []
    max_evidence_per_sample = []
    top_1_pct_list = []
    top_3_pct_list = []
    top_5_pct_list = []
    predictions = []
    true_labels = []
    sample_indices = []
    
    sample_idx = 0
    n_correct = 0
    
    log('Counting activated prototypes (evidence-based)...')
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing samples"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model output and prototype distances
            logits, min_distances = model(images)
            
            # Get predictions
            _, predicted = torch.max(logits.data, 1)
            n_correct += (predicted == labels).sum().item()
            
            # Convert distances to similarities using the model's activation function
            similarities = model.module.distance_2_similarity(min_distances)
            
            # Get the weights from the final fully connected layer
            # Shape: (num_classes, num_prototypes)
            fc_weights = model.module.last_layer.weight.data
            
            # Process each sample in the batch
            for i in range(images.size(0)):
                sim = similarities[i]  # Shape: (num_prototypes,)
                min_dist = min_distances[i]  # Shape: (num_prototypes,)
                pred_class = predicted[i].item()
                
                # Get the weight vector for the predicted class
                # This tells us how much each prototype contributes to this class
                class_weights = fc_weights[pred_class]  # Shape: (num_prototypes,)
                
                # Compute evidence: element-wise multiplication
                # Positive values = prototype provides positive evidence for predicted class
                evidence = sim * class_weights  # Shape: (num_prototypes,)
                
                # Count prototypes with positive evidence (contribution to predicted class)
                activated = (evidence > activation_threshold).sum().item()
                activation_counts.append(activated)
                
                # Calculate total positive evidence
                positive_evidence = evidence[evidence > 0]
                total_evidence = positive_evidence.sum().item()
                
                # Sort evidence in descending order
                sorted_evidence = torch.sort(evidence, descending=True).values
                
                # Calculate cumulative evidence for top-k prototypes
                if total_evidence > 0:
                    top_1_evidence = sorted_evidence[0].item()
                    top_3_evidence = sorted_evidence[0:3].sum().item()
                    top_5_evidence = sorted_evidence[0:5].sum().item()
                    
                    # Calculate percentages
                    top_1_pct = (top_1_evidence / total_evidence) * 100
                    top_3_pct = (top_3_evidence / total_evidence) * 100
                    top_5_pct = (top_5_evidence / total_evidence) * 100
                else:
                    # Edge case: no positive evidence (shouldn't happen but handle it)
                    top_1_pct = 0.0
                    top_3_pct = 0.0
                    top_5_pct = 0.0
                
                top_1_pct_list.append(top_1_pct)
                top_3_pct_list.append(top_3_pct)
                top_5_pct_list.append(top_5_pct)
                
                # Store max similarity, min distance, and max evidence
                max_similarities_per_sample.append(sim.max().item())
                min_distances_per_sample.append(min_dist.min().item())
                max_evidence_per_sample.append(evidence.max().item())
                
                # Store predictions and labels
                predictions.append(pred_class)
                true_labels.append(labels[i].item())
                sample_indices.append(sample_idx)
                
                sample_idx += 1
    
    accuracy = n_correct / sample_idx * 100
    log(f'Test accuracy: {accuracy:.2f}%')
    
    results = {
        'activation_counts': activation_counts,
        'max_similarities': max_similarities_per_sample,
        'min_distances': min_distances_per_sample,
        'max_evidence': max_evidence_per_sample,
        'top_1_pct': top_1_pct_list,
        'top_3_pct': top_3_pct_list,
        'top_5_pct': top_5_pct_list,
        'predictions': predictions,
        'true_labels': true_labels,
        'sample_indices': sample_indices,
        'accuracy': accuracy
    }
    
    return results


def print_statistics(results, log=print):
    """Print statistics about prototype activations (evidence-based)."""
    counts = np.array(results['activation_counts'])
    max_sims = np.array(results['max_similarities'])
    min_dists = np.array(results['min_distances'])
    max_evid = np.array(results['max_evidence'])
    top_1_pct = np.array(results['top_1_pct'])
    top_3_pct = np.array(results['top_3_pct'])
    top_5_pct = np.array(results['top_5_pct'])
    correct = np.array(results['predictions']) == np.array(results['true_labels'])
    
    log(f"\n{'='*60}")
    log(f"Prototype Activation Statistics (Evidence-Based)")
    log(f"{'='*60}")
    log(f"Total test samples: {len(counts)}")
    log(f"Test accuracy: {results['accuracy']:.2f}%")
    
    log(f"\n--- Activated Prototypes per Sample (Positive Evidence) ---")
    log(f"Mean: {np.mean(counts):.2f} ± {np.std(counts):.2f}")
    log(f"Median: {np.median(counts):.0f}")
    log(f"Min: {np.min(counts)}, Max: {np.max(counts)}")
    
    log(f"\n--- Evidence Concentration in Top Prototypes ---")
    log(f"Top-1 Prototype (% of total evidence):")
    log(f"  Mean: {np.mean(top_1_pct):.2f}% ± {np.std(top_1_pct):.2f}%")
    log(f"  Median: {np.median(top_1_pct):.2f}%")
    log(f"  Min: {np.min(top_1_pct):.2f}%, Max: {np.max(top_1_pct):.2f}%")
    
    log(f"\nTop-3 Prototypes (% of total evidence):")
    log(f"  Mean: {np.mean(top_3_pct):.2f}% ± {np.std(top_3_pct):.2f}%")
    log(f"  Median: {np.median(top_3_pct):.2f}%")
    log(f"  Min: {np.min(top_3_pct):.2f}%, Max: {np.max(top_3_pct):.2f}%")
    
    log(f"\nTop-5 Prototypes (% of total evidence):")
    log(f"  Mean: {np.mean(top_5_pct):.2f}% ± {np.std(top_5_pct):.2f}%")
    log(f"  Median: {np.median(top_5_pct):.2f}%")
    log(f"  Min: {np.min(top_5_pct):.2f}%, Max: {np.max(top_5_pct):.2f}%")
    
    log(f"\n--- Max Similarity per Sample ---")
    log(f"Mean: {np.mean(max_sims):.4f} ± {np.std(max_sims):.4f}")
    log(f"Median: {np.median(max_sims):.4f}")
    log(f"Min: {np.min(max_sims):.4f}, Max: {np.max(max_sims):.4f}")
    
    log(f"\n--- Min Distance per Sample ---")
    log(f"Mean: {np.mean(min_dists):.4f} ± {np.std(min_dists):.4f}")
    log(f"Median: {np.median(min_dists):.4f}")
    log(f"Min: {np.min(min_dists):.4f}, Max: {np.max(min_dists):.4f}")
    
    log(f"\n--- Max Evidence per Sample ---")
    log(f"Mean: {np.mean(max_evid):.4f} ± {np.std(max_evid):.4f}")
    log(f"Median: {np.median(max_evid):.4f}")
    log(f"Min: {np.min(max_evid):.4f}, Max: {np.max(max_evid):.4f}")
    
    # Statistics for correctly vs incorrectly classified samples
    log(f"\n--- Correct vs Incorrect Classifications ---")
    log(f"Correctly classified samples: {np.sum(correct)}")
    log(f"  Mean activated prototypes: {np.mean(counts[correct]):.2f}")
    log(f"  Mean top-1 evidence %: {np.mean(top_1_pct[correct]):.2f}%")
    log(f"  Mean top-3 evidence %: {np.mean(top_3_pct[correct]):.2f}%")
    log(f"  Mean top-5 evidence %: {np.mean(top_5_pct[correct]):.2f}%")
    log(f"  Mean max similarity: {np.mean(max_sims[correct]):.4f}")
    log(f"  Mean max evidence: {np.mean(max_evid[correct]):.4f}")
    log(f"Incorrectly classified samples: {np.sum(~correct)}")
    if np.sum(~correct) > 0:
        log(f"  Mean activated prototypes: {np.mean(counts[~correct]):.2f}")
        log(f"  Mean top-1 evidence %: {np.mean(top_1_pct[~correct]):.2f}%")
        log(f"  Mean top-3 evidence %: {np.mean(top_3_pct[~correct]):.2f}%")
        log(f"  Mean top-5 evidence %: {np.mean(top_5_pct[~correct]):.2f}%")
        log(f"  Mean max similarity: {np.mean(max_sims[~correct]):.4f}")
        log(f"  Mean max evidence: {np.mean(max_evid[~correct]):.4f}")
    log(f"{'='*60}\n")


def save_results(results, output_dir, log=print):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, 'activation_counts.npy'), results['activation_counts'])
    np.save(os.path.join(output_dir, 'max_similarities.npy'), results['max_similarities'])
    np.save(os.path.join(output_dir, 'min_distances.npy'), results['min_distances'])
    np.save(os.path.join(output_dir, 'max_evidence.npy'), results['max_evidence'])
    np.save(os.path.join(output_dir, 'top_1_pct.npy'), results['top_1_pct'])
    np.save(os.path.join(output_dir, 'top_3_pct.npy'), results['top_3_pct'])
    np.save(os.path.join(output_dir, 'top_5_pct.npy'), results['top_5_pct'])
    np.save(os.path.join(output_dir, 'predictions.npy'), results['predictions'])
    np.save(os.path.join(output_dir, 'true_labels.npy'), results['true_labels'])
    
    # Save statistics as text file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        def file_log(msg):
            f.write(msg + '\n')
            print(msg)
        print_statistics(results, log=file_log)
    
    log(f"\nResults saved to {output_dir}")


def run_activation_counting(args):
    """Main function to run prototype activation counting."""
    # Load model
    print(f'Loading model from {args.model}')
    
    # Determine device and load model accordingly
    use_cuda = torch.cuda.is_available() and args.gpus
    device = 'cuda' if use_cuda else 'cpu'
    
    if use_cuda:
        ppnet = torch.load(args.model)
        ppnet = ppnet.cuda()
    else:
        ppnet = torch.load(args.model, map_location='cpu')
        ppnet = ppnet.cpu()
    
    ppnet_multi = torch.nn.DataParallel(ppnet)
    
    # Load test dataset
    print(f'Loading dataset from {args.dataset}')
    normalize = T.Normalize(mean=mean, std=std)
    test_dataset = datasets.ImageFolder(
        args.dataset,
        T.Compose([
            T.Resize(size=(ppnet.img_size, ppnet.img_size)),
            T.ToTensor(),
            normalize,
        ])
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    print(f'Test set size: {len(test_dataset)}')
    print(f'Number of classes: {len(test_dataset.classes)}')
    print(f'Number of prototypes: {ppnet.num_prototypes}')
    
    # Count activations
    results = count_activated_prototypes(
        model=ppnet_multi,
        dataloader=test_loader,
        activation_threshold=args.threshold,
        device=device,
        log=print
    )
    
    # Print statistics
    print_statistics(results, log=print)
    
    # Save results
    save_results(results, args.out, log=print)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count activated prototypes for each sample in test set',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42)
    )
    parser.add_argument('--model', type=str, required=True, 
                        help='path to trained model')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='path to test dataset (should be a directory with class subdirectories)')
    parser.add_argument('--threshold', type=float, default=0.0, 
                        help='activation threshold for counting prototypes (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size for processing (default: %(default)s)')
    parser.add_argument('--gpus', type=str, default='0', 
                        help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='number of workers for data loading (default: %(default)s)')
    parser.add_argument('--out', '-o', type=str, default='activation_analysis', 
                        help='output directory for saving results (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Run activation counting
    results = run_activation_counting(args)
