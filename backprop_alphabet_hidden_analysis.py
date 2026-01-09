# ================================================================
# MLP Backprop Hidden Layer Activation Analysis
# ================================================================
# Visualizes the hidden layer activations from a trained PyTorch MLP,
# showing how the network represents different letters internally.
#
# Equivalent to: alphabet_hidden_analysis.py (for EA)
#
# Usage:
#   python backprop_hidden_analysis.py                  # Auto-load mlp_model.pth
#   python backprop_hidden_analysis.py path/to/model.pth
#
# Output:
#   visualizations/backprop_hidden_analysis_[timestamp].png
# ================================================================

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ================================================================
# CONFIGURATION
# ================================================================
ALPHABET_FIELD_WIDTH = 100
ALPHABET_FIELD_HEIGHT = 100
ALPHABET_FONT_SIZE = 64
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# ================================================================
# MODEL DEFINITION (Must match training)
# ================================================================
class MLP(nn.Module):
    def __init__(self, input_size=10000, hidden_size=32, output_size=26):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

# ================================================================
# MODEL LOADING
# ================================================================
def load_backprop_model(model_path):
    """
    Load PyTorch model and extract weights as numpy arrays.
    Returns dictionary matching the structure expected by analysis tools.
    """
    print(f"\nLoading Backprop Model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # Initialize model structure
    model = MLP(input_size=ALPHABET_FIELD_WIDTH * ALPHABET_FIELD_HEIGHT)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        sys.exit(1)

    model.eval()

    # Extract weights to NumPy for analysis
    # PyTorch Linear weights are stored as (out_features, in_features)
    # We leave them as is, because we will do (W @ x + b)
    w1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    w2 = model.fc2.weight.detach().numpy()
    b2 = model.fc2.bias.detach().numpy()

    print(f"  Network: {w1.shape[1]} -> {w1.shape[0]} -> {w2.shape[0]}")
    
    return {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'input_size': w1.shape[1],
        'hidden_size': w1.shape[0],
        'output_size': w2.shape[0],
        'generation': "Backprop",
        'trust': 0.0, # Not applicable to backprop
        'id': 'mlp_backprop'
    }

# ================================================================
# LETTER RENDERING (Matches EA Analysis exactly)
# ================================================================
class LetterRenderer:
    """Render letters for activation analysis."""

    def __init__(self):
        self.width = ALPHABET_FIELD_WIDTH
        self.height = ALPHABET_FIELD_HEIGHT

        # Load fonts
        local_font = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
        font_paths = [
            local_font,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:\\Windows\\Fonts\\DejaVuSans.ttf",
        ]

        self.font = None
        self.font_small = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, ALPHABET_FONT_SIZE)
                    self.font_small = ImageFont.truetype(path, max(8, ALPHABET_FONT_SIZE * 12 // 64))
                    break
                except:
                    continue

        if self.font is None:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def render(self, letter, variation=0):
        """Render a letter with variation."""
        width, height = self.width, self.height

        # Variation styles
        base_style = variation % 4
        use_small = base_style < 2
        invert = (base_style % 2) == 1

        font = self.font_small if use_small else self.font

        if invert:
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)
        else:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)

        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), letter.upper(), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2 - bbox[0]
        y = (height - text_height) // 2 - bbox[1]

        # Add slight jitter for variations > 3
        if variation >= 4:
            random.seed(variation * 1000 + ord(letter))
            x += random.randint(-8, 8)
            y += random.randint(-8, 8)

        draw.text((x, y), letter.upper(), font=font, fill=text_color)

        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32) / 255.0
        return arr.flatten()

def generate_letter_samples(renderer, variations_per_letter=10):
    """Generate sample images for each letter."""
    print(f"\nGenerating {26 * variations_per_letter} letter samples...")
    samples = []
    for letter_idx, letter in enumerate(LETTERS):
        for var in range(variations_per_letter):
            pixels = renderer.render(letter, variation=var)
            samples.append((pixels, letter_idx, letter))
    print(f"Generated {len(samples)} samples.")
    return samples

# ================================================================
# HIDDEN ACTIVATION EXTRACTION
# ================================================================
def compute_hidden_activations(samples, controller):
    """
    Compute hidden layer activations for all samples.
    Using numpy to be exact match with EA script math.
    
    hidden = tanh(w1 @ input + b1)
    """
    w1 = controller['w1'] # Shape (32, 10000)
    b1 = controller['b1'] # Shape (32,)

    activations = []
    letter_indices = []
    letters_list = []

    print(f"\nComputing hidden activations for {len(samples)} samples...")

    for i, (pixels, letter_idx, letter) in enumerate(samples):
        # Forward pass through first layer
        # PyTorch Linear uses x @ W.T + b, but since we extracted 
        # w1 as (out, in), we do w1 @ x + b
        x = np.array(pixels)
        
        # Matrix multiplication: (32, 10000) @ (10000,) -> (32,)
        z = w1 @ x + b1 
        hidden = np.tanh(z)

        activations.append(hidden)
        letter_indices.append(letter_idx)
        letters_list.append(letter)

    activations = np.array(activations)
    print(f"Activation matrix shape: {activations.shape}")

    return activations, letter_indices, letters_list

# ================================================================
# VISUALIZATION (Identical to EA Script)
# ================================================================
def create_visualization(activations, letter_indices, letters_list, controller, output_path):
    """Create 4-panel visualization of hidden activations."""
    n_samples, hidden_size = activations.shape
    n_letters = 26

    # Sort by letter for visualization
    sorted_indices = np.argsort(letter_indices)
    sorted_activations = activations[sorted_indices]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Main title
    fig.suptitle(f"Backprop Hidden Layer Analysis\n"
                 f"Model: {controller['generation']} | "
                 f"{n_samples} samples, 26 letters, {hidden_size} hidden neurons",
                 fontsize=14, fontweight='bold')

    # 1. Hidden Activations by Letter
    ax1 = axes[0, 0]
    im1 = ax1.imshow(sorted_activations, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Hidden Neuron Index')
    ax1.set_ylabel('Sample (sorted by letter)')
    ax1.set_title('Hidden Activations Overview', fontsize=12, fontweight='bold')
    
    # Add letter labels on y-axis
    samples_per_letter = n_samples // n_letters
    label_positions = [i * samples_per_letter + samples_per_letter // 2 for i in range(n_letters)]
    ax1.set_yticks(label_positions[::2])
    ax1.set_yticklabels(LETTERS[::2], fontsize=8)
    plt.colorbar(im1, ax=ax1, label='Activation (tanh)')

    # 2. Mean Activation by Letter
    ax2 = axes[0, 1]
    mean_by_letter = np.zeros((n_letters, hidden_size))
    for letter_idx in range(n_letters):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 0:
            mean_by_letter[letter_idx] = np.mean(activations[mask], axis=0)

    im2 = ax2.imshow(mean_by_letter, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xlabel('Hidden Neuron Index')
    ax2.set_ylabel('Letter')
    ax2.set_title('Mean Activation by Letter', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(26))
    ax2.set_yticklabels(LETTERS, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Mean Activation')

    # 3. t-SNE Projection
    ax3 = axes[1, 0]
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_result = tsne.fit_transform(activations)
    
    colors = plt.cm.rainbow(np.array(letter_indices) / 25)
    scatter = ax3.scatter(tsne_result[:, 0], tsne_result[:, 1],
                          c=letter_indices, cmap='rainbow',
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
    ax3.set_title('t-SNE Projection', fontsize=12, fontweight='bold')

    # Add letter annotations
    for letter_idx, letter in enumerate(LETTERS):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 0:
            center_x = np.mean(tsne_result[mask, 0])
            center_y = np.mean(tsne_result[mask, 1])
            ax3.annotate(letter, (center_x, center_y), fontsize=8, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # 4. Neuron Utilization
    ax4 = axes[1, 1]
    mean_abs_per_neuron = np.mean(np.abs(activations), axis=0)
    std_per_neuron = np.std(activations, axis=0)
    neuron_indices = np.arange(hidden_size)

    ax4.bar(neuron_indices, mean_abs_per_neuron, color='steelblue', alpha=0.7, label='Mean |Activation|')
    ax4.errorbar(neuron_indices, mean_abs_per_neuron, yerr=std_per_neuron, fmt='none', color='darkblue', alpha=0.5)
    
    avg_utilization = np.mean(mean_abs_per_neuron)
    ax4.axhline(y=avg_utilization, color='red', linestyle='--', alpha=0.7, label=f'Avg ({avg_utilization:.3f})')
    
    ax4.set_xlabel('Hidden Neuron Index')
    ax4.set_ylabel('Mean |Activation|')
    ax4.set_title('Hidden Neuron Utilization', fontsize=12, fontweight='bold')
    ax4.legend()

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close(fig)

def create_confusion_heatmap(activations, letter_indices, controller, output_path):
    """Create letter similarity analysis."""
    n_letters = 26
    mean_by_letter = np.zeros((n_letters, controller['hidden_size']))
    for letter_idx in range(n_letters):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 0:
            mean_by_letter[letter_idx] = np.mean(activations[mask], axis=0)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(mean_by_letter)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    
    ax.set_xticks(range(26))
    ax.set_yticks(range(26))
    ax.set_xticklabels(LETTERS, fontsize=9)
    ax.set_yticklabels(LETTERS, fontsize=9)
    ax.set_title("Letter Similarity in Hidden Space (Backprop)", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Save
    similarity_path = output_path.replace('.png', '_similarity.png')
    plt.savefig(similarity_path, dpi=150, bbox_inches='tight')
    print(f"Saved similarity matrix to: {similarity_path}")
    plt.close(fig)
    return similarity_matrix

# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Backprop Hidden Layer Analysis")
    parser.add_argument("model", nargs="?", default="mlp_model.pth", help="Path to mlp_model.pth")
    parser.add_argument("--variations", type=int, default=10, help="Variations per letter")
    parser.add_argument("--output", "-o", help="Output path")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BACKPROP HIDDEN LAYER ANALYSIS")
    print("=" * 60)

    # Load controller
    controller = load_backprop_model(args.model)

    # Generate samples
    renderer = LetterRenderer()
    samples = generate_letter_samples(renderer, variations_per_letter=args.variations)

    # Compute activations
    activations, letter_indices, letters_list = compute_hidden_activations(samples, controller)

    # Output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("visualizations", exist_ok=True)
        output_path = f"visualizations/backprop_hidden_analysis_{timestamp}.png"

    # Create visualizations
    create_visualization(activations, letter_indices, letters_list, controller, output_path)
    similarity_matrix = create_confusion_heatmap(activations, letter_indices, controller, output_path)

    # Stats
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY (BACKPROP)")
    print("=" * 60)
    
    mean_abs_act = np.mean(np.abs(activations))
    print(f"Average Activation Magnitude: {mean_abs_act:.4f}")
    
    # Find similarity pairs
    pairs = []
    for i in range(26):
        for j in range(i + 1, 26):
            pairs.append((LETTERS[i], LETTERS[j], similarity_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nMost Similar Representations:")
    for l1, l2, sim in pairs[:5]:
        print(f"  {l1} - {l2}: {sim:.3f}")

if __name__ == "__main__":
    main()