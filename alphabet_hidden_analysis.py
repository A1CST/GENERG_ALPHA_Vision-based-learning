# ================================================================
# Alphabet Hidden Layer Activation Analysis
# ================================================================
# Visualizes the hidden layer activations from a trained alphabet
# genome, showing how the network represents different letters
# internally.
#
# Usage:
#   python alphabet_hidden_analysis.py                    # Interactive selection
#   python alphabet_hidden_analysis.py path/to/genome.pkl # Specific genome
#
# Output:
#   visualizations/alphabet_hidden_analysis_genXXXXX.png
# ================================================================

import os
import sys
import pickle
import random
import argparse
import numpy as np
from pathlib import Path

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


# ================================================================
# GENOME LOADING
# ================================================================
def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_alphabet_genomes():
    """List all alphabet genomes in best_genomes folder."""
    genome_dir = "best_genomes"
    if not os.path.exists(genome_dir):
        return []

    genomes = []
    for f in os.listdir(genome_dir):
        if f.endswith('.pkl') and 'alphabet' in f.lower():
            path = os.path.join(genome_dir, f)
            size = os.path.getsize(path)

            # Extract generation number
            gen_num = "unknown"
            parts = f.replace('.pkl', '').split('_')
            for part in parts:
                if part.startswith('gen') and len(part) > 3:
                    gen_num = part[3:]

            genomes.append({
                'path': path,
                'filename': f,
                'size': size,
                'generation': gen_num
            })

    # Sort by modification time (newest first)
    genomes.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
    return genomes


def select_genome():
    """Interactive genome selection."""
    genomes = list_alphabet_genomes()

    if not genomes:
        print("No alphabet genomes found in best_genomes/")
        print("Run extract_best_genome.py first to extract a trained genome.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SELECT ALPHABET GENOME FOR ANALYSIS")
    print("=" * 60)
    print("\nAvailable alphabet genomes:\n")

    for i, g in enumerate(genomes[:10]):
        size_str = format_size(g['size'])
        print(f"  [{i+1}] {g['filename']}")
        print(f"       Gen {g['generation']}, {size_str}")

    print(f"\n  [Enter] Use most recent ({genomes[0]['filename']})")

    choice = input("\nYour choice: ").strip()

    if choice == "":
        return genomes[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(genomes):
        return genomes[int(choice) - 1]
    else:
        print("Invalid choice, using most recent...")
        return genomes[0]


def load_genome(genome_path):
    """Load genome data from file."""
    print(f"\nLoading genome: {genome_path}")

    with open(genome_path, 'rb') as f:
        data = pickle.load(f)

    # Extract controller weights
    controller = data['controller']
    w1 = np.array(controller['w1'])
    b1 = np.array(controller['b1'])
    w2 = np.array(controller['w2'])
    b2 = np.array(controller['b2'])

    print(f"  Network: {controller['input_size']} -> {controller['hidden_size']} -> {controller['output_size']}")
    print(f"  Trust: {data['genome']['trust']:.2f}")
    print(f"  Genome ID: {data['genome']['id']}")

    gen = 'unknown'
    if 'extraction_info' in data:
        gen = data['extraction_info'].get('source_generation', 'unknown')

    return {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'input_size': controller['input_size'],
        'hidden_size': controller['hidden_size'],
        'output_size': controller['output_size'],
        'genome_id': data['genome']['id'],
        'trust': data['genome']['trust'],
        'generation': gen
    }


# ================================================================
# LETTER RENDERING
# ================================================================
class LetterRenderer:
    """Render letters for activation analysis."""

    def __init__(self):
        self.width = ALPHABET_FIELD_WIDTH
        self.height = ALPHABET_FIELD_HEIGHT

        # Load fonts
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]

        self.font = None
        self.font_small = None
        for path in font_paths:
            if os.path.exists(path):
                self.font = ImageFont.truetype(path, ALPHABET_FONT_SIZE)
                self.font_small = ImageFont.truetype(path, max(8, ALPHABET_FONT_SIZE * 12 // 64))
                break

        if self.font is None:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def render(self, letter, variation=0):
        """
        Render a letter with variation.

        Returns: flattened grayscale pixel array
        """
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
    """
    Generate sample images for each letter.

    Returns:
        samples: list of (flattened_pixels, letter_idx, letter)
    """
    print(f"\nGenerating {26 * variations_per_letter} letter samples...")

    samples = []
    for letter_idx, letter in enumerate(LETTERS):
        for var in range(variations_per_letter):
            pixels = renderer.render(letter, variation=var)
            samples.append((pixels, letter_idx, letter))

    print(f"Generated {len(samples)} samples ({variations_per_letter} per letter)")
    return samples


# ================================================================
# HIDDEN ACTIVATION EXTRACTION
# ================================================================
def compute_hidden_activations(samples, controller):
    """
    Compute hidden layer activations for all samples.

    hidden = tanh(input @ w1.T + b1)

    Returns:
        activations: numpy array of shape (n_samples, hidden_size)
        letter_indices: list of letter indices (0-25)
        letters: list of letters (A-Z)
    """
    w1 = controller['w1']
    b1 = controller['b1']

    activations = []
    letter_indices = []
    letters_list = []

    print(f"\nComputing hidden activations for {len(samples)} samples...")

    for i, (pixels, letter_idx, letter) in enumerate(samples):
        # Forward pass through first layer
        x = np.array(pixels)
        hidden = np.tanh(w1 @ x + b1)

        activations.append(hidden)
        letter_indices.append(letter_idx)
        letters_list.append(letter)

    activations = np.array(activations)
    print(f"Activation matrix shape: {activations.shape}")

    return activations, letter_indices, letters_list


# ================================================================
# OUTPUT LAYER ANALYSIS
# ================================================================
def compute_output_weights_analysis(controller):
    """
    Analyze the output layer weights to see how hidden neurons
    contribute to each letter prediction.
    """
    w2 = controller['w2']  # Shape: (26, hidden_size)

    # Each row of w2 is the weights for one output letter
    return w2


# ================================================================
# VISUALIZATION
# ================================================================
def create_visualization(activations, letter_indices, letters_list, controller, output_path):
    """
    Create 4-panel visualization of hidden activations.
    """
    n_samples, hidden_size = activations.shape
    n_letters = 26

    # Sort by letter for visualization
    sorted_indices = np.argsort(letter_indices)
    sorted_activations = activations[sorted_indices]
    sorted_letter_indices = np.array(letter_indices)[sorted_indices]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Main title with genome info
    gen_num = controller['generation']
    trust = controller['trust']
    fig.suptitle(f"Alphabet Hidden Layer Analysis\n"
                 f"Generation {gen_num} | Trust: {trust:.2f} | "
                 f"{n_samples} samples, 26 letters, {hidden_size} hidden neurons",
                 fontsize=14, fontweight='bold')

    # ========================================
    # TOP LEFT: Hidden Activations by Letter
    # ========================================
    ax1 = axes[0, 0]

    im1 = ax1.imshow(sorted_activations, aspect='auto', cmap='RdBu_r',
                     vmin=-1, vmax=1)
    ax1.set_xlabel('Hidden Neuron Index')
    ax1.set_ylabel('Sample (sorted by letter)')
    ax1.set_title('Hidden Activations Overview', fontsize=12, fontweight='bold')

    # Add letter boundaries
    samples_per_letter = n_samples // n_letters
    for i in range(1, n_letters):
        ax1.axhline(y=i * samples_per_letter - 0.5, color='black', linewidth=0.5, alpha=0.3)

    # Add letter labels on y-axis
    label_positions = [i * samples_per_letter + samples_per_letter // 2 for i in range(n_letters)]
    ax1.set_yticks(label_positions[::2])  # Every other letter
    ax1.set_yticklabels(LETTERS[::2], fontsize=8)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Activation (tanh)')

    # ========================================
    # TOP RIGHT: Mean Activation by Letter
    # ========================================
    ax2 = axes[0, 1]

    # Compute mean activation per letter per neuron
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

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Mean Activation')

    # ========================================
    # BOTTOM LEFT: t-SNE Projection
    # ========================================
    ax3 = axes[1, 0]

    print("\nComputing t-SNE projection...")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_result = tsne.fit_transform(activations)

    # Create color map for 26 letters
    colors = plt.cm.rainbow(np.array(letter_indices) / 25)

    scatter = ax3.scatter(tsne_result[:, 0], tsne_result[:, 1],
                          c=letter_indices, cmap='rainbow',
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.3)

    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_title('t-SNE Projection of Hidden Activations', fontsize=12, fontweight='bold')

    # Add letter annotations for cluster centers
    for letter_idx, letter in enumerate(LETTERS):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 0:
            center_x = np.mean(tsne_result[mask, 0])
            center_y = np.mean(tsne_result[mask, 1])
            ax3.annotate(letter, (center_x, center_y), fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    cbar3 = plt.colorbar(scatter, ax=ax3, ticks=range(0, 26, 5))
    cbar3.set_label('Letter Index')
    cbar3.ax.set_yticklabels(['A', 'F', 'K', 'P', 'U', 'Z'])

    # ========================================
    # BOTTOM RIGHT: Neuron Utilization + Output Weights
    # ========================================
    ax4 = axes[1, 1]

    # Compute mean absolute activation per neuron
    mean_abs_per_neuron = np.mean(np.abs(activations), axis=0)
    std_per_neuron = np.std(activations, axis=0)

    neuron_indices = np.arange(hidden_size)

    # Plot bars
    bars = ax4.bar(neuron_indices, mean_abs_per_neuron,
                   color='steelblue', alpha=0.7, label='Mean |Activation|')
    ax4.errorbar(neuron_indices, mean_abs_per_neuron, yerr=std_per_neuron,
                 fmt='none', color='darkblue', alpha=0.5, capsize=1)

    # Add horizontal line at average
    avg_utilization = np.mean(mean_abs_per_neuron)
    ax4.axhline(y=avg_utilization, color='red', linestyle='--',
                alpha=0.7, label=f'Average ({avg_utilization:.3f})')

    ax4.set_xlabel('Hidden Neuron Index')
    ax4.set_ylabel('Mean |Activation|')
    ax4.set_title('Hidden Neuron Utilization', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.set_xlim(-1, hidden_size)

    # Summary stats
    active_neurons = np.sum(mean_abs_per_neuron > 0.1)
    very_active = np.sum(mean_abs_per_neuron > 0.3)
    ax4.text(0.02, 0.95, f'Active neurons (>0.1): {active_neurons}/{hidden_size}\n'
                         f'Very active (>0.3): {very_active}/{hidden_size}',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================================
    # FINALIZE
    # ========================================
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    plt.close(fig)

    return fig


def create_confusion_heatmap(activations, letter_indices, controller, output_path):
    """
    Create a letter confusion analysis based on hidden representations.
    Shows which letters have similar hidden representations.
    """
    n_letters = 26

    # Compute mean activation per letter
    mean_by_letter = np.zeros((n_letters, controller['hidden_size']))
    for letter_idx in range(n_letters):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 0:
            mean_by_letter[letter_idx] = np.mean(activations[mask], axis=0)

    # Compute pairwise cosine similarity between letters
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(mean_by_letter)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1)

    ax.set_xticks(range(26))
    ax.set_yticks(range(26))
    ax.set_xticklabels(LETTERS, fontsize=9)
    ax.set_yticklabels(LETTERS, fontsize=9)

    ax.set_xlabel('Letter')
    ax.set_ylabel('Letter')
    ax.set_title(f"Letter Similarity in Hidden Space\n"
                 f"(Cosine similarity of mean activations)",
                 fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')

    # Add values to cells (for high/low similarities only)
    for i in range(26):
        for j in range(26):
            if i != j:
                val = similarity_matrix[i, j]
                if abs(val) > 0.7:  # Only show high correlations
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=6, color='black' if abs(val) < 0.9 else 'white')

    plt.tight_layout()

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
    parser = argparse.ArgumentParser(description="Alphabet Hidden Layer Analysis")
    parser.add_argument("genome", nargs="?", help="Path to genome .pkl file")
    parser.add_argument("--variations", type=int, default=10,
                        help="Variations per letter (default: 10)")
    parser.add_argument("--output", "-o", help="Output path for visualization")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ALPHABET HIDDEN LAYER ACTIVATION ANALYSIS")
    print("=" * 60)
    print("\nThis script visualizes how the neural network's hidden layer")
    print("represents different letters internally.")

    # Load genome
    if args.genome:
        if not os.path.exists(args.genome):
            print(f"Error: File not found: {args.genome}")
            sys.exit(1)
        genome_path = args.genome
    else:
        genome_info = select_genome()
        genome_path = genome_info['path']

    controller = load_genome(genome_path)

    # Generate letter samples
    renderer = LetterRenderer()
    samples = generate_letter_samples(renderer, variations_per_letter=args.variations)

    # Compute hidden activations
    activations, letter_indices, letters_list = compute_hidden_activations(samples, controller)

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        gen_num = controller['generation']
        os.makedirs("visualizations", exist_ok=True)
        output_path = f"visualizations/alphabet_hidden_analysis_gen{gen_num}.png"

    # Create main visualization
    create_visualization(activations, letter_indices, letters_list, controller, output_path)

    # Create similarity matrix
    similarity_matrix = create_confusion_heatmap(activations, letter_indices, controller, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\nSummary Statistics:")
    print(f"  Total samples: {len(activations)}")
    print(f"  Letters: 26")
    print(f"  Hidden dimensions: {controller['hidden_size']}")

    # Compute letter separability
    mean_per_letter = np.zeros((26, controller['hidden_size']))
    for letter_idx in range(26):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 0:
            mean_per_letter[letter_idx] = np.mean(activations[mask], axis=0)

    # Between-letter variance
    between_var = np.var(mean_per_letter, axis=0).mean()
    # Within-letter variance
    within_var = 0
    for letter_idx in range(26):
        mask = np.array(letter_indices) == letter_idx
        if np.sum(mask) > 1:
            within_var += np.var(activations[mask], axis=0).mean()
    within_var /= 26

    print(f"\n  Between-letter variance: {between_var:.4f}")
    print(f"  Within-letter variance:  {within_var:.4f}")
    print(f"  Separability ratio:      {between_var / max(within_var, 1e-8):.4f}")

    # Find most confused letter pairs
    print(f"\nMost Similar Letter Pairs (potential confusions):")
    pairs = []
    for i in range(26):
        for j in range(i + 1, 26):
            pairs.append((LETTERS[i], LETTERS[j], similarity_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for l1, l2, sim in pairs[:10]:
        print(f"  {l1} - {l2}: {sim:.3f}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
