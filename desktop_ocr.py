# ================================================================
# Desktop OCR Pipeline
# ================================================================
# Uses trained genreg alphabet genomes to perform OCR on desktop
# screenshots using connected components segmentation.
#
# Usage:
#   python desktop_ocr.py                        # Interactive genome selection
#   python desktop_ocr.py --genome PATH          # Specific genome
#   python desktop_ocr.py --auto                 # Auto-select newest genome
#   python desktop_ocr.py --image PATH           # Single image
#   python desktop_ocr.py --folder PATH          # Custom folder
#
# Output:
#   {image_name}_ocr.txt  - extracted text
#   {image_name}_ocr.json - detailed data with timing metrics
# ================================================================

import os
import sys
import json
import pickle
import time
import argparse
from datetime import datetime
from glob import glob

import numpy as np
from PIL import Image
import cv2

# ================================================================
# CONFIGURATION
# ================================================================
CHAR_WIDTH = 100
CHAR_HEIGHT = 100
CONFIDENCE_THRESHOLD = 0.3  # Skip characters below this confidence

# Component filtering
MIN_AREA = 20
MAX_AREA = 50000
MIN_HEIGHT = 8
MAX_HEIGHT = 300
MIN_ASPECT = 0.1
MAX_ASPECT = 10.0

# Line detection
LINE_TOLERANCE = 0.5  # fraction of median height
WORD_GAP_THRESHOLD = 1.5  # fraction of median char width


# ================================================================
# NEURAL NETWORK (from alphabet_inference_headless.py)
# ================================================================
class SimpleNetwork:
    """Minimal forward-pass-only neural network for inference."""

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Try to use PyTorch if available
        try:
            import torch
            self._use_torch = True
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except ImportError:
            self._use_torch = False
            self.device = None

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def load_weights(self, ctrl_data):
        """Load weights from checkpoint data."""
        if self._use_torch:
            import torch
            self.w1 = torch.tensor(ctrl_data['w1'], dtype=torch.float32, device=self.device)
            self.b1 = torch.tensor(ctrl_data['b1'], dtype=torch.float32, device=self.device)
            self.w2 = torch.tensor(ctrl_data['w2'], dtype=torch.float32, device=self.device)
            self.b2 = torch.tensor(ctrl_data['b2'], dtype=torch.float32, device=self.device)
        else:
            self.w1 = np.array(ctrl_data['w1'])
            self.b1 = np.array(ctrl_data['b1'])
            self.w2 = np.array(ctrl_data['w2'])
            self.b2 = np.array(ctrl_data['b2'])

    def forward_visual(self, visual_input):
        """Forward pass returning character probabilities."""
        if self._use_torch:
            return self._forward_torch(visual_input)
        else:
            return self._forward_python(visual_input)

    def _forward_torch(self, visual_input):
        """GPU-accelerated forward pass."""
        import torch
        if not isinstance(visual_input, torch.Tensor):
            x = torch.tensor(visual_input, dtype=torch.float32, device=self.device)
        else:
            x = visual_input.to(self.device)

        hidden = torch.tanh(self.w1 @ x + self.b1)
        outputs = self.w2 @ hidden + self.b2

        char_logits = outputs[:26] if len(outputs) >= 26 else outputs
        char_probs = torch.softmax(char_logits, dim=0)

        return char_probs.cpu().tolist()

    def _forward_python(self, visual_input):
        """Pure Python/NumPy forward pass."""
        hidden = np.tanh(self.w1 @ visual_input + self.b1)
        outputs = self.w2 @ hidden + self.b2

        char_logits = outputs[:26]
        exp_logits = np.exp(char_logits - np.max(char_logits))
        char_probs = exp_logits / np.sum(exp_logits)

        return char_probs.tolist()

    def predict(self, visual_input):
        """Get the predicted letter (argmax)."""
        char_probs = self.forward_visual(visual_input)
        max_idx = np.argmax(char_probs[:26])
        return chr(ord('A') + max_idx), char_probs[max_idx]

    def predict_top_k(self, visual_input, k=5):
        """Get top-k predictions with probabilities."""
        char_probs = self.forward_visual(visual_input)
        indices = np.argsort(char_probs)[::-1][:k]
        return [(chr(ord('A') + i), char_probs[i]) for i in indices]


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


def list_available_genomes(genome_dir="best_genomes"):
    """List all alphabet genomes in best_genomes folder."""
    genomes = []

    if not os.path.exists(genome_dir):
        return []

    for filename in os.listdir(genome_dir):
        if filename.endswith('.pkl') and 'alphabet' in filename.lower():
            filepath = os.path.join(genome_dir, filename)
            file_size = os.path.getsize(filepath)

            # Extract generation from filename
            gen_num = "?"
            try:
                if "_gen" in filename:
                    gen_part = filename.split("_gen")[1]
                    gen_num = gen_part.split("_")[0]
            except:
                pass

            genomes.append({
                'path': filepath,
                'filename': filename,
                'size': file_size,
                'generation': gen_num
            })

    genomes.sort(key=lambda x: x['filename'], reverse=True)
    return genomes


def select_genome(genomes=None, auto_select=False):
    """Interactive genome selection or auto-select newest."""
    if genomes is None:
        genomes = list_available_genomes()

    if not genomes:
        print("No alphabet genomes found in best_genomes/")
        print("Extract one first with: python extract_best_genome.py")
        sys.exit(1)

    if auto_select:
        print(f"[AUTO] Selected: {genomes[0]['filename']}")
        return genomes[0]

    print("\n" + "=" * 60)
    print("SELECT GENOME FOR OCR")
    print("=" * 60)
    print("\nAvailable alphabet genomes:\n")

    for i, g in enumerate(genomes[:10]):
        size_str = format_size(g['size'])
        print(f"  [{i+1}] {g['filename']}")
        print(f"       Gen {g['generation']}, {size_str}")

    print(f"\n  [Enter] Use most recent")

    choice = input("\nYour choice: ").strip()

    if choice == "":
        return genomes[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(genomes):
        return genomes[int(choice) - 1]
    else:
        print("Invalid choice, using most recent...")
        return genomes[0]


def load_genome(genome_path):
    """Load a genome from pickle file."""
    print(f"Loading genome: {genome_path}")

    with open(genome_path, 'rb') as f:
        data = pickle.load(f)

    ctrl_data = data['controller']
    input_size = ctrl_data['input_size']
    hidden_size = ctrl_data['hidden_size']
    output_size = ctrl_data['output_size']

    print(f"  Network: {input_size} -> {hidden_size} -> {output_size}")

    network = SimpleNetwork(input_size, hidden_size, output_size)
    network.load_weights(ctrl_data)

    trust = data.get('genome', {}).get('trust', 0)
    print(f"  Trust: {trust:.2f}")

    genome_info = {
        'path': genome_path,
        'generation': data.get('extraction_info', {}).get('source_generation', '?'),
        'trust': trust,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size
    }

    return network, genome_info


# ================================================================
# IMAGE SEGMENTATION (OpenCV Connected Components)
# ================================================================
def preprocess_for_segmentation(image):
    """
    Convert image to binary for connected components.
    Auto-detects dark/light theme and adjusts accordingly.

    Returns: (binary_image, is_inverted)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect if dark theme (mean brightness < 128)
    mean_brightness = np.mean(gray)
    is_dark_theme = mean_brightness < 128

    # Apply Otsu's threshold
    if is_dark_theme:
        # Dark background, light text - threshold normally
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Light background, dark text - threshold and invert
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary, is_dark_theme


def find_connected_components(binary_image):
    """
    Find all connected components using OpenCV.

    Returns: list of component dicts with bbox, area, centroid
    """
    # Find connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    components = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        components.append({
            'label': i,
            'bbox': (x, y, x + w, y + h),
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'area': area,
            'centroid': (cx, cy)
        })

    return components


def filter_components(components):
    """
    Filter components to keep only likely character candidates.
    """
    filtered = []

    for comp in components:
        # Size filter
        if comp['area'] < MIN_AREA or comp['area'] > MAX_AREA:
            continue

        # Height filter
        if comp['height'] < MIN_HEIGHT or comp['height'] > MAX_HEIGHT:
            continue

        # Aspect ratio filter
        aspect = comp['width'] / max(1, comp['height'])
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        filtered.append(comp)

    return filtered


def sort_into_lines(components):
    """
    Group components into text lines and sort in reading order.

    Returns: list of lines, each line is list of components
    """
    if not components:
        return []

    # Calculate median height
    heights = [c['height'] for c in components]
    median_height = np.median(heights) if heights else 20

    # Sort by Y centroid
    sorted_comps = sorted(components, key=lambda c: c['centroid'][1])

    # Group into lines
    lines = []
    current_line = []
    current_y = None
    tolerance = median_height * LINE_TOLERANCE

    for comp in sorted_comps:
        cy = comp['centroid'][1]

        if current_y is None or abs(cy - current_y) <= tolerance:
            current_line.append(comp)
            if current_y is None:
                current_y = cy
            else:
                # Update current_y to average
                current_y = np.mean([c['centroid'][1] for c in current_line])
        else:
            # Start new line
            if current_line:
                lines.append(current_line)
            current_line = [comp]
            current_y = cy

    if current_line:
        lines.append(current_line)

    # Sort each line left to right
    for line in lines:
        line.sort(key=lambda c: c['centroid'][0])

    return lines


# ================================================================
# CHARACTER PREPROCESSING
# ================================================================
def preprocess_character(image, bbox, is_dark_theme):
    """
    Extract and preprocess a character for inference.

    Returns: flattened normalized array (10000,)
    """
    x1, y1, x2, y2 = bbox

    # Add padding
    padding = 2
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    # Crop
    if len(image.shape) == 3:
        crop = image[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_crop = image[y1:y2, x1:x2]

    # Pad to square
    h, w = gray_crop.shape
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)

    # Center the crop
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = gray_crop

    # Resize to target size
    resized = cv2.resize(square, (CHAR_WIDTH, CHAR_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    # Ensure white-on-black for inference
    # Model was trained on white text on black background
    mean_val = np.mean(resized)
    if mean_val > 128:
        # Black on white, need to invert
        resized = 255 - resized

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized.flatten()


# ================================================================
# TEXT ASSEMBLY
# ================================================================
def assemble_text(lines, predictions):
    """
    Assemble extracted text from lines and predictions.

    Returns: multi-line string
    """
    text_lines = []

    for line in lines:
        if not line:
            continue

        # Calculate median width for word gap detection
        widths = [c['width'] for c in line]
        median_width = np.median(widths) if widths else 10
        gap_threshold = median_width * WORD_GAP_THRESHOLD

        line_text = ""
        prev_x2 = None

        for comp in line:
            label = comp['label']
            pred_info = predictions.get(label)

            if pred_info is None:
                continue

            # Check for word gap
            if prev_x2 is not None:
                gap = comp['x'] - prev_x2
                if gap > gap_threshold:
                    line_text += " "

            line_text += pred_info['letter']
            prev_x2 = comp['bbox'][2]

        if line_text.strip():
            text_lines.append(line_text)

    return "\n".join(text_lines)


# ================================================================
# MAIN OCR PIPELINE
# ================================================================
def process_single_image(image_path, network, genome_info, verbose=True):
    """
    Full OCR pipeline for one image.

    Returns: results dict
    """
    timing = {}
    start_total = time.time()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'=' * 60}")

    # Load image
    start = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not load image: {image_path}")
        return None
    timing['load_ms'] = (time.time() - start) * 1000

    height, width = image.shape[:2]
    if verbose:
        print(f"  Image size: {width} x {height}")

    # Segmentation
    start = time.time()
    binary, is_dark_theme = preprocess_for_segmentation(image)
    components = find_connected_components(binary)
    filtered = filter_components(components)
    lines = sort_into_lines(filtered)
    timing['segmentation_ms'] = (time.time() - start) * 1000

    total_components = len(components)
    filtered_count = len(filtered)
    total_chars = sum(len(line) for line in lines)

    if verbose:
        print(f"  Theme: {'dark' if is_dark_theme else 'light'}")
        print(f"  Components: {total_components} found, {filtered_count} filtered")
        print(f"  Lines detected: {len(lines)}")

    # Inference
    start = time.time()
    predictions = {}
    confidences = []

    for line in lines:
        for comp in line:
            obs = preprocess_character(image, comp['bbox'], is_dark_theme)
            letter, confidence = network.predict(obs)
            top_5 = network.predict_top_k(obs, k=5)

            # Skip low confidence (likely non-letter)
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            predictions[comp['label']] = {
                'letter': letter,
                'confidence': confidence,
                'top_5': top_5,
                'bbox': comp['bbox']
            }
            confidences.append(confidence)

    timing['inference_ms'] = (time.time() - start) * 1000

    if verbose:
        chars_kept = len(predictions)
        print(f"  Characters recognized: {chars_kept}")
        if chars_kept > 0:
            print(f"  Inference time: {timing['inference_ms']:.1f}ms ({timing['inference_ms']/chars_kept:.2f}ms/char)")

    # Assemble text
    extracted_text = assemble_text(lines, predictions)

    timing['total_ms'] = (time.time() - start_total) * 1000

    # Calculate stats
    avg_confidence = np.mean(confidences) if confidences else 0
    chars_per_second = len(predictions) / (timing['total_ms'] / 1000) if timing['total_ms'] > 0 else 0

    if verbose:
        print(f"\n  Total time: {timing['total_ms']:.1f}ms")
        print(f"  Speed: {chars_per_second:.1f} chars/sec")
        print(f"  Avg confidence: {avg_confidence:.2f}")
        if extracted_text:
            preview = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
            print(f"\n  Preview:\n  {preview}")

    # Build results
    results = {
        "image_info": {
            "path": image_path,
            "filename": os.path.basename(image_path),
            "width": width,
            "height": height,
            "processed_at": datetime.now().isoformat()
        },
        "genome_info": genome_info,
        "timing": {
            "total_ms": round(timing['total_ms'], 2),
            "load_ms": round(timing['load_ms'], 2),
            "segmentation_ms": round(timing['segmentation_ms'], 2),
            "inference_ms": round(timing['inference_ms'], 2),
            "inference_per_char_ms": round(timing['inference_ms'] / max(1, len(predictions)), 2),
            "chars_per_second": round(chars_per_second, 1)
        },
        "results": {
            "total_components": int(total_components),
            "filtered_components": int(filtered_count),
            "total_characters": int(len(predictions)),
            "total_lines": int(len(lines)),
            "extracted_text": extracted_text,
            "average_confidence": round(float(avg_confidence), 4)
        },
        "characters": [
            {
                "id": int(label),
                "bbox": [int(x) for x in pred['bbox']],
                "predicted": pred['letter'],
                "confidence": round(float(pred['confidence']), 4),
                "top_5": [[l, round(float(p), 4)] for l, p in pred['top_5']]
            }
            for label, pred in predictions.items()
        ]
    }

    return results


def save_results(results, image_path):
    """Save OCR results to txt and json files."""
    base_name = os.path.splitext(image_path)[0]

    # Save text file
    txt_path = f"{base_name}_ocr.txt"
    with open(txt_path, 'w') as f:
        f.write(results['results']['extracted_text'])
    print(f"  Saved: {txt_path}")

    # Save JSON file
    json_path = f"{base_name}_ocr.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    return txt_path, json_path


def process_folder(folder_path, network, genome_info, pattern="*.png", verbose=True):
    """Process all images in a folder."""
    # Find images
    search_pattern = os.path.join(folder_path, pattern)
    image_files = glob(search_pattern)

    # Also try common image extensions
    for ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
        if ext.lower() != pattern.lower():
            image_files.extend(glob(os.path.join(folder_path, ext)))

    # Remove duplicates and sort
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {folder_path}")
        return []

    print(f"\nFound {len(image_files)} images to process")

    all_results = []
    for image_path in image_files:
        results = process_single_image(image_path, network, genome_info, verbose=verbose)
        if results:
            save_results(results, image_path)
            all_results.append(results)

    return all_results


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Desktop OCR using GENREG alphabet genomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python desktop_ocr.py                           # Interactive genome selection
  python desktop_ocr.py --auto                    # Auto-select newest genome
  python desktop_ocr.py --genome path/to/genome.pkl
  python desktop_ocr.py --image screenshot.png    # Single image
  python desktop_ocr.py --folder ./screenshots    # Custom folder
        """
    )
    parser.add_argument('--genome', '-g', help='Path to genome .pkl file')
    parser.add_argument('--auto', '-a', action='store_true', help='Auto-select newest genome')
    parser.add_argument('--image', '-i', help='Process single image')
    parser.add_argument('--folder', '-f', default='desktop_images', help='Folder of images (default: desktop_images)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DESKTOP OCR PIPELINE")
    print("=" * 60)
    print("Using GENREG alphabet genome for character recognition")

    # Load genome
    if args.genome:
        if not os.path.exists(args.genome):
            print(f"Error: Genome file not found: {args.genome}")
            sys.exit(1)
        network, genome_info = load_genome(args.genome)
    else:
        genome_data = select_genome(auto_select=args.auto)
        network, genome_info = load_genome(genome_data['path'])

    # Process images
    verbose = not args.quiet

    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        results = process_single_image(args.image, network, genome_info, verbose=verbose)
        if results:
            save_results(results, args.image)
    else:
        # Folder mode
        folder = args.folder
        if not os.path.exists(folder):
            print(f"Error: Folder not found: {folder}")
            sys.exit(1)
        process_folder(folder, network, genome_info, verbose=verbose)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
