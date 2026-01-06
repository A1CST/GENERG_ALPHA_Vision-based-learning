# ================================================================
# GENREG Alphabet Inference Tool (Headless)
# ================================================================
# Command-line tool to test a trained alphabet model.
# No GUI - pure terminal output with detailed statistics.
#
# Usage:
#   python alphabet_inference_headless.py                    # Interactive genome selection
#   python alphabet_inference_headless.py path/to/genome.pkl # Specific genome
#   python alphabet_inference_headless.py --cycles 10        # Run 10 test cycles
#
# Features:
# - Tests all 26 letters with augmentation variations
# - Multiple randomized test cycles
# - Detailed per-letter accuracy breakdown
# - JSON output for analysis
# ================================================================

import os
import sys
import json
import pickle
import random
import math
import argparse
import time
import zipfile
import urllib.request
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================================================================
# CONFIGURATION
# ================================================================
ALPHABET_FIELD_WIDTH = 100
ALPHABET_FIELD_HEIGHT = 100
ALPHABET_FONT_SIZE = 64
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def download_dejavu_font():
    """Download DejaVuSans.ttf if not present."""
    font_path = os.path.join(FONTS_DIR, "DejaVuSans.ttf")

    if os.path.exists(font_path):
        return font_path

    print("[FONT] DejaVuSans not found, downloading...")

    # Create fonts directory
    os.makedirs(FONTS_DIR, exist_ok=True)

    # Download from GitHub releases
    url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
    zip_path = os.path.join(FONTS_DIR, "dejavu-fonts.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("[FONT] Downloaded, extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find and extract just DejaVuSans.ttf
            for file in zip_ref.namelist():
                if file.endswith("DejaVuSans.ttf"):
                    # Extract to fonts dir with just the filename
                    with zip_ref.open(file) as src, open(font_path, 'wb') as dst:
                        dst.write(src.read())
                    print(f"[FONT] Extracted DejaVuSans.ttf to {font_path}")
                    break

        # Clean up zip
        os.remove(zip_path)

        if os.path.exists(font_path):
            return font_path
        else:
            print("[FONT] Failed to extract DejaVuSans.ttf from zip")
            return None

    except Exception as e:
        print(f"[FONT] Failed to download font: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None


# ================================================================
# SIMPLE NEURAL NETWORK
# ================================================================
class SimpleNetwork:
    """Minimal forward-pass-only neural network for inference."""

    def __init__(self, input_size, hidden_size, output_size, force_cpu=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Device info for display
        self.device_type = "cpu"  # 'cpu', 'cuda', or 'numpy'
        self.device_name = "CPU (NumPy)"

        # Try to use PyTorch if available
        try:
            import torch
            self._use_torch = True
            if torch.cuda.is_available() and not force_cpu:
                self.device = torch.device('cuda')
                self.device_type = "cuda"
                self.device_name = f"GPU: {torch.cuda.get_device_name(0)}"
            else:
                self.device = torch.device('cpu')
                self.device_type = "cpu"
                self.device_name = "CPU (PyTorch)"
        except ImportError:
            self._use_torch = False
            self.device = None
            self.device_type = "numpy"
            self.device_name = "CPU (NumPy - no PyTorch)"

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def get_device_info(self):
        """Return device information as a dict."""
        info = {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "using_torch": self._use_torch,
            "using_gpu": self.device_type == "cuda"
        }
        if self._use_torch and self.device_type == "cuda":
            import torch
            info["cuda_version"] = torch.version.cuda
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        return info

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
        # Hidden layer
        hidden = np.tanh(self.w1 @ visual_input + self.b1)

        # Output layer
        outputs = self.w2 @ hidden + self.b2

        # Softmax
        char_logits = outputs[:26]
        exp_logits = np.exp(char_logits - np.max(char_logits))
        char_probs = exp_logits / np.sum(exp_logits)

        return char_probs.tolist()

    def predict(self, visual_input):
        """Get the predicted letter (argmax, no sampling)."""
        char_probs = self.forward_visual(visual_input)
        max_idx = np.argmax(char_probs[:26])
        return chr(ord('A') + max_idx), char_probs[max_idx]

    def predict_top_k(self, visual_input, k=3):
        """Get top-k predictions with probabilities."""
        char_probs = self.forward_visual(visual_input)
        indices = np.argsort(char_probs)[::-1][:k]
        return [(chr(ord('A') + i), char_probs[i]) for i in indices]


# ================================================================
# HEADLESS LETTER RENDERER (PIL-based, matches training)
# ================================================================
class HeadlessLetterRenderer:
    """
    Render letters using PIL (no pygame required).
    Matches the HeadlessAlphabetEnv from training as closely as possible.
    """

    def __init__(self, use_augmentation=True):
        self.width = ALPHABET_FIELD_WIDTH
        self.height = ALPHABET_FIELD_HEIGHT
        self.use_augmentation = use_augmentation

        # Augmentation parameters (match training)
        self.max_rotation = 25  # degrees
        self.max_jitter_ratio = 0.2  # fraction of image size

        # Load font - try local fonts dir first, then system paths
        local_font = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
        font_paths = [
            local_font,  # Local fonts directory (auto-downloaded)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/Helvetica.ttc",  # Mac
            "C:\\Windows\\Fonts\\DejaVuSans.ttf",  # Windows (if manually installed)
        ]

        self.font = None
        self.font_small = None

        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, ALPHABET_FONT_SIZE)
                    self.font_small = ImageFont.truetype(path, max(8, ALPHABET_FONT_SIZE * 12 // 64))
                    print(f"[RENDERER] Loaded font: {path}")
                    break
                except:
                    pass

        # If no font found, try to download DejaVuSans
        if self.font is None:
            downloaded_path = download_dejavu_font()
            if downloaded_path:
                try:
                    self.font = ImageFont.truetype(downloaded_path, ALPHABET_FONT_SIZE)
                    self.font_small = ImageFont.truetype(downloaded_path, max(8, ALPHABET_FONT_SIZE * 12 // 64))
                    print(f"[RENDERER] Loaded font: {downloaded_path}")
                except Exception as e:
                    print(f"[RENDERER] Failed to load downloaded font: {e}")

        if self.font is None:
            print("[RENDERER] FAILED TO LOAD FONTS")
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def render(self, letter, variation=0):
        """
        Render a letter with variation matching training augmentation.

        Variations cycle through:
            - Base style: 4 combinations (2 font sizes × 2 color schemes)
            - With augmentation: adds rotation and jitter

        Returns: flattened grayscale pixel array
        """
        width, height = self.width, self.height

        # Base style cycles through 4 options
        base_style = variation % 4
        use_small_font = base_style < 2
        use_white_on_black = (base_style % 2) == 0

        font = self.font_small if use_small_font else self.font

        if use_white_on_black:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
        else:
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)

        # Create background image
        img = Image.new('RGB', (width, height), bg_color)

        if self.use_augmentation:
            # Match training: render on larger canvas, rotate, then paste
            padding = int(max(width, height) * 0.5)
            canvas_size = max(width, height) + 2 * padding
            letter_canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
            letter_draw = ImageDraw.Draw(letter_canvas)

            # Get text bounding box
            bbox = letter_draw.textbbox((0, 0), letter.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw letter centered on canvas
            text_x = (canvas_size - text_width) // 2 - bbox[0]
            text_y = (canvas_size - text_height) // 2 - bbox[1]
            letter_draw.text((text_x, text_y), letter.upper(), font=font, fill=text_color + (255,))

            # Apply rotation (deterministic based on variation and letter)
            random.seed(variation * 1000 + ord(letter.upper()))
            rotation = random.uniform(-self.max_rotation, self.max_rotation)
            rotated = letter_canvas.rotate(rotation, resample=Image.BICUBIC, expand=False)

            # Calculate safe jitter bounds
            rotated_bbox = rotated.getbbox()
            if rotated_bbox:
                rot_width = rotated_bbox[2] - rotated_bbox[0]
                rot_height = rotated_bbox[3] - rotated_bbox[1]
            else:
                rot_width, rot_height = text_width, text_height

            max_jitter_x = max(0, (width - rot_width) // 2 - 2)
            max_jitter_y = max(0, (height - rot_height) // 2 - 2)

            jitter_range_x = min(int(width * self.max_jitter_ratio), max_jitter_x)
            jitter_range_y = min(int(height * self.max_jitter_ratio), max_jitter_y)

            jitter_x = random.randint(-jitter_range_x, jitter_range_x) if jitter_range_x > 0 else 0
            jitter_y = random.randint(-jitter_range_y, jitter_range_y) if jitter_range_y > 0 else 0

            # Paste
            paste_x = (width - canvas_size) // 2 + jitter_x
            paste_y = (height - canvas_size) // 2 + jitter_y
            img.paste(rotated, (paste_x, paste_y), rotated)
        else:
            # Simple centered rendering (no augmentation)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), letter.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2 - bbox[0]
            y = (height - text_height) // 2 - bbox[1]
            draw.text((x, y), letter.upper(), font=font, fill=text_color)

        # Convert to grayscale normalized array
        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32) / 255.0
        return arr.flatten()

    def render_simple(self, letter, invert=False):
        """
        Render a letter with NO augmentation (simple centered).
        Use this for testing models trained without augmentation.
        """
        width, height = self.width, self.height

        if invert:
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)
        else:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)

        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), letter.upper(), font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2 - bbox[0]
        y = (height - text_height) // 2 - bbox[1]
        draw.text((x, y), letter.upper(), font=self.font, fill=text_color)

        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32) / 255.0
        return arr.flatten()


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
    genomes = []
    genome_dir = "best_genomes"

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


def select_genome_interactive():
    """Interactive genome selection."""
    genomes = list_alphabet_genomes()

    if not genomes:
        print("No alphabet genomes found in best_genomes/")
        print("Extract one first with: python extract_best_genome.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SELECT ALPHABET GENOME")
    print("=" * 60)
    print("\nAvailable alphabet genomes:\n")

    for i, g in enumerate(genomes[:10]):
        size_str = format_size(g['size'])
        print(f"  [{i+1}] {g['filename']}")
        print(f"       Gen {g['generation']}, {size_str}")

    if len(genomes) == 1:
        print(f"\n  [Enter] Use {genomes[0]['filename']}")
    else:
        print(f"\n  [Enter] Use most recent")

    choice = input("\nYour choice: ").strip()

    if choice == "":
        return genomes[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(genomes):
        return genomes[int(choice) - 1]
    else:
        print("Invalid choice, using most recent...")
        return genomes[0]


def check_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_device_name():
    """Get CUDA device name if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def select_device_interactive():
    """
    Prompt user to select CPU or GPU if CUDA is available.
    Returns True if user wants to force CPU, False otherwise.
    """
    if not check_cuda_available():
        return False  # No choice needed, will use CPU

    gpu_name = get_cuda_device_name()

    print(f"\n{'=' * 60}")
    print("SELECT DEVICE")
    print(f"{'=' * 60}")
    print(f"\nCUDA is available! GPU detected: {gpu_name}")
    print()
    print("  [1] GPU (CUDA) - Faster inference")
    print("  [2] CPU - Use processor instead")
    print()

    while True:
        choice = input("Select device (1/2) [default: 1]: ").strip()
        if choice == "" or choice == "1":
            print(f"\n  -> Using GPU: {gpu_name}")
            return False  # Don't force CPU
        elif choice == "2":
            print("\n  -> Using CPU")
            return True  # Force CPU
        else:
            print("  Invalid choice. Enter 1 or 2.")


def load_genome(genome_path, force_cpu=False):
    """Load a genome from pickle file."""
    print(f"\nLoading: {genome_path}")

    with open(genome_path, 'rb') as f:
        data = pickle.load(f)

    # Extract controller data
    ctrl_data = data['controller']
    input_size = ctrl_data['input_size']
    hidden_size = ctrl_data['hidden_size']
    output_size = ctrl_data['output_size']

    print(f"  Network: {input_size} -> {hidden_size} -> {output_size}")

    # Verify it's an alphabet model
    expected_input = ALPHABET_FIELD_WIDTH * ALPHABET_FIELD_HEIGHT
    if input_size != expected_input:
        print(f"  WARNING: Expected {expected_input} input (alphabet), got {input_size}")

    if output_size != 26:
        print(f"  WARNING: Expected 26 output (letters), got {output_size}")

    # Create network and load weights
    network = SimpleNetwork(input_size, hidden_size, output_size, force_cpu=force_cpu)
    network.load_weights(ctrl_data)

    # Get metadata
    trust = data.get('genome', {}).get('trust', 0)
    print(f"  Trust: {trust:.2f}")

    return network, data


# ================================================================
# TEST FUNCTIONS
# ================================================================
def test_single_pass(network, renderer, variations=None):
    """
    Test all 26 letters once.

    Returns: dict with results
    """
    if variations is None:
        variations = [0]  # Default: just base variation

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    results = {}

    for letter in letters:
        letter_results = []
        for var in variations:
            obs = renderer.render(letter, variation=var)
            predicted, confidence = network.predict(obs)
            is_correct = (predicted == letter)
            letter_results.append({
                'variation': var,
                'predicted': predicted,
                'confidence': round(confidence, 4),
                'correct': is_correct
            })

        # Aggregate: correct if majority are correct
        correct_count = sum(1 for r in letter_results if r['correct'])
        results[letter] = {
            'predictions': letter_results,
            'correct_count': correct_count,
            'total_variations': len(variations),
            'accuracy': correct_count / len(variations)
        }

    return results


def run_test_cycles(network, renderer, num_cycles=5, variations=None, verbose=True):
    """
    Run multiple randomized test cycles.

    Returns: comprehensive results dict
    """
    if variations is None:
        variations = [0, 1, 2, 3]  # Test 4 variations

    all_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Tracking
    per_letter_stats = {letter: {"correct": 0, "total": 0, "predictions": [], "inference_times_ms": []}
                        for letter in all_letters}
    confusion_matrix = {letter: {pred: 0 for pred in all_letters}
                        for letter in all_letters}
    cycle_results = []

    total_correct = 0
    total_tests = 0

    # Timing tracking
    all_inference_times = []
    overall_start_time = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RUNNING {num_cycles} TEST CYCLES")
        print(f"Variations per letter: {len(variations)}")
        print(f"Total tests: {num_cycles} x 26 x {len(variations)} = {num_cycles * 26 * len(variations)}")
        print(f"Device: {network.device_name}")
        print(f"{'=' * 60}")

    for cycle_num in range(1, num_cycles + 1):
        # Shuffle for this cycle
        shuffled = all_letters.copy()
        random.shuffle(shuffled)

        cycle_data = {
            "cycle": cycle_num,
            "order": shuffled.copy(),
            "results": []
        }

        cycle_correct = 0
        cycle_total = 0
        cycle_start_time = time.perf_counter()

        for letter in shuffled:
            for var in variations:
                obs = renderer.render(letter, variation=var)

                # Time the inference
                inference_start = time.perf_counter()
                predicted, confidence = network.predict(obs)
                inference_end = time.perf_counter()
                inference_time_ms = (inference_end - inference_start) * 1000

                all_inference_times.append(inference_time_ms)
                per_letter_stats[letter]["inference_times_ms"].append(inference_time_ms)

                is_correct = (predicted == letter)

                total_tests += 1
                cycle_total += 1
                per_letter_stats[letter]["total"] += 1

                if is_correct:
                    total_correct += 1
                    cycle_correct += 1
                    per_letter_stats[letter]["correct"] += 1

                per_letter_stats[letter]["predictions"].append(predicted)
                confusion_matrix[letter][predicted] += 1

                cycle_data["results"].append({
                    "letter": letter,
                    "variation": var,
                    "predicted": predicted,
                    "confidence": round(confidence, 4),
                    "correct": is_correct,
                    "inference_time_ms": round(inference_time_ms, 4)
                })

        cycle_end_time = time.perf_counter()
        cycle_time_s = cycle_end_time - cycle_start_time
        cycle_inferences_per_sec = cycle_total / cycle_time_s if cycle_time_s > 0 else 0

        cycle_data["accuracy"] = round(cycle_correct / cycle_total * 100, 2)
        cycle_data["cycle_time_s"] = round(cycle_time_s, 3)
        cycle_data["inferences_per_sec"] = round(cycle_inferences_per_sec, 1)
        cycle_results.append(cycle_data)

        if verbose:
            print(f"  Cycle {cycle_num}/{num_cycles}: {cycle_correct}/{cycle_total} "
                  f"({cycle_data['accuracy']:.1f}%) | {cycle_inferences_per_sec:.1f} inf/s")

    # Calculate final stats
    overall_end_time = time.perf_counter()
    total_time_s = overall_end_time - overall_start_time
    overall_accuracy = total_correct / total_tests * 100

    # Timing statistics
    if all_inference_times:
        avg_inference_ms = sum(all_inference_times) / len(all_inference_times)
        min_inference_ms = min(all_inference_times)
        max_inference_ms = max(all_inference_times)
        sorted_times = sorted(all_inference_times)
        median_inference_ms = sorted_times[len(sorted_times) // 2]
        # Standard deviation
        variance = sum((t - avg_inference_ms) ** 2 for t in all_inference_times) / len(all_inference_times)
        std_inference_ms = variance ** 0.5
        overall_inferences_per_sec = total_tests / total_time_s if total_time_s > 0 else 0
    else:
        avg_inference_ms = min_inference_ms = max_inference_ms = median_inference_ms = std_inference_ms = 0
        overall_inferences_per_sec = 0

    # Letter accuracies
    letter_accuracies = []
    for letter in all_letters:
        stats = per_letter_stats[letter]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
        else:
            acc = 0
        letter_accuracies.append((letter, acc))

    letter_accuracies.sort(key=lambda x: x[1], reverse=True)
    perfect_letters = [l for l, a in letter_accuracies if a == 100]
    problem_letters = [(l, a) for l, a in letter_accuracies if a < 100]
    problem_letters.sort(key=lambda x: x[1])

    # Build results
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "num_cycles": num_cycles,
            "variations_per_letter": len(variations),
            "total_tests": total_tests,
            "device": network.get_device_info()
        },
        "summary": {
            "overall_accuracy": round(overall_accuracy, 2),
            "total_correct": total_correct,
            "total_wrong": total_tests - total_correct,
            "perfect_letters": perfect_letters,
            "perfect_letter_count": len(perfect_letters),
            "problem_letters": [{"letter": l, "accuracy": round(a, 1)}
                               for l, a in problem_letters[:10]]
        },
        "timing": {
            "total_time_s": round(total_time_s, 3),
            "inferences_per_sec": round(overall_inferences_per_sec, 1),
            "avg_inference_ms": round(avg_inference_ms, 4),
            "min_inference_ms": round(min_inference_ms, 4),
            "max_inference_ms": round(max_inference_ms, 4),
            "median_inference_ms": round(median_inference_ms, 4),
            "std_inference_ms": round(std_inference_ms, 4)
        },
        "per_letter_stats": {
            letter: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": round(stats["correct"] / max(1, stats["total"]) * 100, 1),
                "predictions": stats["predictions"],
                "avg_inference_ms": round(sum(stats["inference_times_ms"]) / max(1, len(stats["inference_times_ms"])), 4)
            }
            for letter, stats in per_letter_stats.items()
        },
        "confusion_matrix": confusion_matrix,
        "cycle_details": cycle_results
    }

    return results


def print_results_summary(results):
    """Print a formatted summary of test results."""
    summary = results["summary"]
    timing = results["timing"]
    device_info = results["test_info"]["device"]

    print(f"\n{'=' * 60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 60}")

    # Device info
    print(f"\nDevice: {device_info['device_name']}")
    if device_info['using_gpu']:
        print(f"  CUDA: {device_info.get('cuda_version', 'N/A')}")
        print(f"  GPU Memory: {device_info.get('gpu_memory_total', 'N/A')}")

    # Accuracy
    print(f"\nOverall Accuracy: {summary['overall_accuracy']:.1f}%")
    print(f"Correct: {summary['total_correct']} / {summary['total_correct'] + summary['total_wrong']}")

    # Timing/Speed statistics
    print(f"\n{'─' * 40}")
    print("INFERENCE SPEED")
    print(f"{'─' * 40}")
    print(f"  Total time:        {timing['total_time_s']:.3f} s")
    print(f"  Throughput:        {timing['inferences_per_sec']:.1f} inferences/sec")
    print(f"  Avg inference:     {timing['avg_inference_ms']:.4f} ms")
    print(f"  Min inference:     {timing['min_inference_ms']:.4f} ms")
    print(f"  Max inference:     {timing['max_inference_ms']:.4f} ms")
    print(f"  Median inference:  {timing['median_inference_ms']:.4f} ms")
    print(f"  Std deviation:     {timing['std_inference_ms']:.4f} ms")

    # Per-letter timing (show slowest 5)
    letter_times = [(letter, stats['avg_inference_ms'])
                    for letter, stats in results['per_letter_stats'].items()]
    letter_times.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Slowest letters:")
    for letter, avg_ms in letter_times[:5]:
        print(f"    {letter}: {avg_ms:.4f} ms avg")

    print(f"\n{'─' * 40}")
    print("ACCURACY BREAKDOWN")
    print(f"{'─' * 40}")

    print(f"\nPerfect Letters ({summary['perfect_letter_count']}/26):")
    if summary['perfect_letters']:
        # Print in rows of 13
        perfect = summary['perfect_letters']
        for i in range(0, len(perfect), 13):
            print(f"  {' '.join(perfect[i:i+13])}")
    else:
        print("  None")

    if summary['problem_letters']:
        print(f"\nProblem Letters:")
        for item in summary['problem_letters'][:10]:
            letter = item['letter']
            acc = item['accuracy']
            # Get most common wrong predictions
            preds = results['per_letter_stats'][letter]['predictions']
            wrong_preds = [p for p in preds if p != letter]
            if wrong_preds:
                from collections import Counter
                common = Counter(wrong_preds).most_common(3)
                pred_str = ", ".join([f"{p}({c})" for p, c in common])
                print(f"  {letter}: {acc:.0f}% (confused with: {pred_str})")
            else:
                print(f"  {letter}: {acc:.0f}%")

    print(f"\n{'=' * 60}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="GENREG Alphabet Inference (Headless)")
    parser.add_argument("genome", nargs="?", help="Path to genome .pkl file")
    parser.add_argument("--cycles", type=int, default=5, help="Number of test cycles (default: 5)")
    parser.add_argument("--variations", type=int, default=4, help="Variations per letter (default: 4)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable augmentation (for models trained without rotation/jitter)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GENREG ALPHABET INFERENCE (HEADLESS)")
    print("=" * 60)

    # Load genome
    if args.genome:
        if not os.path.exists(args.genome):
            print(f"Error: File not found: {args.genome}")
            sys.exit(1)
        genome_path = args.genome
    else:
        genome_info = select_genome_interactive()
        genome_path = genome_info['path']

    # Device selection (only prompts if CUDA is available)
    force_cpu = select_device_interactive()

    network, genome_data = load_genome(genome_path, force_cpu=force_cpu)

    # Display device info prominently
    device_info = network.get_device_info()
    print(f"\n{'─' * 40}")
    if device_info['using_gpu']:
        print(f"  RUNNING ON GPU")
        print(f"  {device_info['device_name']}")
        print(f"  CUDA {device_info.get('cuda_version', 'N/A')} | {device_info.get('gpu_memory_total', 'N/A')}")
    else:
        print(f"  RUNNING ON CPU")
        print(f"  {device_info['device_name']}")
    print(f"{'─' * 40}")

    # Create renderer
    use_aug = not args.no_augmentation
    renderer = HeadlessLetterRenderer(use_augmentation=use_aug)
    if not use_aug:
        print("[RENDERER] Augmentation DISABLED (simple centered rendering)")

    # Run tests
    variations = list(range(args.variations))
    results = run_test_cycles(
        network, renderer,
        num_cycles=args.cycles,
        variations=variations,
        verbose=not args.quiet
    )

    # Print summary
    if not args.quiet:
        print_results_summary(results)

    # Save JSON output
    if args.output:
        output_path = args.output
    else:
        os.makedirs("inference/output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inference/output/headless_test_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Return exit code based on accuracy
    if results["summary"]["overall_accuracy"] >= 90:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
