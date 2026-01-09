# ================================================================
# MLP Backpropagation Inference Tool (Headless)
# ================================================================
# Command-line tool to test a trained MLP model (PyTorch .pth file).
# Uses the same inference logic as alphabet_inference_headless.py.
#
# Usage:
#   python backprop_inference.py path/to/mlp_model.pth # Specific model
#   python backprop_inference.py --cycles 10           # Run 10 test cycles
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
import random
import math
import argparse
import time
import zipfile
import urllib.request
from datetime import datetime
import torch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlp_model import MLP

# ================================================================
# CONFIGURATION
# ================================================================
ALPHABET_FIELD_WIDTH = 100
ALPHABET_FIELD_HEIGHT = 100
ALPHABET_FONT_SIZE = 64
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
MLP_MODEL_PATH = "mlp_model.pth" # Default path for the trained MLP model


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
# MLP NETWORK LOADER
# ================================================================
class MlpInferenceNetwork:
    """PyTorch MLP for inference."""

    def __init__(self, model_path, input_size, hidden_size, output_size, force_cpu=False, force_gpu=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.device_type = "cpu"
        self.device_name = "CPU (PyTorch)"
        self._use_torch = True # Always True for PyTorch model

        use_cuda = torch.cuda.is_available() and not force_cpu
        if force_gpu and torch.cuda.is_available():
            use_cuda = True

        if use_cuda:
            self.device = torch.device('cuda')
            self.device_type = "cuda"
            self.device_name = f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            self.device = torch.device('cpu')
            self.device_type = "cpu"
            self.device_name = "CPU (PyTorch)"

        self.model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def get_device_info(self):
        """Return device information as a dict."""
        info = {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "using_torch": self._use_torch,
            "using_gpu": self.device_type == "cuda"
        }
        if self.device_type == "cuda":
            info["cuda_version"] = torch.version.cuda
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        return info

    def forward_visual(self, visual_input):
        """Forward pass returning character probabilities."""
        with torch.no_grad():
            x = torch.tensor(visual_input, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0) # Add batch dimension
            outputs = self.model(x)
            char_probs = torch.softmax(outputs, dim=1) # Softmax over class dimension
            return char_probs.squeeze(0).cpu().tolist() # Remove batch dim, convert to list

    def predict(self, visual_input):
        """Get the predicted letter (argmax, no sampling)."""
        char_probs = self.forward_visual(visual_input)
        max_idx = np.argmax(char_probs[:self.output_size])
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
    def __init__(self, use_augmentation=True):
        self.width = ALPHABET_FIELD_WIDTH
        self.height = ALPHABET_FIELD_HEIGHT
        self.use_augmentation = use_augmentation
        
        # Augmentation parameters (Must match EA exactly)
        self.max_rotation = 25
        self.max_jitter_ratio = 0.2

        # Define specific font paths to search (Local -> Linux -> Windows -> Mac)
        local_font = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
        font_paths = [
            local_font,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:\\Windows\\Fonts\\DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]

        self.font = None
        self.font_small = None

        # 1. Try to load from known paths
        for path in font_paths:
            if os.path.exists(path):
                try:
                    # Load Large Font (64pt)
                    self.font = ImageFont.truetype(path, ALPHABET_FONT_SIZE)
                    
                    # Load Small Font (Calculated exactly like EA: ~12pt)
                    # The formula (64 * 12 // 64) ensures precise scaling ratio
                    small_size = max(8, ALPHABET_FONT_SIZE * 12 // 64)
                    self.font_small = ImageFont.truetype(path, small_size)
                    
                    print(f"[RENDERER] Loaded font: {path}")
                    break
                except Exception:
                    continue

        # 2. If not found, try to download DejaVuSans
        if self.font is None:
            downloaded_path = download_dejavu_font()
            if downloaded_path:
                try:
                    # Load Large
                    self.font = ImageFont.truetype(downloaded_path, ALPHABET_FONT_SIZE)
                    
                    # Load Small
                    small_size = max(8, ALPHABET_FONT_SIZE * 12 // 64)
                    self.font_small = ImageFont.truetype(downloaded_path, small_size)
                    
                    print(f"[RENDERER] Loaded downloaded font: {downloaded_path}")
                except Exception as e:
                    print(f"[RENDERER] Failed to load downloaded font: {e}")

        # 3. Last Resort: Default PIL Font
        if self.font is None:
            print("[RENDERER] WARNING: FAILED TO LOAD FONTS. Using default.")
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
	    
    def render(self, letter, variation=0):
        """
        Render a letter with variation matching the EA training exactly.
        Cycles through 4 base styles (Size/Color) + Augmentation.
        """
        width, height = self.width, self.height

        # --- 1. DETERMINE BASE STYLE (Crucial for fair comparison) ---
        base_style = variation % 4
        use_small_font = base_style < 2      # 0, 1 use small font
        use_white_on_black = (base_style % 2) == 0 # 0, 2 are white on black

        # Select the correct font object
        font = self.font_small if use_small_font else self.font

        # Set colors
        if use_white_on_black:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
        else:
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)

        # Create base image
        img = Image.new('RGB', (width, height), bg_color)

        # --- 2. AUGMENTATION PIPELINE ---
        if self.use_augmentation:
            # Create a larger canvas to safely rotate without clipping corners
            padding = int(max(width, height) * 0.5)
            canvas_size = max(width, height) + 2 * padding
            
            # Use RGBA for the temporary rotation canvas
            letter_canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
            letter_draw = ImageDraw.Draw(letter_canvas)

            # Get text bounding box to center it
            bbox = letter_draw.textbbox((0, 0), letter.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = (canvas_size - text_width) // 2 - bbox[0]
            text_y = (canvas_size - text_height) // 2 - bbox[1]
            
            # Draw text onto transparent canvas
            # Note: text_color + (255,) adds full opacity alpha channel
            letter_draw.text((text_x, text_y), letter.upper(), font=font, fill=text_color + (255,))

            # Deterministic rotation based on variation index
            random.seed(variation * 1000 + ord(letter.upper()))
            rotation = random.uniform(-self.max_rotation, self.max_rotation)
            
            # Rotate
            rotated = letter_canvas.rotate(rotation, resample=Image.BICUBIC, expand=False)

            # Calculate safe bounds for jitter (shifting)
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

            # Paste rotated text onto background with jitter offset
            paste_x = (width - canvas_size) // 2 + jitter_x
            paste_y = (height - canvas_size) // 2 + jitter_y
            
            # Use the rotated image itself as the mask for transparency
            img.paste(rotated, (paste_x, paste_y), rotated)

        # --- 3. NO AUGMENTATION (Simple Center) ---
        else:
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), letter.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2 - bbox[0]
            y = (height - text_height) // 2 - bbox[1]
            draw.text((x, y), letter.upper(), font=font, fill=text_color)

        # --- 4. FINALIZE ---
        # Convert to grayscale and normalize to 0.0 - 1.0 float array
        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32) / 255.0
        
        return arr.flatten()        # ... (Rest of render function remains the same) ...

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


def run_test_cycles(network, renderer, num_cycles=5, verbose=True):
    # We want to test all 4 base styles (Small/Large, Black/White)
    variations_to_test = [0, 1, 2, 3]
    num_variations_per_letter = len(variations_to_test)

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
        print(f"Variations per letter: {num_variations_per_letter}")
        print(f"Total tests: {num_cycles * 26 * num_variations_per_letter}")
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
            # FIX: Loop through all 4 variations
            for var in variations_to_test:
                
                # Render specific variation
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
                    "variation": var, # Log the actual variation
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
    overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0

    # ... (Rest of stats calculation remains the same) ...
    # Be sure to include the rest of the original function here down to "return results"
    
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
            "variations_per_letter": num_variations_per_letter,
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
    parser.add_argument("model", nargs="?", help="Path to mlp_model.pth file")
    parser.add_argument("--cycles", type=int, default=5, help="Number of test cycles (default: 5)")
    parser.add_argument("--variations", type=int, default=4, help="Variations per letter (default: 4)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable augmentation (for models trained without rotation/jitter)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU if CUDA is available")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MLP BACKPROPAGATION INFERENCE (HEADLESS)")
    print("=" * 60)

    # Load model
    model_path = args.model if args.model else MLP_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the MLP model first by running mlp_trainer.py")
        sys.exit(1)

    # Device selection
    force_cpu = args.force_cpu
    force_gpu = args.force_gpu

    # Assuming a fixed architecture for the MLP model
    input_size = ALPHABET_FIELD_WIDTH * ALPHABET_FIELD_HEIGHT
    hidden_size = 32 # From the original request
    output_size = 26 # From the original request

    network = MlpInferenceNetwork(model_path, input_size, hidden_size, output_size, force_cpu=force_cpu, force_gpu=force_gpu)

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

    results = run_test_cycles(
        network, renderer,
        num_cycles=args.cycles,
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
        output_path = f"inference/output/mlp_backprop_test_{timestamp}.json"

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
