import os
import random
import torch
import torch.nn as nn # Added for CrossEntropyLoss
import torch.optim as optim # Added for Adam
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zipfile
import urllib.request
import time

# Use your existing MLP definition
from mlp_model import MLP

# ================================================================
# CONFIGURATION
# ================================================================
MLP_FIELD_WIDTH = 100
MLP_FIELD_HEIGHT = 100
ALPHABET_FONT_SIZE = 64
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# IMPROVED TRAINING CONFIGURATION
EPOCHS = 200                     # Increased from 1 to 20
SAMPLES_PER_LETTER = 3000        # Samples per letter per epoch
BATCH_SIZE = 32                 # New: Train in batches for stability
LEARNING_RATE = 3e-3
MODEL_OUTPUT_PATH = "mlp_model.pth"


def download_dejavu_font():
    """Download DejaVuSans.ttf if not present."""
    font_path = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
    if os.path.exists(font_path): return font_path
    
    print("[FONT] DejaVuSans not found, downloading...")
    os.makedirs(FONTS_DIR, exist_ok=True)
    try:
        url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
        zip_path = os.path.join(FONTS_DIR, "dejavu-fonts.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith("DejaVuSans.ttf"):
                    with zip_ref.open(file) as src, open(font_path, 'wb') as dst:
                        dst.write(src.read())
                    break
        os.remove(zip_path)
        return font_path
    except Exception as e:
        print(f"[FONT] Download failed: {e}")
        return None

class HeadlessLetterRenderer:
    def __init__(self, use_augmentation=True):
        self.width = MLP_FIELD_WIDTH
        self.height = MLP_FIELD_HEIGHT
        self.use_augmentation = use_augmentation
        self.max_rotation = 25
        self.max_jitter_ratio = 0.2

        # Font Loading Logic
        local_font = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
        font_paths = [local_font, "C:\\Windows\\Fonts\\DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        
        self.font = None
        self.font_small = None
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, ALPHABET_FONT_SIZE)
                    small_size = max(8, ALPHABET_FONT_SIZE * 12 // 64)
                    self.font_small = ImageFont.truetype(path, small_size)
                    break
                except: continue
                
        if self.font is None:
            dl_path = download_dejavu_font()
            if dl_path:
                self.font = ImageFont.truetype(dl_path, ALPHABET_FONT_SIZE)
                small_size = max(8, ALPHABET_FONT_SIZE * 12 // 64)
                self.font_small = ImageFont.truetype(dl_path, small_size)
            else:
                self.font = ImageFont.load_default()
                self.font_small = ImageFont.load_default()

    def render(self, letter, variation=0):
        # ... (Your robust render logic from previous step) ...
        # I will paste the summarized version to save space, assuming you have the logic
        width, height = self.width, self.height
        base_style = variation % 4
        use_small = base_style < 2
        use_white_on_black = (base_style % 2) == 0

        font = self.font_small if use_small else self.font
        bg_color = (0, 0, 0) if use_white_on_black else (255, 255, 255)
        text_color = (255, 255, 255) if use_white_on_black else (0, 0, 0)
        
        img = Image.new('RGB', (width, height), bg_color)
        
        if self.use_augmentation:
            pad = int(max(width, height) * 0.5)
            cvs_size = max(width, height) + 2 * pad
            tmp_cvs = Image.new('RGBA', (cvs_size, cvs_size), (0,0,0,0))
            draw = ImageDraw.Draw(tmp_cvs)
            
            bbox = draw.textbbox((0,0), letter.upper(), font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            tx, ty = (cvs_size - tw)//2 - bbox[0], (cvs_size - th)//2 - bbox[1]
            
            draw.text((tx, ty), letter.upper(), font=font, fill=text_color+(255,))
            
            random.seed(variation * 1000 + ord(letter.upper()))
            rot = random.uniform(-self.max_rotation, self.max_rotation)
            rotated = tmp_cvs.rotate(rot, resample=Image.BICUBIC, expand=False)
            
            rbox = rotated.getbbox()
            rw = rbox[2]-rbox[0] if rbox else tw
            rh = rbox[3]-rbox[1] if rbox else th
            
            mjx = max(0, (width - rw)//2 - 2)
            mjy = max(0, (height - rh)//2 - 2)
            jrx = min(int(width * self.max_jitter_ratio), mjx)
            jry = min(int(height * self.max_jitter_ratio), mjy)
            jx = random.randint(-jrx, jrx) if jrx > 0 else 0
            jy = random.randint(-jry, jry) if jry > 0 else 0
            
            px, py = (width - cvs_size)//2 + jx, (height - cvs_size)//2 + jy
            img.paste(rotated, (px, py), rotated)
        else:
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0,0), letter.upper(), font=font)
            tx, ty = (width - (bbox[2]-bbox[0]))//2 - bbox[0], (height - (bbox[3]-bbox[1]))//2 - bbox[1]
            draw.text((tx, ty), letter.upper(), font=font, fill=text_color)
            
        return np.array(img.convert('L'), dtype=np.float32).flatten() / 255.0

def main():
    print("========================================")
    print("MLP Trainer (Batch Mode)")
    print("========================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize
    renderer = HeadlessLetterRenderer(use_augmentation=True)
    # Ensure MLP is initialized with correct flattened size (100*100 = 10000)
    model = MLP(input_size=10000, hidden_size=32, output_size=26).to(device)
    
    # Use standard PyTorch Optimizer/Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        
        # 1. GENERATE TRAINING TASK LIST (SHUFFLED)
        # Create a list of (letter_index, variation_seed) tuples
        tasks = []
        for i in range(26): # For each letter
            for s in range(SAMPLES_PER_LETTER):
                # Unique variation ID ensures different random augmentations
                variation = epoch * SAMPLES_PER_LETTER + s 
                tasks.append((i, variation))
        
        # CRITICAL: Shuffle the tasks to prevent catastrophic forgetting
        random.shuffle(tasks)
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        # 2. MINI-BATCH TRAINING LOOP
        start_time = time.time()
        
        # Process in chunks of BATCH_SIZE
        for i in range(0, len(tasks), BATCH_SIZE):
            batch_tasks = tasks[i : i + BATCH_SIZE]
            
            inputs = []
            targets = []
            
            # Render batch on CPU
            for label_idx, variation in batch_tasks:
                letter = LETTERS[label_idx]
                pixels = renderer.render(letter, variation=variation)
                inputs.append(pixels)
                targets.append(label_idx)
            
            # Move batch to GPU/Device
            data = torch.tensor(np.array(inputs), dtype=torch.float32).to(device)
            target = torch.tensor(np.array(targets), dtype=torch.long).to(device)
            
            # Gradient Descent Step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total_samples += len(batch_tasks)
            
        elapsed = time.time() - start_time
        avg_loss = total_loss / (len(tasks) / BATCH_SIZE)
        accuracy = (correct / total_samples) * 100
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | {total_samples/elapsed:.0f} img/s")

    # Save
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()