import torch
import numpy as np
from PIL import Image
import os
import requests
from tqdm import tqdm
from OmniTokenizer import OmniTokenizer_VQGAN
import cv2

def load_model(ckpt_path, device='cpu'):
    """Loads the OmniTokenizer model from a checkpoint."""
    model = OmniTokenizer_VQGAN.load_from_checkpoint(ckpt_path, strict=False, weights_only=False).to(device)
    model.eval()
    return model

def preprocess_image(image, resolution=256):
    """Preprocesses an image for the model."""
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), Image.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    return img_tensor

def postprocess_image(tensor):
    """Converts a model output tensor back to a PIL image."""
    tensor = (tensor + 1.0) / 2.0  # Denormalize to [0, 1]
    tensor = tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_array = (tensor * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def download_from_hf(url, filename):
    """Downloads a file from Hugging Face if it doesn't exist."""
    os.makedirs("ckpts", exist_ok=True)
    dest = os.path.join("ckpts", filename)
    if os.path.exists(dest):
        return dest
    
    print(f"Downloading {filename} from Hugging Face...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    return dest

def process_video(video_path, model, device='cpu', resolution=256, sequence_length=16):
    """Processes a video file through the model."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resolution, resolution))
        frames.append(frame)
    cap.release()
    
    if len(frames) < sequence_length:
        # Pad with last frame if video is too short
        while len(frames) < sequence_length:
            frames.append(frames[-1])
            
    video_array = np.stack(frames).astype(np.float32) / 255.0
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0) # [B, C, T, H, W]
    video_tensor = video_tensor * 2.0 - 1.0
    
    with torch.no_grad():
        recons = model.decode(model.encode(video_tensor.to(device), is_image=False), is_image=False)
        
    recons = (recons + 1.0) / 2.0
    recons = recons.clamp(0, 1).squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    recons_frames = (recons * 255).astype(np.uint8)
    
    return recons_frames

def visualize_tokens(encodings, resolution=256, patch_size=16):
    """Visualizes discrete tokens as a grid of colors."""
    # encodings shape: [B, N] where N is number of tokens
    # For image, N = (resolution // patch_size) ** 2
    tokens = encodings[0].cpu().numpy()
    grid_size = int(np.sqrt(len(tokens)))
    tokens_reshaped = tokens.reshape(grid_size, grid_size)
    
    # Create a pseudo-color image for tokens
    # Normalize token IDs to 0-255 for visualization
    max_token = tokens.max() if tokens.max() > 0 else 1
    tokens_norm = (tokens_reshaped / max_token * 255).astype(np.uint8)
    
    # Use a colormap
    tokens_colored = cv2.applyColorMap(tokens_norm, cv2.COLORMAP_JET)
    tokens_colored = cv2.cvtColor(tokens_colored, cv2.COLOR_BGR2RGB)
    
    # Resize for better visibility
    tokens_viz = Image.fromarray(tokens_colored).resize((resolution, resolution), Image.NEAREST)
    return tokens_viz

def get_token_statistics(encodings):
    """Calculates statistics for the tokens."""
    tokens = encodings[0].cpu().numpy()
    unique, counts = np.unique(tokens, return_counts=True)
    stats = {
        "unique_tokens": len(unique),
        "most_frequent_token": int(unique[np.argmax(counts)]),
        "token_entropy": float(-np.sum((counts/len(tokens)) * np.log2(counts/len(tokens))))
    }
    return stats
