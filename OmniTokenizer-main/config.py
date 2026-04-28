# OmniVista Configuration

APP_NAME = "OmniVista"
APP_SUBTITLE = "The Universal Visual Tokenizer"

# Model Checkpoints from Hugging Face
MODELS = {
    "ImageNet + K600 (VQVAE)": {
        "url": "https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_k600.ckpt",
        "name": "imagenet_k600.ckpt",
        "description": "General purpose model trained on diverse image and video data."
    },
    "CelebAHQ (VQVAE)": {
        "url": "https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq.ckpt",
        "name": "celebahq.ckpt",
        "description": "Optimized for high-quality human face reconstruction."
    },
    "ImageNet + UCF (VAE)": {
        "url": "https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_ucf_vae.ckpt",
        "name": "imagenet_ucf_vae.ckpt",
        "description": "Variational Autoencoder version for smoother latent spaces."
    }
}

# UI Settings
THEME_COLOR = "#00d2ff"
SECONDARY_COLOR = "#9ea4b0"
DEFAULT_RESOLUTION = 256
MAX_VIDEO_LENGTH = 3 # seconds
