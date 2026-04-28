import streamlit as st
import torch
import os
from PIL import Image
import numpy as np
import tempfile
import cv2
import time
from utils_app import load_model, preprocess_image, postprocess_image, process_video, visualize_tokens, get_token_statistics
from OmniTokenizer.download import download
import config

# Page Configuration
st.set_page_config(
    page_title=f"{config.APP_NAME} | {config.APP_SUBTITLE}",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown(f"""
<style>
    .main {{
        background-color: #0e1117;
    }}
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }}
    .header-style {{
        font-size: 48px;
        font-weight: 800;
        background: -webkit-linear-gradient({config.THEME_COLOR}, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }}
    .subheader-style {{
        font-size: 18px;
        color: {config.SECONDARY_COLOR};
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: 1px;
    }}
    .stats-card {{
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://www.wangjunke.info/OmniTokenizer/static/images/teaser.png", use_container_width=True)
st.sidebar.title(f"🚀 {config.APP_NAME} Control")

selected_model_key = st.sidebar.selectbox("Model Weight", list(config.MODELS.keys()))
model_info = config.MODELS[selected_model_key]
st.sidebar.caption(model_info["description"])

resolution = st.sidebar.select_slider("Inference Resolution", options=[128, 256, 384, 512], value=config.DEFAULT_RESOLUTION)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.divider()
st.sidebar.info(f"⚡ Engine: **{device.upper()}**")

# Load Model Logic
@st.cache_resource
def get_cached_model(model_key, device):
    m_info = config.MODELS[model_key]
    ckpt_path = download(m_info["url"], m_info["name"])
    return load_model(ckpt_path, device)

with st.spinner("Initializing OmniVista Engine..."):
    model = get_cached_model(selected_model_key, device)

# Main Content
st.markdown(f'<div class="header-style">{config.APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subheader-style">{config.APP_SUBTITLE}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Engine", "🎥 Video Engine", "🧩 Token Analysis", "📚 Documentation"])

with tab1:
    st.markdown("### Reconstruction & Batch Processing")
    uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_images:
        for uploaded_image in uploaded_images:
            with st.expander(f"Processing: {uploaded_image.name}", expanded=True):
                col1, col2 = st.columns(2)
                input_image = Image.open(uploaded_image)
                with col1:
                    st.image(input_image, caption="Original", use_container_width=True)
                
                if st.button(f"Generate Reconstruction for {uploaded_image.name}", key=uploaded_image.name):
                    start_time = time.time()
                    input_tensor = preprocess_image(input_image, resolution=resolution).to(device)
                    with torch.no_grad():
                        encoded = model.encode(input_tensor, is_image=True)
                        decoded = model.decode(encoded, is_image=True)
                    
                    end_time = time.time()
                    output_image = postprocess_image(decoded)
                    with col2:
                        st.image(output_image, caption=f"Reconstructed ({end_time-start_time:.2f}s)", use_container_width=True)
                        st.download_button("Download Image", data=cv2.imencode('.png', np.array(output_image))[1].tobytes(), file_name=f"omnivista_{uploaded_image.name}", mime="image/png")

with tab2:
    st.info(f"Note: Video reconstruction handles up to {config.MAX_VIDEO_LENGTH} seconds. Processing is GPU-heavy.")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        
        if st.button("🚀 Process Video Engine"):
            with st.spinner("Analyzing temporal patches..."):
                recons_frames = process_video(tfile.name, model, device=device, resolution=resolution)
                
                # Write to a proper temp file with browser-compatible format
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out_tmp.close()
                
                height, width = recons_frames[0].shape[:2]
                # Use XVID codec which is more universally supported
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                tmp_avi = out_tmp.name.replace('.mp4', '.avi')
                out = cv2.VideoWriter(tmp_avi, fourcc, 10, (width, height))
                for frame in recons_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                
                # Read bytes directly and pass to Streamlit
                with open(tmp_avi, 'rb') as f:
                    video_bytes = f.read()
                
                st.video(video_bytes)
                st.success("✅ Reconstruction success! Download the video above.")
                st.download_button(
                    "⬇️ Download Reconstructed Video",
                    data=video_bytes,
                    file_name="omnivista_reconstructed.avi",
                    mime="video/x-msvideo"
                )


with tab3:
    st.markdown("### 🧩 Latent Space & Discrete Tokens")
    st.markdown("Tokenization converts images into a grid of 'visual words'. This tab visualizes those discrete units.")
    
    token_image = st.file_uploader("Upload image for Token Analysis", type=["jpg", "png"], key="token_upload")
    
    if token_image:
        col_img, col_tok = st.columns(2)
        img = Image.open(token_image)
        col_img.image(img, caption="Original Input", use_container_width=True)
        
        if st.button("Visualize Tokens"):
            with st.spinner("Extracting tokens from codebook..."):
                input_tensor = preprocess_image(img, resolution=resolution).to(device)
                with torch.no_grad():
                    # For tokenization, we don't use VAE latent sampling if not needed, 
                    # but here we use the encode method we checked.
                    encodings = model.encode(input_tensor, is_image=True)
                
                if isinstance(encodings, torch.Tensor):
                    viz = visualize_tokens(encodings, resolution=resolution)
                    col_tok.image(viz, caption="Token Map (Pseudo-color)", use_container_width=True)
                    
                    stats = get_token_statistics(encodings)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="stats-card"><b>Unique Tokens</b><br><span style="font-size:24px; color:#00d2ff">{stats["unique_tokens"]}</span></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="stats-card"><b>Most Active Code</b><br><span style="font-size:24px; color:#00d2ff">#{stats["most_frequent_token"]}</span></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="stats-card"><b>Token Entropy</b><br><span style="font-size:24px; color:#00d2ff">{stats["token_entropy"]:.2f}</span></div>', unsafe_allow_html=True)

with tab4:
    st.markdown(f"""
    ## {config.APP_NAME} Implementation Details
    
    This engine leverages **OmniTokenizer**, a joint image-video tokenizer that features:
    - **One weight set** for joint image and video tokenization.
    - **State-of-the-art** reconstruction on ImageNet and Kinetics datasets.
    - **High adaptability** to variable resolutions.
    
    ### How it Works:
    1. **Encoding**: The visual input is divided into patches and projected into a latent space.
    2. **Quantization**: These latents are mapped to the nearest entries in a learned **Codebook** (visual vocabulary).
    3. **Decoding**: The sequence of tokens is projected back into pixel space to reconstruct the image/video.
    
    ### Developer API:
    ```python
    from OmniTokenizer import OmniTokenizer_VQGAN
    model = OmniTokenizer_VQGAN.load_from_checkpoint("path/to/ckpt")
    tokens = model.encode(image_tensor)
    reconstruction = model.decode(tokens)
    ```
    """)

st.sidebar.divider()
st.sidebar.markdown("Made with ✨ by OmniVista Team")
