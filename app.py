import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

st.set_page_config(page_title="ForensiFace | Craniofacial Approximation", layout="wide")

st.title("ForensiFace: AI-Assisted Craniofacial Approximation System")
st.caption("Morphological Face Reconstruction from Skeletal Remains using Anatomically Constrained Diffusion")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def load_pipeline():
    cleanup_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 1. Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype
    )
    
    # 2. Load Base Diffusion
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # CRITICAL FIX: Direct GPU placement without accelerate offload hooks
    # This completely eliminates the "Tensor on device meta" error
    if device == "cuda":
        pipe.to("cuda")
        pipe.enable_attention_slicing(slice_size=1)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    else:
        pipe.to("cpu")
        
    return pipe

pipe = load_pipeline()

# Morphological contour cleanup to remove internal hollow bone textures
def extract_clean_skull_contours(pil_img):
    img = np.array(pil_img)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(smooth, 60, 160)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)

# Sidebar UI
st.sidebar.header("Forensic Parameters")
sex = st.sidebar.selectbox("Anthropological Sex Estimate", ["Male", "Female", "Undetermined"])
age_range = st.sidebar.slider("Estimated Age Bracket", 18, 65, (22, 32))
num_hypotheses = st.sidebar.slider("Number of Candidates", 1, 4, 2)
conditioning_scale = st.sidebar.slider("Bone Constraint Weight", 0.40, 0.85, 0.55, 0.05,
                                      help="0.55 provides balanced facial geometry while avoiding hollow bone textures.")

uploaded_file = st.sidebar.file_uploader("Upload Skull / Radiograph", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns([1, 2])

if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")
    edge_constraint = extract_clean_skull_contours(input_img)
    
    with col1:
        st.subheader("Skeletal Input & Contours")
        st.image(input_img, caption="Input Cranial Radiograph / Skull", width=300)
        st.image(edge_constraint, caption="Filtered Structural Bone Constraints", width=300)
    
    with col2:
        st.subheader("Plausible Facial Approximations")
        if st.button("Generate Face Candidates", type="primary"):
            progress_bar = st.progress(0)
            
            variations = [
                ("Lean Soft Tissue", "lean facial build, defined cheekbones, thin tissue layer"),
                ("Average Tissue Depth", "average subcutaneous tissue thickness, balanced proportions"),
                ("Full Soft Tissue", "full cheeks, rounded soft jawline, standard facial fullness"),
                ("Athletic Tissue Build", "firm facial tissue, clear jaw definition, athletic build")
            ]
            
            cols = st.columns(num_hypotheses)
            
            for i in range(num_hypotheses):
                label, tissue_desc = variations[i % len(variations)]
                gender_term = "individual" if sex == "Undetermined" else sex.lower()
                
                prompt = (
                    f"A clear front-view identification color passport portrait of a real {age_range[0]}-{age_range[1]} year old {gender_term}, "
                    f"{tissue_desc}, normal human face, realistic skin texture with natural pores, neutral expression, "
                    f"eyes looking forward, mouth closed, normal human eyes and nose, studio headshot lighting, highly detailed photography"
                )
                
                neg_prompt = (
                    "skull, skeleton, bone, teeth showing, hollow eyes, dark hollow eye sockets, "
                    "corpse, deformed, cartoon, 3d render, blurry, bad anatomy, painting"
                )
                
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(2026 + i * 47)
                
                with torch.inference_mode():
                    face = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=edge_constraint,
                        num_inference_steps=18,
                        controlnet_conditioning_scale=conditioning_scale,
                        generator=generator
                    ).images[0]
                
                with cols[i]:
                    st.image(face, caption=f"Hypothesis #{i+1}\n({label})", width=240)
                
                cleanup_memory()
                progress_bar.progress((i + 1) / num_hypotheses)
            
            st.success("Candidate hypotheses generated successfully on GPU!")
else:
    st.info("Please upload a skull image from the sidebar to begin.")