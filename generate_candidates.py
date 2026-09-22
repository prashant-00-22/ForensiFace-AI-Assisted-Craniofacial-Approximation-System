import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

# 1. Device check
assert torch.cuda.is_available(), "CUDA GPU nahi mila! Check environment."
device = "cuda"
print(f"[INFO] Running on GPU: {torch.cuda.get_device_name(0)}")

INPUT_SKULL = "input_skull.jpg"
OUTPUT_DIR = "generated_candidates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Extract anatomical edge constraints
def extract_skull_edges(image_path):
    if not os.path.exists(image_path):
        print(f"[WARN] '{image_path}' nahi mila! Generating synthetic sample skull outline...")
        synthetic = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(synthetic, (256, 220), 140, (255, 255, 255), 2)
        cv2.ellipse(synthetic, (256, 320), (80, 100), 0, 0, 180, (255, 255, 255), 2)
        cv2.circle(synthetic, (200, 200), 30, (255, 255, 255), 2)
        cv2.circle(synthetic, (312, 200), 30, (255, 255, 255), 2)
        cv2.imwrite("input_skull.jpg", synthetic)
        image_path = "input_skull.jpg"

    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 75, 175)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)

print("[INFO] Extracting skull anatomical constraints...")
skull_condition = extract_skull_edges(INPUT_SKULL)
skull_condition.save("skull_edges_preview.png")

# 3. Load models in FP16 for 4GB RTX 3050
print("[INFO] Loading diffusion models in FP16...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# 4. Forensic Anthropological Variation Profiles
hypotheses = [
    {"tag": "H1_Male_Lean", "prompt": "Forensic facial reconstruction ID photo of a 24-year-old male, lean facial tissue, sharp cheekbones, realistic skin texture, passport photo, front view"},
    {"tag": "H2_Male_Avg", "prompt": "Forensic facial reconstruction ID photo of a 28-year-old male, average soft tissue depth, natural skin tone, passport photo, front view"},
    {"tag": "H3_Male_Full", "prompt": "Forensic facial reconstruction ID photo of a 32-year-old male, fuller cheeks, standard tissue thickness, realistic studio lighting, front view"},
    {"tag": "H4_Female_Lean", "prompt": "Forensic facial reconstruction ID photo of a 25-year-old female, slender soft tissue, clear jawline, natural skin texture, front view"},
    {"tag": "H5_Female_Avg", "prompt": "Forensic facial reconstruction ID photo of a 29-year-old female, average soft tissue distribution, neutral forensic photo, front view"}
]

neg_prompt = "skull visible, bones, cartoon, 3d render, distorted eyes, blurry, low quality, high contrast shadows"

# 5. Generation
print(f"[INFO] Generating {len(hypotheses)} facial approximations on RTX 3050...")

for idx, hyp in enumerate(hypotheses):
    print(f"--> Processing [{idx+1}/{len(hypotheses)}]: {hyp['tag']}...")
    generator = torch.Generator(device="cuda").manual_seed(2026 + idx * 17)
    
    image = pipe(
        prompt=hyp["prompt"],
        negative_prompt=neg_prompt,
        image=skull_condition,
        num_inference_steps=18,
        controlnet_conditioning_scale=0.85,
        generator=generator
    ).images[0]
    
    out_file = os.path.join(OUTPUT_DIR, f"{hyp['tag']}.png")
    image.save(out_file)
    print(f"    Saved: {out_file}")

print(f"\n[SUCCESS] Completed! All candidate faces saved in: '{OUTPUT_DIR}'")