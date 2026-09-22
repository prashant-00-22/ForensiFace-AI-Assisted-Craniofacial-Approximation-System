# ForensiFace — AI-Assisted Craniofacial Approximation System

> AI-assisted forensic lead-generation prototype for generating multiple plausible facial approximations from cranial skeletal structure using structural image conditioning and diffusion models.

ForensiFace is an experimental computer-vision and generative-AI system designed to assist forensic investigation workflows involving unidentified skeletal remains.

Instead of producing a single deterministic facial reconstruction, ForensiFace generates multiple plausible facial hypotheses while conditioning the generation process on structural information extracted from the input skull.

---

## ⚠️ Forensic Disclaimer

> **IMPORTANT INVESTIGATIVE LEAD NOTICE**
>
> ForensiFace generates synthetic facial approximations intended only to assist forensic teams and investigators in exploring potential missing-person candidate galleries.
>
> **The generated faces are NOT identity matches and must not be treated as legally verified identification.**
>
> Final identification must be established through appropriate forensic standards such as DNA profiling, comparative dental records, fingerprint analysis, or other validated forensic procedures.

---

## 🚀 Overview

Traditional craniofacial reconstruction can involve manual sculpting or deterministic facial approximation techniques. These approaches may produce a single facial hypothesis even though soft-tissue characteristics cannot be uniquely determined from skeletal structure alone.

ForensiFace treats craniofacial approximation as a **multi-hypothesis synthesis task**.

The system:

1. Accepts a frontal skull image or cranial radiograph.
2. Extracts structural skeletal contours.
3. Reduces unwanted internal cavity artifacts using morphological filtering.
4. Converts structural information into a ControlNet spatial conditioning map.
5. Combines skeletal constraints with anthropological parameters.
6. Uses Stable Diffusion and ControlNet to generate multiple plausible facial hypotheses.
7. Displays the generated approximations through an interactive Streamlit dashboard.

---

## 🎯 Problem Statement

When unidentified skeletal remains are discovered, investigators may have limited visual information about the individual's appearance.

Several challenges arise:

- Soft tissue is no longer available from skeletal remains.
- A single deterministic reconstruction can introduce anchoring bias.
- Facial appearance cannot be uniquely determined from skull geometry alone.
- Different soft-tissue assumptions can produce different plausible appearances.
- Manual reconstruction can be time-consuming and difficult to reproduce consistently.

Therefore, the objective of ForensiFace is **not to determine a person's definitive identity**, but to create multiple visually plausible investigative hypotheses that may assist downstream forensic investigation workflows.

---

## 💡 Proposed Solution

ForensiFace combines structural computer vision with conditional image generation.

### 1. Structural Constraint Extraction

The input skull image undergoes a preprocessing pipeline consisting of:

- **Bilateral Filtering** — reduces image noise while preserving important structural boundaries.
- **Canny Edge Detection** — extracts prominent skeletal boundaries.
- **Morphological Closing** — bridges small discontinuities and helps reduce unwanted internal cavity artifacts.

### 2. Conditional Generation

The extracted structural edge map is passed to a ControlNet-conditioned Stable Diffusion pipeline.

Generation is guided by:

- Skeletal structural boundary map
- Estimated age bracket
- Estimated sex
- Soft-tissue variation profile

### 3. Multi-Hypothesis Output

Instead of producing a single facial approximation, the system can generate multiple candidate hypotheses using different soft-tissue assumptions, including:

- Lean Soft Tissue
- Average Tissue Depth
- Full Soft Tissue
- Athletic Tissue Build

---

## ✨ Key Features

### 🦴 Morphological Contour Extraction

Extracts structural skull boundaries using image-processing operations while suppressing unwanted internal cavity information.

### 🧬 Multi-Hypothesis Facial Synthesis

Generates multiple facial approximations from the same skeletal input using different soft-tissue profiles.

### 🎛️ Anthropological Parameter Control

The dashboard provides controls for:

- Estimated Sex
- Estimated Age Bracket
- Number of Candidates
- Bone Constraint Weight

### 🖥️ Interactive Streamlit Dashboard

Provides an interactive interface for:

- Skull image upload
- Parameter configuration
- Structural contour visualization
- Candidate generation
- Side-by-side hypothesis comparison

### ⚡ GPU-Optimized Inference

Designed to operate on consumer GPUs with approximately 4 GB VRAM using:

- FP16 inference
- Attention slicing
- VAE slicing
- Explicit GPU memory cleanup

### 🔬 Investigative Lead Generation

The system is designed as an investigative hypothesis-generation prototype rather than a biometric identification system.

---

## 🧠 How It Works

```text
                  ┌──────────────────────────┐
                  │   Skull / Cranial Image  │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Image Preprocessing    │
                  │                          │
                  │ • Bilateral Filtering    │
                  │ • Canny Edge Detection   │
                  │ • Morphological Closing  │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Structural Edge Map    │
                  └────────────┬─────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  Anthropological Parameters    │
              │                                │
              │ • Estimated Sex                │
              │ • Estimated Age Bracket        │
              │ • Soft-Tissue Profile          │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  CLIP Text Encoder │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ ControlNet + Stable  │
                    │ Diffusion v1.5       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ DPMSolver Scheduler  │
                    │ FP16 GPU Inference   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Multiple Facial      │
                    │ Hypotheses           │
                    └──────────────────────┘
🏗️ System Architecture
🔄 End-to-End Workflow
🛠️ Technology Stack
Category	Technology / Package	Purpose
Programming Language	Python 3.11	Core runtime
Deep Learning	PyTorch 2.5.1	Tensor computation and GPU acceleration
CUDA	CUDA 12.1	NVIDIA GPU acceleration
Generative AI	Hugging Face Diffusers	Diffusion pipeline
Base Model	Stable Diffusion v1.5	Image generation
Conditioning	ControlNet Canny	Structural conditioning
Text Conditioning	CLIP	Prompt conditioning
Computer Vision	OpenCV	Filtering, edges and morphology
Image Processing	Pillow	Image loading and processing
Web Interface	Streamlit	Interactive dashboard
GPU	NVIDIA RTX 3050 4GB	Local inference
🤖 AI/ML Pipeline
1. Input Processing

The system accepts a frontal skull image, cranial radiograph, or 2D skeletal projection.

Example:

input_skull.jpg
2. Image Preprocessing

The input image passes through several preprocessing operations.

Bilateral Filtering

Reduces image noise while preserving important structural boundaries.

smooth = cv2.bilateralFilter(
    gray,
    9,
    75,
    75
)
Canny Edge Detection

Extracts prominent skeletal boundaries.

edges = cv2.Canny(
    smooth,
    60,
    160
)
Morphological Closing

Helps bridge small gaps and reduce unwanted cavity artifacts.

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

cleaned_edges = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel
)
🧠 Generative Model Pipeline

ForensiFace uses pretrained diffusion foundations rather than training a new diffusion model from scratch.

Base Model
runwayml/stable-diffusion-v1-5
Structural Conditioning
lllyasviel/sd-controlnet-canny

The extracted skull contours act as spatial conditioning information for the generative pipeline.

Text Conditioning

Anthropological parameters are incorporated into the generation prompt.

Parameters include:

Estimated Sex
Estimated Age Bracket
Soft-Tissue Profile
Sampling

The system uses:

DPMSolverMultistepScheduler

with the configured inference step count and FP16 inference.

📊 Dataset & Pretrained Foundations

ForensiFace relies on pretrained generative and conditioning models rather than a project-specific diffusion model trained from scratch on a proprietary forensic dataset.

Generative Backbone
runwayml/stable-diffusion-v1-5
Structural Conditioning
lllyasviel/sd-controlnet-canny
Inference Inputs

The prototype uses frontal cranial images, 2D skeletal projections, and anatomical test images during development and inference.

Important: Generated faces should be understood as plausible generative hypotheses guided by structural information, not as medically or biometrically validated reconstructions.

📁 Project Structure
ForensiFace-AI-Assisted-Craniofacial-Approximation-System/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── app.py
├── generate_candidates.py
│
├── input_skull.jpg
├── skull_edges_preview.png
├── forensic_summary_grid.png
│
├── screenshots/
│   └── forensiface-dashboard.png
│
└── generated_candidates/
    └── ...
⚙️ Requirements
Software
Windows 10/11 64-bit or Linux
Python 3.11
NVIDIA GPU driver compatible with CUDA 12.1+
Git
Reference Hardware

Development and testing environment:

GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 4 GB
RAM: 16 GB

The implementation includes memory-management optimizations intended to make local inference practical within a 4 GB VRAM environment.

🔧 Installation
1. Clone the Repository
git clone <YOUR_REPOSITORY_URL>
cd ForensiFace-AI-Assisted-Craniofacial-Approximation-System
2. Create a Virtual Environment
Windows
py -3.11 -m venv venv
.\venv\Scripts\activate
Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
3. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
4. Install Project Dependencies
pip install -r requirements.txt
5. Verify GPU Support
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

Expected output:

CUDA Available: True
GPU: NVIDIA GeForce RTX 3050
🔐 Environment Variables

No paid third-party API is required for the basic local inference workflow.

An optional Hugging Face token can be configured to reduce model-download rate-limit issues.

Create:

.env

Then add:

HF_TOKEN=your_huggingface_read_token_here

Never commit real API keys or authentication tokens to GitHub.

▶️ Running the Application
Option 1 — Streamlit Dashboard

Start the interactive application:

streamlit run app.py

Then open:

http://localhost:8501
Option 2 — Standalone Candidate Generation

Run:

python generate_candidates.py

Generated candidate outputs are stored in:

generated_candidates/
🎛️ Using the Dashboard
Step 1 — Configure Parameters

Configure:

Estimated Sex
Estimated Age Bracket
Number of Candidates
Bone Constraint Weight
Step 2 — Upload Skull Image

Upload a clean frontal skull image or cranial radiograph.

Supported image formats include:

.jpg
.png
Step 3 — Inspect Structural Constraints

The application displays the processed structural contour map.

This allows the user to inspect the structural information being passed to the generative pipeline.

Step 4 — Generate Candidates

Click:

Generate Face Candidates
Step 5 — Compare Hypotheses

The generated facial approximations are displayed side-by-side.

Different hypotheses can represent different soft-tissue assumptions.

📸 Results & Dashboard Preview
Interactive Web Application

The following screenshot demonstrates the complete workflow:

Anthropological parameter configuration
Skull input
Structural contour extraction
Candidate generation
Multiple facial hypotheses

Example Output Pipeline
             Input Skull Image
                    │
                    ▼
          Structural Contour Map
                    │
                    ▼
          ControlNet Conditioning
                    │
                    ▼
        Diffusion-based Generation
                    │
             ┌──────┴──────┐
             ▼             ▼
        Hypothesis 1   Hypothesis 2
        Lean Tissue    Average Tissue
Example Project Outputs
Pipeline Stage	File
Input Skull	input_skull.jpg
Structural Constraints	skull_edges_preview.png
Composite Output	forensic_summary_grid.png
Generated Candidates	generated_candidates/
⚡ Performance & Memory Optimization

Running diffusion pipelines on a 4 GB VRAM GPU introduces significant memory constraints.

ForensiFace implements several optimizations.

FP16 Inference

The pipeline uses:

torch.float16

to reduce GPU memory usage.

Attention Slicing
pipe.enable_attention_slicing(slice_size=1)

This reduces peak attention-memory requirements.

VAE Slicing

VAE decoding uses slicing to reduce memory spikes during image generation.

Direct GPU Placement

The pipeline is loaded directly onto:

cuda:0

instead of relying on CPU offloading mechanisms that previously caused meta tensor issues.

Explicit Memory Cleanup

Between candidate-generation iterations:

gc.collect()
torch.cuda.empty_cache()

are used to reduce memory accumulation.

📈 Empirical Observations

Testing was performed on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM.

Inference Time

Inference time depends on:

GPU utilization
Background processes
Model loading state
Number of candidates
Inference configuration

During local testing with the configured diffusion pipeline, generation was observed to take several seconds per candidate after model initialization.

Memory Stability

The implemented memory optimizations allow the pipeline to operate within the 4 GB hardware environment while reducing the risk of GPU out-of-memory interruptions during multi-candidate generation.

Structural Adherence

Generated outputs were observed to retain structural relationships associated with the input edge map, including:

Mandibular boundaries
Eye-region positioning
Nasal-base positioning

These observations are experimental and should not be interpreted as forensic validation metrics.

🧩 Challenges & Engineering Solutions
Challenge	Cause	Solution
Hollow / ghost-like facial artifacts	Internal skull cavities detected during edge extraction	Bilateral filtering + morphological closing
Excessive GPU memory usage	Diffusion model memory requirements	FP16 + attention slicing + VAE slicing
Tensor on device meta error	CPU offloading created placeholder tensors	Direct CUDA pipeline loading
Python C-extension conflicts	Dependency and Python-version compatibility issues	Standardized environment on Python 3.11
Internal eye/teeth edges influencing generation	Raw Canny output contained cavity structures	Morphological preprocessing
⚠️ Limitations

ForensiFace is an experimental research prototype and has several limitations.

1. 2D Input Constraint

The current implementation primarily works with frontal 2D skull/radiographic inputs.

2. Missing or Damaged Skeletal Elements

Significant trauma, asymmetry, or missing skeletal regions may reduce the quality of structural conditioning.

3. Non-Deducible Appearance Features

Features such as:

Hair style
Eye color
Facial hair
Specific skin characteristics
Other soft-tissue details

cannot be reliably determined from skeletal structure alone.

The diffusion model therefore introduces learned generative priors.

4. Output Resolution

The current pipeline primarily generates images at:

512 × 512

to remain practical on a 4 GB VRAM environment.

5. No Identity Verification

The system does not currently perform validated biometric identification.

Generated faces are hypotheses, not identity matches.

🔮 Future Scope
3D CT-Based Reconstruction

Future versions could process DICOM/3D CT volume data to generate:

Frontal views
Profile views
45° oblique views
Experimental Candidate Ranking

Future research could investigate facial embedding models such as:

ArcFace
FaceNet

for experimental comparison of generated hypotheses against controlled candidate galleries.

This functionality is not part of the current implementation.

Soft-Tissue Landmark Calibration

Future versions could incorporate empirical tissue-depth landmark datasets for more controlled soft-tissue parameterization.

Improved Anatomical Conditioning

Future work could investigate stronger landmark-level conditioning alongside dense edge maps for improved morphological control.

🧪 Experimental Nature

ForensiFace should be understood as:

Research Prototype
        +
Computer Vision Pipeline
        +
Generative AI
        +
Investigative Hypothesis Generation

It is not a replacement for:

DNA analysis
Fingerprint identification
Dental identification
Professional forensic reconstruction
Validated biometric identification systems
🔒 Security & Data Responsibility

When working with real forensic material:

Do not upload sensitive forensic images to public repositories.
Do not commit personally identifiable information.
Do not commit confidential investigation records.
Never commit authentication tokens or API credentials.
Use synthetic or legally authorized test data during development.
Follow applicable institutional, legal, and forensic data-handling requirements.
🤝 Contributing

Contributions are welcome, particularly around:

Image preprocessing
Anatomical conditioning
Memory optimization
3D reconstruction
Soft-tissue modeling
Evaluation methodologies
Documentation
Contribution Workflow
git checkout -b feature/your-feature

git add .

git commit -m "feat: describe your change"

git push origin feature/your-feature

Then open a Pull Request.

👨‍💻 Author
Prashant

Computer Science & Engineering — Artificial Intelligence & Machine Learning

GitHub: prashant-00-22

📄 License

This project is distributed under the MIT License.

See the LICENSE file in the repository for the complete license terms.

📚 References
Stable Diffusion

Rombach et al.

High-Resolution Image Synthesis with Latent Diffusion Models

CVPR 2022.

ControlNet

Zhang & Agrawala.

Adding Conditional Control to Text-to-Image Diffusion Models

ICCV 2023.

Hugging Face Diffusers

The project uses the Diffusers ecosystem for diffusion-model pipeline implementation and inference.

Forensic Reconstruction Research

The project is conceptually inspired by research into craniofacial reconstruction and multi-hypothesis approaches in computational forensic anthropology.

⭐ Project Summary

ForensiFace explores how computer vision and conditional diffusion models can be combined to transform skeletal structural information into multiple plausible facial hypotheses.

The core pipeline is:

Skull Image
     ↓
Bilateral Filtering
     ↓
Canny Edge Detection
     ↓
Morphological Closing
     ↓
Structural Constraint Map
     ↓
ControlNet
     ↓
Stable Diffusion
     ↓
FP16 GPU Inference
     ↓
Multiple Facial Hypotheses

ForensiFace is an experimental AI research prototype for investigative hypothesis generation, not an identity-verification system.
