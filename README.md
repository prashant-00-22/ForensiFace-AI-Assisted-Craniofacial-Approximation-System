# ForensiFace: AI-Assisted Craniofacial Approximation System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA12.1-red.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-orange.svg)](https://github.com/huggingface/diffusers)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An AI-powered forensic investigation tool that synthesizes anatomically constrained, multi-hypothesis facial approximations from damaged skeletal remains and cranial radiographs to generate investigative candidate leads.**

---

## 📌 Abstract & Overview

Traditional forensic facial reconstruction relies heavily on deterministic 3D clay modeling or static mesh deformation, typically yielding only a single face that fails to capture natural soft-tissue thickness variance across diverse populations. 

**ForensiFace** implements a multi-hypothesis paradigm inspired by recent state-of-the-art craniofacial diffusion research (such as *ICCR-Diff*). Rather than claiming a single definitive identification, the system locks rigid skeletal landmarks (orbital margins, piriform aperture, mandibular angles) using morphological edge constraints, then leverages condition-guided diffusion to generate multiple plausible facial approximations conditioned on anthropological parameters (estimated age bracket, sex, and soft-tissue depth).

---

## 🚀 Key Features

* **Rigid Anatomical Boundary Lock:** Employs bilateral and morphological filtering alongside Canny operators to suppress internal cranial cavity noise while strictly retaining skeletal structure.
* **Conditioned Multi-Hypothesis Generation:** Generates diverse candidate portraits varying across subcutaneous fat distribution, cheek morphology, and jaw fullness under a locked skull framework.
* **Interactive Investigator Portal:** A web dashboard built on Streamlit for real-time radiograph uploads, parameter modulation, and immediate side-by-side hypothesis review.
* **Hardware Optimized:** Tailored for consumer-tier GPUs (tested on NVIDIA GeForce RTX 3050 4GB VRAM) using FP16 precision, attention slicing, and memory caching without performance degradation.

---

## 🏗️ System Architecture

```text
       +---------------------------------------------+
       |   Cranial Radiograph / 3D Skull Projection  |
       +---------------------------------------------+
                              │
                              ▼
       +---------------------------------------------+
       |       Morphological Feature Extraction       |
       |  (Bilateral Smoothing + Canny Edge Filter)  |
       +---------------------------------------------+
                              │
               [Structural Bone Constraints]
                              │
                              ▼
       +---------------------------------------------+
       |      Condition-Guided Diffusion Engine      |
       |  (ControlNet + Stable Diffusion v1.5 FP16)  |
       +──────────────────────┬──────────────────────+
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     [Hypothesis #1]   [Hypothesis #2]   [Hypothesis #3]
       Lean Tissue      Average Tissue     Full Tissue
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
       +---------------------------------------------+
       |       Investigative Candidate Gallery       |
       |       (Lead Generation for Review)          |
       +---------------------------------------------+
🛠️ Technology Stack
Language: Python 3.11

Deep Learning Framework: PyTorch 2.5.1 + CUDA 12.1

Generative Pipeline: Hugging Face diffusers (Stable Diffusion 1.5, ControlNet Canny)

Image Processing: OpenCV (opencv-python), Pillow, NumPy

Frontend / UI: Streamlit

📦 Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/prashant-00-22/ForensiFace-AI-Assisted-Craniofacial-Approximation-System.git](https://github.com/prashant-00-22/ForensiFace-AI-Assisted-Craniofacial-Approximation-System.git)
cd ForensiFace-AI-Assisted-Craniofacial-Approximation-System
2. Create and Activate Virtual Environment
Windows (PowerShell):

PowerShell
py -3.11 -m venv venv
.\venv\Scripts\activate
Linux / macOS:

Bash
python3.11 -m venv venv
source venv/bin/activate
3. Install PyTorch with CUDA Support
Bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
4. Install Dependencies
Bash
pip install -r requirements.txt
🖥️ Running the Application
Launch the Streamlit dashboard:

Bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501.

Upload a cranial frontal radiograph or skull image (.jpg, .png).

Set the anthropological parameters (estimated sex, age bracket, tissue thickness variation).

Click "Generate Face Candidates" to produce multiple plausible facial leads.

🔬 Forensic Disclaimer
Investigative Lead Notice: Outputs generated by ForensiFace are synthetic approximations intended exclusively to assist investigators in narrowing down potential missing persons candidate galleries. They do not constitute legally verified identification. Final positive identification must always be verified through secondary forensic standards (DNA profiling, odontological records, or fingerprint comparison).

📜 Citation & Acknowledgements
This implementation is inspired by principles established in deep craniofacial reconstruction research:

ICCR-Diff: Zhang et al., Identity-Preserving and Controllable Craniofacial Reconstruction with Diffusion Models, Knowledge-Based Systems (2025).

ControlNet: Zhang & Agrawala, Adding Conditional Control to Text-to-Image Diffusion Models, ICCV (2023).