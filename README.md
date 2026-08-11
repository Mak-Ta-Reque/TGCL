# XL-VLMS: Multimodal Concept Feature Extraction

## Overview
This project provides tools for extracting, analyzing, and decomposing concept features from multimodal data (images and text) using large vision-language models. It includes scripts for feature generation, decomposition, and analysis, as well as a demo notebook for binary task explanation.

## Demo Output

### Token-wise Concept Grounding Visualization

Given a grid of 2x2 images (total 4 different images), the model is tasked to predict the object in each grid cell. For each predicted token, the corresponding residual stream is mapped to relevant concepts. The demo outputs are provided as image files in the `grounding_per_token` directory.

#### Example Visualization

![Token-wise Concept Grounding](readme_samples/images/per_token_viz_0.png)

#### What This Visualization Shows

This visualization demonstrates **how the Vision-Language Model (VLM) explains its predictions** by breaking down each token in its output and showing which visual concepts contribute to that token's generation.

**Layout Structure:**

1. **Top Section - Input & Prediction:**
   - **Prompt**: The question asked to the model (e.g., "What are items in each grid?")
   - **Input Image Grid**: A 2×2 grid showing 4 different images (in this example: fruit baskets, market scenes, and kitchen interiors)
   - **Model Prediction**: The VLM's complete response shown in a highlighted box (e.g., "fruits and vegetables fruits and vegetables kitchen fruits and vegetables fruits and vegetables")

2. **Bottom Section - Token-wise Explanations:**
   Each row explains **one token** from the prediction:
   - **Token Label** (left): The specific token being explained (e.g., "fr", "uits", "and", "vegetables", "kitchen")
   - **Concept Similarity Bar** (middle): A vertical bar chart showing:
     - **Concept Name**: The semantic concept associated with this token (e.g., "fruit, apple, based", "refrigerator, freezer", "vegetable")
     - **Similarity Score**: A numerical value (0.0-1.0) indicating how strongly this concept contributes to the token's generation
     - The bar extends upward from the bottom, with the score displayed at the base
   - **Visual Examples** (right): 5-6 small image crops showing real examples of visual elements that activate this concept

#### What This Explanation Means

This visualization provides **interpretability** for the VLM's decision-making process:

- **Token-level transparency**: You can see exactly which concepts the model uses when generating each word
- **Concept grounding**: Each concept is grounded in actual visual examples (the image crops), showing what the model "sees" when it thinks about that concept
- **Similarity scores**: Higher scores indicate stronger activation - concepts with scores >0.3 are typically the primary drivers for that token
- **Multi-concept activation**: A single token (like "fruits") may activate multiple related concepts (fruit, apple, food, etc.), showing the model's rich semantic understanding

**Example Interpretation:**
- When the model generates "fruits", it primarily activates concepts like "fruit, apple, based" (score: 0.394) and "food, based" (score: 0.381)
- The visual crops show actual fruit images (apples, bananas, grapes) that the model associates with this concept
- This demonstrates that the model isn't just pattern-matching text, but genuinely connecting visual features to semantic concepts

This type of explanation is crucial for **debugging model behavior**, **understanding failures**, and **validating that the model is using appropriate visual reasoning** rather than spurious correlations.


#### Additional Example Visualizations:

![Group 11 Example](readme_samples/images/group_11.png)

![Group 16 Example](readme_samples/images/group_16.png)

![Group 17 Example](readme_samples/images/group_17.png)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd xl-vlms-rsml
   ```

2. **Set up Python environment**
   - Recommended: Python 3.10
   - You can use Conda or a virtualenv:
     ```bash
     conda create -n xlvlms python=3.10
     conda activate xlvlms
     # or
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**
   - Using pip:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
     pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 # for CUDA 12.5 please use your device configuration
     pip install tqdm git+https://github.com/bckim92/language-evaluation.git psutil spacy timm accelerate
     python -m spacy download en_core_web_sm

     # For BERTScore and CLIPScore evaluation:
     pip install bert-score
     pip install git+https://github.com/openai/CLIP.git

     # For Qwen model support:
     pip install qwen-vl-utils
     conda install -c conda-forge inflect
     conda install -c conda-forge scikit-learn
     # For Java support (if needed):
     conda install -c conda-forge openjdk
     # Download COCO evaluation data:
     python -c "import language_evaluation; language_evaluation.download('coco')"

     pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
     pip install pycocotools
     pip install matplotlib
     pip install streamlit
     ```

## 🚀 Getting Started

**[`docs/cgdl_quickstart.md`](docs/cgdl_quickstart.md) is the canonical guide** —
start there. It covers configuring `.env`, building the dataset, running the
pipeline with the `cgdl` contrastive prompt template, and evaluating a run
(faithfulness insertion/deletion AUC, grounding BERTScore/CLIPScore, and
concept-identity object-classification F1).

For the full multi-config ablation matrix (`PROMPT_TEMPLATE` × `CROP_MODE` ×
`DECOMP_STRATEGY`) and exactly what each metric measures and where it's
computed, see [`docs/coco10_ablation_methods.md`](docs/coco10_ablation_methods.md).

## 📂 Folder Structure

- `src/` : Main source code (models, datasets, metrics, helpers, analysis)
- `scripts/` : Pipeline entry points, feature generation/decomposition, ablation drivers
- `preprocessing/` : Dataset building, crop generation, concept-image mapping
- `inference/` : Dataset inference and VLM explainer scripts
- `eval/` : Faithfulness, grounding, and object-classification evaluation scripts
- `docs/` : `cgdl_quickstart.md` (start here), `coco10_ablation_methods.md`, and `docs/notes/` (older design notes/plans)
- `notebooks/` : Demo and visualization notebooks
- `demo/` : Streamlit demo app
- `legacy/` : Superseded scripts kept for reference only — see `legacy/README.md`
- `tests/` : Unit/integration tests

For questions or issues, please refer to the repository or contact the maintainers.

**Contact:** abdul.kadir@dfki.de
