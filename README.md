# Early Dysgraphia Screening from Handwriting Images

An end-to-end machine learning pipeline designed to assist educators and therapists in the early screening of dysgraphia using handwriting analysis. 

## The Project Journey (CRISP-DM in Action)
This project represents a real-world machine learning iteration loop:
1. **Phase 1 (Baseline):** We initially trained a baseline CNN and MobileNetV2 transfer model on a small, expert-labeled dataset of 249 images. The evaluation proved the models were overfitting and collapsing (Validation AUC: 0.45).
2. **Phase 2 (Field Data Collection):** To address the data bottleneck, the team collected real-world handwriting samples from a local dyslexia support school (Rare Gem Talent School). We gathered consent, anonymized student data, and processed full notebook pages into line-level strips.
3. **Phase 3 (Iteration & Scaling):** By combining the original clinical data with our field-collected data (~2,600+ line samples), we retrained the transfer models. Validation AUC jumped to **0.80+**, and our latest Colab runs on the full expanded dataset are reaching **0.91+ AUC**, demonstrating that data scale and quality were the true bottlenecks.

## Repository Structure
* **`01_data_exploration.ipynb` / `02_data_quality.ipynb`**: Data loading, quality checks, and auditing of the original clinical dataset.
* **`03_data_preparation.ipynb`**: Preprocessing pipeline (grayscale, deskew, aspect-ratio padding, normalization, and augmentation).
* **`04_Modeling.ipynb` / `04_Modelingv2.ipynb`**: Training the baseline CNN and MobileNetV2 transfer models (v1 on small data, v2 on combined large data).
* **`05_evaluation.ipynb`**: Honest evaluation, threshold sweeps, ROC/PR curves, and Grad-CAM visualizations.
* **`src/`**: Modular Python scripts for evaluation (`evaluate.py`) and interpretability (`gradcam.py`).
* **`models/`**: Saved `.keras` weights for the trained models.
* **`data/`**: Directory structure for local dataset storage.

## Setup & Local Execution
To reproduce the notebooks locally, use the following steps:

1. Create the environment:
   ```bash
   conda create -n group python=3.11 -y
   conda activate group

bash
      pip install tensorflow pandas scikit-learn matplotlib opencv-python pillow
# Team & Roles
Sammy - Project Lead & Report Assembly
Ahmed - Evaluation, Honest Verdict Reporting
Eric - Baseline & Transfer Modeling, Hyperparameter Tuning
Emmanuel - Data Preprocessing Pipeline
Lucy - Data Quality Checks & Auditing
Dax - Deployment (Streamlit UI)
Sam - Interpretable Indicators (Spacing & Baseline)

# Current Status & Next Steps
The combined dataset model is performing well (0.80 - 0.91 Validation AUC).
Next steps: Re-validating splits at the student-level (to prevent data leakage from lines of the same child), performing a domain-breakdown evaluation (Malaysian vs. Kenyan test sets separately), and finalizing the Streamlit deployment UI.

