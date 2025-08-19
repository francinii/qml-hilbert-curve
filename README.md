# 🧠 QML-Hilbert-Curve: Hybrid Quantum-Classical Brain Tumor Classifier

---

## 📖 Project Overview

This project implements a **binary classifier for brain tumors** using a hybrid quantum-classical model. The main research goal is to evaluate the potential of **Hilbert curves** for image preprocessing, aiming to preserve more spatial information from medical images before they are processed by the hybrid model.

> ⚠️ **Note:** Hilbert curve preprocessing is a planned feature and not yet fully implemented in the current version.

---

## 📂 Dataset

The dataset used for this project is a public brain MRI dataset for tumor classification.
- **Source:** [Brain MRI Dataset on Kaggle](https://www.kaggle.com/datasets/pradeep2665/brain-mri/code)
- **Image size:** Images are loaded at 512x512 pixels for training,for model input and preprocessing.

---

## 🏗️ Hybrid Model Architecture

- **Input preprocessing:**
  - Images loaded at 512x512, converted to grayscale.
  - Flattened and normalized to [0, 1].
  - StandardScaler applied.
  - PCA applied (12 components).
  - MinMaxScaler applied to map features to [0, π/2] for quantum embedding.

- **Hybrid Model:**
  - **Classical Layer 1:** Linear (12, 12)
  - **Quantum Layers:** Two Pennylane TorchLayers, each operating on half the features (6 qubits each, total 12 qubits). Each quantum layer uses AngleEmbedding and BasicEntanglerLayers.
  - **Classical Layer 2:** Linear (12, 12)
  - **Output Layer:** Linear (12, 1)
  - **Activation functions:** No explicit ReLU; the model is mostly linear except for quantum nonlinearity.
  - **Optimizer:** SGD
  - **Loss function:** BCEWithLogitsLoss (Binary Cross-Entropy with logits)
  - **Training epochs:** Typically 200
  - **Batch size:** 64
  - **Best accuracy:** Up to ~76% (varies by run)
  - **Quantum device:** `default.qubit` (inference) and `lightning.qubit` (training)

---

## ⚙️ How to Run the Project

### 1️⃣ Create and Activate a Virtual Environment

**On Windows:**
```sh
python -m venv venv
venv\Scripts\activate
```
**On Mac/Linux:**
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Requirements

```sh
pip install -r requirements.txt
```

### 📥 Dataset Setup

Get the data from the Kaggle dataset:  
[Brain MRI Dataset on Kaggle](https://www.kaggle.com/datasets/pradeep2665/brain-mri/code)

Place the extracted folders in your project as follows:

```
app/
  data/
    dataset_binary/
      Training/
        tumor/
        no_tumor/
      Testing/
        tumor/
        no_tumor/
      Validation/
        tumor/
        no_tumor/
```

- **Each of the `Training`, `Testing`, and `Validation` folders must contain two subfolders:**
  - `tumor/` (images with tumors)
  - `no_tumor/` (images without tumors)
- You can use the script [`app/scripts/create_binary_dataset.py`](app/scripts/create_binary_dataset.py) to help organize the dataset into this required structure automatically.

### 3️⃣ Run the Model or Notebooks

- To run a Jupyter notebook for training or analysis:
  ```sh
  cd app/notebooks
  jupyter notebook
  ```
  Open the desired notebook (e.g., `binary_classification_last.ipynb`).

- To run the Streamlit app for interactive demo:
  ```sh
  cd app/notebooks
  streamlit run app.py
  ```

### 4️⃣ Using Pretrained Models

- Pretrained models (`.pkl`, `.pt`) are stored in `app/models/`.
- The code will automatically select the available model files for inference.

---

## 🗂️ Project Structure

```
app/
  models/         # Pretrained models (.pkl, .pt)
  notebooks/      # Jupyter notebooks and Streamlit app
  scripts/        # Data processing and Hilbert curve scripts
  results/        # Results, plots, and logs
  data/           # (If present) Raw or processed data
```

---

## 🧪 Methodology Summary

- **Preprocessing:** Images are loaded at 512x512, converted to grayscale, normalized, reduced with PCA, and scaled for quantum embedding.
- **(Planned) Hilbert Curve:** Future versions will use Hilbert curve mapping to preserve spatial information before feature extraction.
- **Hybrid Model:** Combines classical and quantum layers for robust binary classification.


---

## 🚀 Future Work

- [ ] Full implementation of Hilbert curve preprocessing
- [ ] Training and evaluation with Hilbert-processed images
- [ ] Comparative analysis of classical vs. quantum vs. hybrid models

---

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

**For research and educational purposes only. Not for clinical use.**
