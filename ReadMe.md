# 🐱 Deepfake Cat Detector

A PyTorch-based deep learning project to detect whether cat images are **real** or **AI-generated deepfakes**.

---

## 📌 Project Overview

This assignment trains and evaluates various classifiers to detect **deepfake cat images**. The fake images are generated using **Stable Diffusion 1.5**, resized to match the CIFAR-10 resolution. Real images come from CIFAR-10's cat class.

---

## 🧠 Techniques Used

- **PyTorch Neural Networks**
- **Cross-Validation** with `sklearn.model_selection.KFold`
- **Logistic Regression** baseline
- **MLP classifier**
- **Feature extraction** using pre-trained vision transformers (ViT)
- **Visualization** using `torchvision` utilities

---

## 🧪 Model Architecture & Approach

### 1. **Baseline Logistic Regression**
- Flattened image input
- Trained using `sklearn.linear_model.LogisticRegression`

### 2. **MLP Classifier**
- Custom PyTorch model using `nn.Sequential`
- ReLU activation, fully connected layers
- Trained with **Stochastic Gradient Descent (SGD)**

### 3. **Transfer Learning (Bonus/Extra Credit)**
- Used **HuggingFace’s ViT (Vision Transformer)** for feature extraction
- Fine-tuned downstream classifier on top of extracted embeddings

---

## 📊 Results Summary

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression     | ~68%     |
| MLP                     | ~75-80%  |
| ViT + Classifier (Bonus)| ~90%+    |

> Results vary slightly based on initialization and random folds.

---

## 📂 Dataset Info

- File: `hw2_data.pt`
- 2000 total images:  
  - `X`: shape `(2000, 3, 32, 32)`  
  - `y`: binary labels (0 = real, 1 = fake)

To use this dataset, download it from:  
🔗 [UCR eLearn Data Link](https://elearn.ucr.edu/courses/169673/files/17302822/download?download_frd=1)

---

## 📈 Visualization

Sample grid plots display:
- **Real Cat Images**
- **Fake Cat Images (Deepfakes)**

Useful for understanding model behavior and dataset bias.

---

## 🛠️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/deepfake-cat-detector.git
   cd deepfake-cat-detector
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn matplotlib transformers
   ```

3. Add dataset file `hw2_data.pt` to the working directory.

4. Run the notebook:
   ```bash
   jupyter notebook Deepfake_Cat_Detector.ipynb
   ```

---

## 🧠 Learnings & Takeaways

- Deepfake detection is feasible even on small, low-resolution images.
- Pretrained models (like ViT) drastically improve performance with minimal training.
- Proper validation and visualization are key for real-world AI detection tasks.

---

## 👩‍💻 Author

**Vismaya Anand Bolbandi**  
📧 vbolb001@ucr.edu  
🏫 University of California, Riverside

---