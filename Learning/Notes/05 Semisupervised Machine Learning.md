# Semi-supervised Machine Learning

**Semi-supervised learning** is a type of machine learning that lies between **supervised** and **unsupervised** learning. It uses a small amount of **labeled data** along with a large amount of **unlabeled data** to train a model. This is especially useful when labeling data is expensive or time-consuming, but unlabeled data is abundant.

---

### 🔧 **Technical Explanation**

* The model is first trained on the small labeled dataset.
* Then it uses the learned patterns to make predictions on the unlabeled data.
* These predictions help refine the model further, often through iterative techniques like pseudo-labeling or consistency training.

---

### 📊 **Common Algorithms Used**

* **Semi-supervised Support Vector Machines (S3VM)**
* **Label Propagation**
* **Self-training**
* **Generative models (e.g., Variational Autoencoders, GANs with labeled seeds)**

---

### 💡 **Real-Life Examples**

#### 1. **Medical Diagnosis**

* **Labeled data**: A few thousand X-ray images labeled as “diseased” or “healthy”.
* **Unlabeled data**: Millions of X-rays with no diagnosis.
* Semi-supervised learning uses the labeled images to train an initial model, then predicts and learns from the unlabeled ones, improving the accuracy with minimal expert annotation.

#### 2. **Speech Recognition**

* Annotated voice samples are costly to produce.
* With a small labeled dataset (e.g., audio with transcriptions) and a large corpus of raw audio, models can learn better pronunciation, accents, and patterns.

#### 3. **E-commerce Recommendations**

* **Labeled data**: User interactions where the system knows that a user clicked or bought a product.
* **Unlabeled data**: Browsing histories without actions.
* Semi-supervised learning helps make smarter recommendations based on partial feedback.

---

### ⚙️ **Technical Example: Pseudo-Labeling**

In image classification:

1. Train a neural network on 100 labeled cat/dog images.
2. Predict labels for 10,000 unlabeled images.
3. Add high-confidence predictions back into the training set.
4. Retrain the model to improve performance.

---
