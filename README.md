
# 📊 Telco Customer Churn Prediction with Deep Learning

This project aims to predict **customer churn** for a telecommunications company using a deep learning approach. Built using Python and Keras, the model explores how regularization, model tuning, and proper preprocessing can significantly impact predictive performance.

---

## 🚀 Overview

Customer churn is a crucial business metric. By predicting which customers are likely to leave, companies can take proactive steps to retain them. In this notebook, we develop and evaluate a deep learning model trained on Telco customer data to detect churners before they leave.

The project covers:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature transformation and scaling
- Neural network modeling with:
  - L1, L2, and L1+L2 regularization
- Performance evaluation via metrics and ROC curves
- Hyperparameter optimization using **Keras Tuner**

---

## 🧠 Deep Learning Model

### Why 16 Neurons in the First Layer?

- **Input Shape**: 18 features after preprocessing
- **Chosen Hidden Units**: 16 neurons (close to number of input features)
- **Reason**: Based on empirical practice for simplicity and efficiency
- **Tuning Ready**: The architecture allows for easy scaling with tools like `KerasTuner`

### Regularization Techniques Used:
- **L1**: Encourages sparsity
- **L2**: Penalizes large weights to reduce overfitting
- **Combined**: A hybrid regularization technique balancing sparsity and generalization

---

## 🛠️ Technologies Used

- Python (Jupyter Notebook)
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for visualization
- Scikit-learn for preprocessing and metrics
- TensorFlow / Keras for building and training deep learning models
- Keras Tuner for hyperparameter search

---

## 📈 Metrics and Evaluation

- Accuracy, Precision, Recall
- ROC Curve Analysis
  - Used to detect overfitting and evaluate model discrimination
- Random Search with `Keras Tuner` for optimal architecture

---

## 📦 How to Run

```bash
git clone https://github.com/your-username/telco-churn-dl.git
cd telco-churn-dl
jupyter notebook telco_customer_churn_.ipynb
```

---

## 📊 Production-Ready Inference

The notebook ends with a **prediction section**, simulating deployment in production for unseen customer data.

---

## ✅ Key Takeaways

- Preprocessing is critical: categorical encoding and scaling improved model performance.
- Regularization helped reduce overfitting.
- Keras Tuner provided a simple interface for model optimization.
- A well-tuned neural network can achieve competitive performance in classification tasks like churn prediction.

---

## 📌 Next Steps

- Export the trained model using `.h5` format
- Create a REST API using Flask or FastAPI
- Integrate with a dashboard for live monitoring
