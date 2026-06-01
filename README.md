# Data Classification Using AI
## Project 2 — DecodeLabs AI Internship

Welcome to the repository for **Project 2: Data Classification Using AI**. This project marks the transition from static, heuristic-based rule design ("simple if/then rules") into the paradigm of **Supervised Learning**. Using an algorithmic logic pipeline, this project demonstrates how to train, test, and validate a machine learning model to recognize underlying data patterns and accurately categorize new, unseen information.

---

## 📌 Project Overview
The primary objective of this project is to implement a robust machine learning pipeline using the foundational **IPO (Input-Process-Output) Framework**:
* **Input:** Load and explore the classic **Iris Benchmark Dataset** (150 balanced samples across 3 unique botanical classes, characterized by 4 distinct features: Sepal Length, Sepal Width, Petal Length, and Petal Width).
* **Process:** Handle the data integrity lifecycle by performing a randomized **Train-Test Split** to eliminate ordering bias. Apply the **K-Nearest Neighbors (KNN)** algorithm to map decision boundaries.
* **Output:** Evaluate model precision using rigorous statistical metrics, including an **Accuracy Score**, **Confusion Matrix**, and a detailed **Classification Report** (evaluating Precision, Recall, and F1-Score).

---

## ⚙️ Core Technical Pipeline

1. **Data Ingestion & Understanding (`Load`):** Fetch the structured multi-dimensional dataset and structure it into an easily manipulable dataframe context.
2. **Structural Integrity (`Split`):** Randomly shuffle and partition data into a **Training Set** (80%) for pattern recognition and a **Test Set** (20%) for absolute validation.
3. **Model Materialization (`Apply`):** Fit a supervised learning `KNeighborsClassifier` to dynamically derive the underlying decision boundary logic rather than writing manual heuristics.
4. **Performance Evaluation (`Validate`):** Test model predictions against true labels using validation scoring frameworks to ensure absolute alignment.

---

## 📂 File Architecture
```bash
├── AI_Project2_Data_Classification.py  # Main Python execution pipeline script
└── README.md                           # Comprehensive documentation file
