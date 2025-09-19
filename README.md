
## 🚢 Titanic Survival Prediction

This project predicts the survival of passengers aboard the Titanic using **Machine Learning models**.
We preprocess the dataset, apply feature engineering, and evaluate multiple classifiers to find the best performing model.

---

## 📂 Dataset

The dataset used is the **Titanic dataset** (from Kaggle or other sources).
It contains passenger details such as:

* PassengerId
* Survived (Target variable: 0 = Did not survive, 1 = Survived)
* Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
* Engineered features (HasCabin, Titles, One-hot encoded Embarked etc.)

---

## ⚙️ Preprocessing Steps

1. Handle missing values (Age, Embarked, Cabin).
2. Feature engineering:

   * Extracted titles (Mr, Miss, Mrs, Rare, Royalty).
   * Created `HasCabin` feature.
   * One-hot encoding for categorical variables (`Sex`, `Embarked`).
3. Normalization (Age, Fare).
4. Train-test split for evaluation.

---

## 🧠 Models Used

We implemented and compared the following ML algorithms:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **K-Nearest Neighbors (KNN)**

---

## 📊 Evaluation Metrics

For each model, we evaluated performance using:

* **Accuracy**
* **Classification Report** (Precision, Recall, F1-score)
* **Confusion Matrix**

---

## 🔍 Key Results

* Logistic Regression and Decision Tree gave similar results (\~74–75% accuracy).
* Random Forest showed better generalization when tuned with parameters.
* KNN required normalization and tuning of `k` to perform well.

---

## 📈 Visualizations

We used different plots for understanding data and model results:

* **Boxenplots** → to check distribution and outliers (Age, Fare, Pclass).
* **Correlation Heatmap** → to see feature relationships.
* **Confusion Matrix Heatmaps** → to visualize model predictions vs actual values.
* **Feature Importance (Tree-based models)** → to check most important predictors.

---

## 🚀 Future Work

* Try **SVM, Gradient Boosting, XGBoost** for better performance.
* Apply **cross-validation** to tune hyperparameters more robustly.
* Deploy the model using Flask/Django or Streamlit.

---

## ▶️ How to Run

1. Clone the repo

```bash
git clone <repo-link>
cd titanic-ml
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook or Python script

```bash
jupyter notebook Titanic.ipynb
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy (Data handling)
* Matplotlib, Seaborn (Visualization)
* Scikit-learn (ML models & evaluation)

---
