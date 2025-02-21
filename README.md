### **README.md for Predicting Readability of Texts Using Machine Learning**

---

## **Predicting Readability of Texts Using Machine Learning**

This project focuses on predicting the readability of texts using **machine learning** and **deep learning** techniques. The goal is to determine how difficult or easy a given text is to read, which can be useful for educators, students, and content creators. The project uses various text preprocessing methods, feature engineering, and machine learning models to achieve accurate predictions.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [How It Works](#how-it-works)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)
12. [Contact](#contact)

---

## **Overview**
This project uses machine learning to predict the readability of texts. It involves preprocessing text data, extracting meaningful features, and training models to predict readability scores. The project explores various models, including neural networks and gradient boosting decision trees, to achieve the best results.

---

## **Key Features**
- **Text Preprocessing**: Cleaning, tokenizing, and lemmatizing text.
- **Feature Engineering**: Extracting features like word count, sentence length, and punctuation counts.
- **Word Embeddings**: Using **Glove Vectors (Word2Vec)** and **TF-IDF Word2Vec** to capture semantic meaning.
- **Machine Learning Models**: Neural networks, gradient boosting decision trees, linear regression, and more.
- **Evaluation Metrics**: Mean Squared Error (MSE) and Mean Absolute Error (MAE) for model evaluation.
- **Visualizations**: Pairplots, heatmaps, and regression plots for data analysis.

---

## **Technologies Used**
- **Python Libraries**:
  - `pandas`, `numpy` for data manipulation.
  - `scikit-learn` for machine learning models.
  - `tensorflow` and `keras` for neural networks.
  - `nltk` for natural language processing.
  - `seaborn` and `matplotlib` for data visualization.
- **Word Embeddings**:
  - **Glove Vectors** for semantic understanding.
  - **TF-IDF Vectorizer** for word importance.

---

## **Dataset**
The dataset used in this project is from **Kaggle** and contains text excerpts with readability scores. The goal is to predict the readability score (target) based on the text features.

Dataset Link: [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/data)

---

## **How It Works**
1. **Preprocessing**: Clean and preprocess the text data.
2. **Feature Extraction**: Create numerical features from the text (e.g., word counts, sentence lengths, Glove Vectors).
3. **Model Training**: Train machine learning and deep learning models on the preprocessed data.
4. **Evaluation**: Evaluate models using MSE and MAE, and visualize results.
5. **Prediction**: Predict readability scores for new texts.

---

## **Results**
- The **Gradient Boosting Decision Tree** and **Neural Network with TF-IDF Word2Vec** performed the best, achieving the lowest errors.
- The use of **Glove Vectors** and **TF-IDF Word2Vec** significantly improved model performance by preserving the semantic meaning of the text.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/predicting-readability.git
   ```
2. Navigate to the project directory:
   ```bash
   cd predicting-readability
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Predicting_Readability_of_Texts.ipynb
   ```
2. Follow the steps in the notebook to:
   - Preprocess the data.
   - Extract features.
   - Train and evaluate models.
   - Visualize results.

---

## **Contributing**
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- Dataset provided by [CommonLit](https://www.commonlit.org/) and hosted on [Kaggle](https://www.kaggle.com/).
- Glove Vectors from [Stanford NLP](https://nlp.stanford.edu/projects/glove/).

---

## **Contact**
For questions or feedback, feel free to reach out:
- **Email**: rohinikoli076@gmail.com
- **GitHub**: [Rohini-Koli9](https://github.com/Rohini-Koli9)

---

**Happy Coding!** 🚀
