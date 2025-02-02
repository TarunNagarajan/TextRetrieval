# Learning-to-Rank (LTR) Model for Search Ranking

## Overview
This project implements a **Learning-to-Rank (LTR)** model using advanced ranking algorithms such as **Gradient Boosting** and **XGBoost**, as well as deep learning approaches to improve search ranking in information retrieval tasks. The main objective is to preprocess sparse datasets, optimize hyperparameters, and evaluate ranking performance using metrics like **Mean Reciprocal Rank (MRR)** and **Normalized Discounted Cumulative Gain (NDCG)**.

## Table of Contents
1. [Project Goals](#project-goals)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Project Goals
- Develop a Learning-to-Rank model that learns from the features of the query-document pairs.
- Implement algorithms like **XGBoost** and **Gradient Boosting** for efficient learning from sparse data.
- Preprocess sparse data from text files, handle feature indexing and normalization.
- Optimize model hyperparameters to maximize ranking metrics (e.g., MRR, NDCG).
- Evaluate the ranking models on a custom dataset and compare results across models.

## Technologies Used
This project uses the following technologies:
- **Python 3.x**: The main programming language used for developing the models.
- **NumPy**: Used for handling arrays and numerical operations.
- **Pandas**: For handling and manipulating structured data.
- **XGBoost**: A high-performance gradient boosting framework for ranking tasks.
- **Scikit-learn**: For building and evaluating machine learning models.
- **TensorFlow/Keras**: For deep learning-based ranking models (if included in the project).
- **Matplotlib/Seaborn**: For visualizing results and model evaluation metrics.
- **Google Colab**: Used for coding, running experiments, and sharing the project.
- **GitHub**: Version control and project collaboration.

## Dataset
The model is trained on a sparse dataset containing query-document pairs. Each query-document pair includes:
- **Relevance Score**: The ranking or relevance score of the document for the given query.
- **Feature Vector**: A vector of features representing various aspects of the query-document pair.

An example of the data format:
```
1 1:0.2 2:0.5 3:0.3
0 2:0.8 4:0.6
1 1:0.4 3:0.2 5:0.7
```
- The first number is the relevance score (0 or 1).
- The subsequent numbers represent the feature index (starting from 1) and the corresponding feature value.

## Installation
To set up the project locally, follow the steps below.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Learning-to-Rank.git
   cd Learning-to-Rank
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   Ensure that the sparse dataset (`train.txt`, `test.txt`, etc.) is downloaded and placed in the correct folder (e.g., `/data/`).

## Project Structure
The project follows a modular structure to keep the code clean and organized.

```
Learning-to-Rank/
├── data/                  # Raw and processed datasets
│   ├── train.txt
│   ├── test.txt
│   └── validation.txt
├── notebooks/             # Jupyter Notebooks for data exploration and analysis
├── src/                   # Source code for model training and evaluation
│   ├── data_preprocessing.py  # Functions for preprocessing the dataset
│   ├── xgboost_model.py      # XGBoost model training script
│   ├── gradient_boosting.py  # Gradient Boosting model training script
│   └── evaluation.py         # Model evaluation scripts (MRR, NDCG, etc.)
├── requirements.txt       # List of project dependencies
├── README.md              # Project documentation
└── results/               # Saved models, logs, and evaluation metrics
```

## Model Training and Evaluation
### Data Preprocessing
The data preprocessing includes:
1. **Parsing Sparse Data**: The sparse feature data is parsed and transformed into a dense feature vector.
2. **Feature Normalization**: Feature values are normalized for better model convergence.

### Model Training
- **XGBoost Model**: A gradient boosting model is trained using the sparse dataset.
- **Gradient Boosting Model**: An alternative gradient boosting model is also trained for comparison.

```python
from xgboost import XGBRanker
model = XGBRanker(objective='rank:pairwise', eval_metric='ndcg')
model.fit(X_train, y_train)
```

### Model Evaluation
The model is evaluated using metrics like **Mean Reciprocal Rank (MRR)** and **Normalized Discounted Cumulative Gain (NDCG)** to assess the ranking quality.

```python
from sklearn.metrics import mean_squared_error
mrr = calculate_mrr(y_true, y_pred)
ndcg = calculate_ndcg(y_true, y_pred)
```

## Results
### Model Performance
- **XGBoost** achieved a Mean Squared Error of `0.1199` and an MRR of `1.0`.
- **Gradient Boosting** showed similar results, offering a comparison point for hyperparameter tuning.

You can visualize the performance of the models through the evaluation scripts in `src/evaluation.py`.

### Example Output:
```text
XGBoost Model MSE: 0.1199
Gradient Boosting Model MSE: 0.1205
XGBoost MRR: 1.0
Gradient Boosting MRR: 0.98
```

## Usage
1. **Preprocess Data**:
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train Models**:
   ```bash
   python src/xgboost_model.py
   python src/gradient_boosting.py
   ```

3. **Evaluate Models**:
   ```bash
   python src/evaluation.py
   ```

## Contributing
Contributions are welcome! If you'd like to improve or add new features to the project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make changes and commit them (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
