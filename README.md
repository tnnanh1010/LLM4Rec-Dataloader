```markdown
# **LLM4Rec Dataloader**

A robust and easy-to-use dataloader for recommendation system datasets. The `LLM4Rec Dataloader` automates the process of downloading, loading, and preprocessing datasets for machine learning tasks, particularly for matrix factorization and other collaborative filtering techniques (working on).

---

## **Table of Contents**

- [Features](#features)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Loading](#dataset-loading)
  - [Data Splitting](#data-splitting)
  - [Matrix Factorization Example](#matrix-factorization-example)
  - [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## **Features**

- **Dataset Automation**:
  - Automatically downloads and loads datasets.
  - Supports preprocessing for machine learning tasks.
  
- **Supported Splitting Techniques**:
  - **Random Split**: Split data randomly into specified ratios.
  - **Chrono Split**: Split data chronologically based on timestamps.
  - **Interactive Split**: Ensures all splits contain data for all users using random splitting.
  - **Sequential Split**: Maintains chronological order within each user's split.

- **Pre-integrated with Matrix Factorization**:
  - Ready-to-use tools for preparing data for Matrix Factorization models.

- **Evaluation Metrics**:
  - Includes implementations for standard metrics like NDCG, MRR, and accuracy.

---

## **Supported Datasets**

### **MovieLens**
- Variants: `"100k"`, `"1M"`, `"10M"`, `"20M"`

### **MIND**
- Variants: `"demo"`, `"small"`, `"large"`

---

## **Installation**

Clone the repository:
```bash
git clone https://github.com/tnnanh1010/LLM4Rec-Dataloader.git
```

---

## **Usage**

### **Dataset Loading**

#### MovieLens Example:
```python
import movielens

movie_df = movielens.load_pandas_df(
    size="1M",
    header=("userID", "itemID", "rating", "timestamp"),
    title_col=None,
    genres_col=None,
    year_col=None
)
```

#### MIND Example:
```python
import mind

behaviors_train_df, behaviors_dev_df, news_train_df, news_dev_df = mind.load_pandas_df(
    size="small",
    behaviors_header=None,
    news_header=None,
    npratio=4
)
```

---

### **Data Splitting**

#### Random Split:
```python
from splitter import random_split

splits_df = random_split(
    movie_df, 
    ratio=[0.8, 0.2])
```

#### Chronological Split:
```python
from splitter import chrono_split

splits_df = chrono_split(
    movie_df,
    ratio=[0.8, 0.2],
    min_rating=1,
    filter_by="user",
    col_user='userID',
    col_item='itemID',
    col_timestamp='timestamp'
)
```

#### Interactive Split:
```python
from splitter import interactive_split

splits_df = interactive_split(
    movie_df, 
    ratio=[0.8, 0.2])
```

#### Sequential Split:
```python
from splitter import sequential_split

splits_df = sequential_split(
    movie_df, 
    ratio=[0.8, 0.2])
```

---

### **Matrix Factorization Example**

#### Import Libraries and Preprocess Data:
```python
from model.MF.MF import MatrixFactorization
from model.MF.preprocessing import ids_encoder, format_data
import pandas as pd
import movielens
import splitter

# Load dataset
df = movielens.load_pandas_df("1M")
ratings = pd.DataFrame(df[["userID", "itemID", "rating"]])

# Encode IDs
ratings, uencoder, iencoder = ids_encoder(ratings)

# Prepare training and testing sets
split_df = splitter.interactive_split(ratings)
x_train, y_train, x_test, y_test = format_data(split_df[0], split_df[1], 'movielens')

# Define Matrix Factorization model
m = ratings["userID"].nunique()
n = ratings["itemID"].nunique()
MF = MatrixFactorization(m, n, k=10, alpha=0.01, lamb=1.5)
```

#### Train and Evaluate Model:
```python
history = MF.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
MF.evaluate(x_test, y_test)
```

---

### **Evaluation Metrics**

The following evaluation metrics are implemented in `metric.py`:

#### 1. **Normalized Discounted Cumulative Gain (NDCG)**:
```python
from metric import ndcg_score

ndcg = ndcg_score(y_true, y_score, k=10)
print(f"NDCG: {ndcg}")
```

#### 2. **Mean Reciprocal Rank (MRR)**:
```python
from metric import mrr_score

mrr = mrr_score(y_true, y_score)
print(f"MRR: {mrr}")
```

#### 3. **Click-through Rate (CTR)**:
```python
from metric import ctr_score

ctr = ctr_score(y_true, y_score, k=1)
print(f"CTR: {ctr}")
```

#### 4. **Accuracy**:
```python
from metric import acc
import torch

accuracy = acc(torch.tensor(y_true), torch.tensor(y_hat))
print(f"Accuracy: {accuracy}")
```

#### 5. **Discounted Cumulative Gain (DCG)**:
```python
from metric import dcg_score

dcg = dcg_score(y_true, y_score, k=10)
print(f"DCG: {dcg}")
```

---

## **Contributing**

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Add new feature"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request.

---

## **License**

This project is licensed under the [MIT License](LICENSE).
```