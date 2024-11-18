# LLM_Dataloader

A robust and easy-to-use dataloader for recommendation system datasets. The `LLM4Rec Dataloader` automates the process of downloading, loading, and preprocessing datasets for machine learning tasks, particularly for matrix factorization and other collaborative filtering techniques.

---

## **Table of Contents**

- [Features](#features)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Loading](#dataset-loading)
  - [Data Splitting](#data-splitting)
  - [Matrix Factorization Example](#matrix-factorization-example)
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

---

## **Supported Datasets**

### **MovieLens**
- Variants: `"100k"`, `"1M"`, `"10M"`, `"20M"`
- Example usage:
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

### **MIND**
- Variants: `"demo"`, `"small"`, `"large"`
- Example usage:
  ```python
  from mind import load_pandas_df

  behaviors_train_df, behaviors_dev_df, news_train_df, news_dev_df = load_pandas_df(
      size="small",
      behaviors_header=None,
      news_header=None,
      npratio=4
  )
  ```

---

## **Installation**

Clone the repository:
```bash
git clone https://github.com/tnnanh1010/LLM4Rec-Dataloader.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Dataset Loading**

#### MovieLens Example:
```python
import movielens

movie_df = movielens.load_pandas_df(
    size="1M",
    header=("userID", "itemID", "rating", "timestamp")
)
```

#### MIND Example:
```python
from mind import load_pandas_df

behaviors_train_df, behaviors_dev_df, news_train_df, news_dev_df = load_pandas_df("small")
```

---

### **Data Splitting**

#### Random Split:
```python
from splitter import random_split

splits_df = random_split(movie_df, [0.8, 0.2])
```

#### Chronological Split:
```python
from splitter import chrono_split

splits_df = chrono_split(movie_df, ratio=[0.8, 0.2], min_rating=1, filter_by="user", col_user='userID', col_item='itemID', col_timestamp='timestamp')
```

#### Interactive Split:
```python
from splitter import interactive_split

splits_df = interactive_split(movie_df, [0.8, 0.2])
```

#### Sequential Split:
```python
from splitter import sequential_split

splits_df = sequential_split(movie_df, [0.8, 0.2])
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