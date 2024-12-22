# Machine Learning Spam Detection

A machine learning project that classifies Yelp reviews of Chicago hotels and restaurants as **spam** or **not spam**, using the review text together with reviewer and product metadata.

## Overview

The notebook covers the full workflow: exploratory analysis, text cleaning, feature engineering, training and comparing three classifiers (Logistic Regression, Random Forest, XGBoost).

<!-- The main challenge is **class imbalance**: only about 13% of the reviews are spam, so accuracy alone is misleading. See [Results](#results) and [Limitations and next steps](#limitations-and-next-steps). -->

## Dataset

The data are Yelp reviews of hotels and restaurants in the Chicago area, split into a training and a test file. The target column `Label` is `Y` for spam and `N` for not spam.

| File                     |   Rows |
| ------------------------ | -----: |
| `data/training_data.csv` | 47,176 |
| `data/testing_data.csv`  | 20,219 |

Columns:

| Column                                               | Description                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------- |
| `id`                                                 | Row identifier                                                      |
| `Date`                                               | Date the review was posted                                          |
| `review ID`, `reviewer ID`, `product ID`             | Identifiers for the review, its author and the hotel/restaurant     |
| `rating_Helpful`, `rating_Thanks`, `rating_LoveThis` | Vote counts the review received                                     |
| `rating_OhNo`                                        | Integer from 1 to 5; despite its name it behaves like a star rating |
| `reviews`                                            | Review text                                                         |
| `Label`                                              | `Y` = spam, `N` = not spam                                          |

**Source.** The combined size of the two files (67,395 reviews) is from the **YelpChi** dataset, introduced by Mukherjee et al. (2013) and used by Rayana & Akoglu (2015). In that dataset the labels come from Yelp's own review filter. See [References](#references).

## Pipeline

### 1. Exploratory analysis

- Shapes, summary statistics and a null check.
- Class distribution plot (strongly imbalanced toward non-spam).
- Date parsing and a review-length feature.
- Correlation heatmap of the numeric features.

### 2. Text cleaning

Each review goes through `clean_text`, which applies, in order:
punctuation removal, correction of exaggerated letter repetitions (TextBlob), homoglyph normalization (`unidecode`), masking of URLs, HTML and e-mail addresses, masking of numbers, lowercasing and stopword removal (NLTK).

### 3. Feature engineering

| Feature                                       | Description                                                     |
| --------------------------------------------- | --------------------------------------------------------------- |
| `langth`                                      | Review length in characters                                     |
| `review_frequency`, `total_reviews`           | Number of reviews written by the same reviewer                  |
| `helpful_ratio`, `thanks_ratio`, `love_ratio` | Reviewer's average Helpful / Thanks / LoveThis votes per review |
| `reviews_count`                               | Number of reviews for the same product                          |
| `overall_rating`                              | Per-review score: Helpful + Thanks + LoveThis - `rating_OhNo`   |
| `avg_rating`                                  | Mean of `overall_rating` over the product's reviews             |
| `polarity`, `subjectivity`                    | Sentiment scores from TextBlob                                  |

### 4. Modeling

Every model is a scikit-learn-style pipeline:

**TF-IDF on the cleaned text + numeric features (`langth`, `review_frequency`, `reviews_count`, `avg_rating`) → `MaxAbsScaler` → classifier**

Data are split 80% / 20% into training and validation sets (`random_state=42`). Three classifiers are compared with default hyperparameters: Logistic Regression, Random Forest and XGBoost.

### 5. Final predictions

XGBoost is refit on the full training set and used to predict labels for the test set. <!--  The test file has no labels, so no test score is computed. -->

## Results

Scores on the 20% validation split:

| Model               | Accuracy | F1 (spam class) |
| ------------------- | -------: | --------------: |
| Logistic Regression |   0.8572 |          0.1607 |
| Random Forest       |   0.8662 |          0.0141 |
| XGBoost             |   0.8649 |          0.1103 |

## Limitations and next steps

- **Imbalance is not handled.** No class weights, resampling or threshold tuning were used, which explains the low F1 scores. Next: `class_weight` / `scale_pos_weight`, resampling, threshold tuning, and evaluation with precision-recall metrics.
- **Not all engineered features are used.** `overall_rating`, `polarity` and `subjectivity` are computed but the models only receive the four numeric features listed above. Next: include them and compare.
<!-- - **Test features are computed within the test set.** Reviewer and product statistics for the test set come from the test data only. -->
- **No hyperparameter tuning or cross-validation.** All models use default settings and a single split.

<!-- ## Repository structure

```
.
├── data/
│   ├── training_data.csv
│   └── testing_data.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
``` -->

## Getting started

Tested with Python 3.12.

```bash
git clone https://github.com/chakibouzane/Machine-Learning-Spam-Detection.git
cd Machine-Learning-Spam-Detection

python -m venv .venv
source .venv/bin/activate

jupyter notebook notebook.ipynb
```

## References

- A. Mukherjee, V. Venkataraman, B. Liu, N. Glance. _What Yelp Fake Review Filter Might Be Doing?_ ICWSM, 2013.
- S. Rayana, L. Akoglu. _Collective Opinion Spam Detection: Bridging Review Networks and Metadata._ ACM SIGKDD, 2015.
