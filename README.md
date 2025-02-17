# Practical Application (Module-17)
Berkley HAAS (ML AI - August 2024 Batch)

---

# Strategy

- It was evident that we may have to try to use combinition of 
  - Overloading the Minority class as "yes" was just 10% of all dataset
    - SMOTETomek (LogisticTrgression, KNN)
    - BalancedBaggingClassifier (DecisionTreeClassification)
  - FeatureSelection to pick the fields with most variance.
  - Model Classification
    - LogisticRegression
    - DecisionTreeClassification 
    - K-Nearest Neighbor
- Additionally different scoring strategy were tried out like 1.fbeta_score, 2. recall_score and 3. precession_score. Accuracy Score was not considered as the dataset was not balanced. But we did plot it for various threshold.
- After each fit, different combinition of threshold and score were plotted to see which is best threshold for the prediction.
- The 4 scores which were considered are 1. Accuracy, 2. Precision Score, 3. Recall Score, F1 Score

---

**Note**

FeatureSelection was just done in one of the the models, as it follows column_transformation and this 2 steps are always first and   independent, of following steps, for all the future models we did not set HyperParameter, but selected the once whch was initially determined. This saved a ton of computing steps, time and resources. Following 2 combinition were tried out

```python
    "feature_selection__estimator__penalty": ['l1', 'l2'],
    'feature_selection__estimator__solver': ['liblinear'],
    "feature_selection__estimator__C": [0.001, 0.01, 0.1, 10, 100],    
```

```python
    "feature_selection__estimator__penalty": ['None', 'l2'],
    'feature_selection__estimator__solver': ['lbfgs','newton-cg', 'newton-cholesky'],
```

The finalized parameters were

```json
{
 'feature_selection__estimator__C': 0.01,
 'feature_selection__estimator__penalty': 'l1',
 'feature_selection__estimator__solver': 'liblinear'
}
```

---

Below Chart shows details each run for each classifier.

# LogisticRegression

2_LogisticRegression-2.ipynb

| Metric                    | recall_score | f1_score   | fbeta_score | recall_score | recall_score | f1_score   |
| :------------------------ | :----------- | :--------- | ----------- | ------------ | ------------ | ---------- |
| Data Sampling Algorithms  | SMOTETomek   | SMOTETomek | SMOTETomek  | SMOTETomek   | SMOTETomek   | SMOTETomek |
| Cost-Sensitive Algorithms | Yes          | Yes        | Yes         | No           | No           | Yes        |
| Feature Selection         | Yes          | Yes        | Yes         | Yes          | No           | No         |
| Accuracy                  | 0.8188       | 0.8534     | 0.8534      | 0.8534       | 0.8562       | 0.8562     |
| Precision Score           | 0.3753       | 0.4264     | 0.4264      | 0.4264       | 0.4319       | 0.4319     |
| Recall Score              | 0.9200       | 0.8804     | 0.8804      | 0.8804       | 0.8840       | 0.8840     |
| F1 Score                  | 0.5331       | 0.5745     | 0.5745      | 0.5745       | 0.5803       | 0.5803     |

- 

# DecisionTreeClassification

3_DecisionTree_1.ipynb

|                           | Model 1     | Model 2     | Model 3      | Model 4     | Model 5     |
| :------------------------ | :---------- | :---------- | ------------ | ----------- | ----------- |
| Metric                    | fbeta_score | fbeta_score | Recall_score | fbeta_score | fbeta_score |
| Data Sampling Algorithms  | No          | No          | No           | No          | No          |
| Cost-Sensitive Algorithms | No          | No          | No           | Yes         | Yes         |
| Feature Selection         | No          | Yes         | Yes          | No          | Yes         |
| Threshold                 | 0.50        | 0.50        | 0.45         | 0.75        | 0.77        |
| Accuracy                  | 0.9128      | 0.9114      | 0.9114       | 0.8429      | 0.8385      |
| Precision Score           | 0.6287      | 0.6080      | 0.6080       | 0.4126      | 0.4065      |
| Recall Score              | 0.5485      | 0.5975      | 0.5975       | 0.9388      | 0.9488      |
| F1 Score                  | 0.5859      | 0.6027      | 0.6027       | 0.5733      | 0.5692      |

Both **Model 4** and **Model 5** have fantastic recall score, but low **Precision Score**. Rest 3 Models have performed around average with respect to Precision and Recall. **Model 3** performed worst, after the threshold of 70, all the F1 Score, Recall and Precision score dropped to 0



# BalancedBaggingClassifier + DecisionTreeClassifier

3_DecisionTree_2.ipynb

| Metric                    | fbeta_score | fbeta_score | recall_score | fbeta_score | fbeta_score | precision_score |
| :------------------------ | :---------- | :---------- | ------------ | ----------- | ----------- | --------------- |
| Data Sampling Algorithms  | Yes         | Yes         | Yes          | Yes         | Yes         | Yes             |
| Cost-Sensitive Algorithms | No          | No          | No           | Yes         | Yes         | Yes             |
| Feature Selection         | No          | Yes         | Yes          | No          | Yes         | Yes             |
| Threshold                 | 0.50        | 0.50        | 0.45         | 0.75        | 0.77        |
| Accuracy                  | 0.8642      | 0.8674      | 0.7873       | 0.8714      | 0.8674      | 0.9076          |
| Precision Score           | 0.4498      | 0.4549      | 0.3404       | 0.4635      | 0.4549      | 0.6089          |
| Recall Score              | 0.9294      | 0.9049      | 0.9517       | 0.9121      | 0.9049      | 0.4989          |
| F1 Score                  | 0.6062      | 0.6054      | 0.50151      | 0.6147      | 0.6054      | 0.5484          |



# K-Nearest Neighbor

|                           | Model 1     | Model 2     | Model 3     | Model 4     | Model 5      | Model 6    |
| :------------------------ | :---------- | :---------- | ----------- | ----------- | ------------ | ---------- |
| Metric                    | fbeta_score | fbeta_score | fbeta_score | fbeta_score | recall_score | f1_score   |
| Data Sampling Algorithms  | No          | No          | SMOTETomek  | SMOTETomek  | SMOTETomek   | SMOTETomek |
| Cost-Sensitive Algorithms | No          | No          | No          | Yes         | Yes          | Yes        |
| Feature Selection         | No          | Yes         | Yes         | Yes         | Yes          | Yes        |
| Threshold                 | 0.45        | 0.45        | 0.80        | 0.80        | 0.75         | 0.70       |
| Accuracy                  | 0.8991      | 0.9024      | 0.8864      | 0.8897      | 0.8830       | 0.8750     |
| Precision Score           | 0.5552      | 0.5691      | 0.4965      | 0.5071      | 0.4873       | 0.4677     |
| Recall Score              | 0.5176      | 0.5421      | 0.7336      | 0.6889      | 0.7760       | 0.8092     |
| F1 Score                  | 0.5357      | 0.5553      | 0.5922      | 0.5842      | 0.5987       | 0.5928     |

- The Best result for KNN Clasifier was **Model 6**, which helps us high recall_score, with marginal improvement in **F1 Score** with slight hit on **Precission Score**
