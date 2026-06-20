# Naive Bayes

## Overview
Naive Bayes Classification – NBA Player Longevity Prediction
Project Overview
In this project, you will analyze engineered NBA player data using Python and Scikit-learn to build a Gaussian Naive Bayes classification model. You will learn how to identify target variables, implement probabilistic classifiers, evaluate performance using confusion matrices and business-relevant metrics, and interpret model assumptions in the context of sports analytics. This project helps you translate statistical predictions into actionable scouting insights.

## Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

## Loading the Dataset

Load the dataset and clearly define the `target_5yrs` column as the dependent variable
 Load the Housing dataset
data = pd.read_csv('extracted_nba_players_data.csv')

Add a visualization (e.g., sns.countplot) to check the distribution of 'target_5yrs'.
plt.figure(figsize=(6, 4))
sns.countplot(x='target_5yrs', data=data, hue='target_5yrs', palette='viridis', legend=False)
plt.title('Distribution of Target_5yrs (Career Longevity)')
plt.xlabel('Played 5+ years in NBA (0=No, 1=Yes)')
plt.ylabel('Number of Players')
plt.show()
<img width="540" height="393" alt="feature_engr_dist" src="https://github.com/user-attachments/assets/3ecb74f7-8c54-49da-8cb4-2ce5edf386a6" />

## Check for class balance/distribution of the target variable.
## Checking Class Balance of Target Variable
We will visualize the distribution of the `target_5yrs` column to understand the balance between the two classes (players who played 5+ years in NBA vs. those who did not). This is important for evaluating model performance, especially if the classes are imbalanced.
plt.figure(figsize=(7, 5))
sns.countplot(x='target_5yrs', data=data, hue='target_5yrs', palette='coolwarm', legend=False)
plt.title('Distribution of Target_5yrs (Career Longevity)')
plt.xlabel('Played 5+ years in NBA (0=No, 1=Yes)')
plt.ylabel('Number of Players')
plt.show()

# Also print the value counts for a numerical summary
print("\nValue counts for 'target_5yrs':")
print(data['target_5yrs'].value_counts())

<img width="618" height="470" alt="feature_engr_dist_career" src="https://github.com/user-attachments/assets/5bd272fb-2bee-4ff3-baaa-44c5c802358a" />

Preprocess features to ensure compatibility with Gaussian Naive Bayes (continuous metrics)
X = data.drop(columns=["target_5yrs"])
y = data["target_5yrs"]

X.head(5)

# Splitting into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Feature Scaling
Implementing feature scaling using `sklearn.preprocessing.StandardScaler`. Scaling is performed after splitting the data to ensure that the scaling parameters are learned only from the training data, preventing data leakage from the test set.
from sklearn.preprocessing import StandardScaler

#### Initialize the StandardScaler
scaler = StandardScaler()

#### Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#### Convert the scaled arrays back to DataFrames, preserving column names
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("X_train after scaling:")
display(X_train.head())
print("\nX_test after scaling:")
display(X_test.head())

### Features for Gaussian Naive 

---

Bayes

---



Gaussian Naive Bayes assumes that the features are continuous and follow a Gaussian (normal) distribution. Our features have already been scaled using `StandardScaler`, which transforms them to have a mean of 0 and a standard deviation of 1. This scaling step helps to meet the distributional assumptions of Gaussian Naive Bayes, making the data suitable for this classifier.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


### Training the Gaussian Naive Bayes Model
#### Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

#### Train the model on the scaled training data
gnb.fit(X_train, y_train)

print("Gaussian Naive Bayes model trained successfully.")

### Making Predictions and 

---

Evaluating the Model

---



```
# This is formatted as code
```

### Making Predictions and 

---

Evaluating the Model

---



```
# This is formatted as code
```

#### Make predictions on the scaled test data
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)[:, 1] # Probabilities for the positive class

#### Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

#### Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Longevity', 'Predicted Longevity'],
            yticklabels=['Actual No Longevity', 'Actual Longevity'])
plt.title('Confusion Matrix for Gaussian Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

<img width="683" height="547" alt="naive_bayes_confusion_matrix" src="https://github.com/user-attachments/assets/89b209ef-4357-450b-b58c-cec80e4653d1" />

## Identifying Influential Features in Gaussian Naive Bayes

Unlike models such as Linear Regression (which uses coefficients) or tree-based models (which use impurity-based importance), Gaussian Naive Bayes doesn't directly output a 'feature importance' score. Instead, its predictions are driven by the estimated mean and variance of each feature within each class.

For a Gaussian Naive Bayes model, features that are most influential in discriminating between classes (e.g., long vs. short career) are typically those that exhibit:

1.  **Larger differences in means between the classes**: If the average value of a feature is significantly different for players with long careers compared to those with short careers, that feature is highly indicative.
2.  **Smaller variances within each class**: If a feature's values are tightly clustered around its mean within each class, it provides a more consistent signal for that class.

### Interpreting Feature Influence from Model Parameters

To identify potentially influential features, one would typically inspect the `theta_` (means) and `var_` (variances) attributes of the `GaussianNB` model:

*   `gnb.theta_`: Contains the mean of each feature per class.
*   `gnb.var_`: Contains the variance of each feature per class.

By comparing these values, a scouting department could gain insights:

*   **Example**: If `total_points` has a much higher mean for `target_5yrs=1` than for `target_5yrs=0`, it suggests that players who score more points tend to have longer careers. If the variance for `total_points` within each group is small, it implies this trend is consistent.

**Actionable Insight for Scouting**: While not a direct ranking, analyzing these parameters can help scouts understand which statistical profiles are most characteristic of players likely to achieve longevity according to the model. Features with a clear separation in means between the '0' and '1' classes, especially when combined with low variance within those classes, would be considered highly influential.

### Naive Bayes "Independence Assumption" and Its Realism for Basketball Stats

The **Naive Bayes classifier** operates under a fundamental principle called the **"Independence Assumption"** (also known as conditional independence). It assumes that all features are independent of each other, given the class label. In simpler terms, it believes that the presence or absence of one feature does not affect the presence or absence of any other feature, as long as we know the outcome (e.g., whether a player will have a long career or not).

Let's break it down:

*   **Formal Statement**: P(Feature1, Feature2 | Class) = P(Feature1 | Class) * P(Feature2 | Class)

### Realism for Basketball Stats

For basketball statistics, the independence assumption is often **unrealistic** and frequently violated. As you rightly pointed out with the example of "points" and "minutes played", many basketball statistics are highly correlated. Here's why:

1.  **Correlation Between Points and Minutes Played**: A player who plays more minutes (Feature 1) is inherently going to have more opportunities to score points (Feature 2). Therefore, `points` and `minutes played` are **not independent**. If you know a player played a high number of minutes, you can reasonably expect their points total to be higher, regardless of their career longevity.

2.  **Other Examples of Interdependence in Basketball Stats**:
    *   **Field Goals Made (FGM) and Field Goals Attempted (FGA)**: A player cannot make a field goal without attempting one. These are directly linked.
    *   **Rebounds (REB) and Minutes Played (MIN)**: More minutes generally lead to more opportunities for rebounds.
    *   **Assists (AST) and Turnover (TOV)**: These often come from similar ball-handling situations; players with high usage rates might have both high assists and high turnovers.
    *   **Offensive Rebounds (OREB) and Defensive Rebounds (DREB)**: While distinct, a player's overall rebounding prowess often means they excel at both.

### Impact on Naive Bayes Model

Despite this unrealistic assumption, Naive Bayes models can still perform surprisingly well in practice. However, violating the independence assumption can lead to a few issues:

*   **Suboptimal Probability Estimates**: While the classification might still be accurate (predicting the correct class), the probability scores that the model outputs might be overconfident or skewed because it's multiplying independent probabilities when they should be conditionally dependent.
*   **Reduced Performance**: In cases of strong feature dependencies, other models that explicitly account for feature relationships (like Logistic Regression, SVMs, or tree-based models) might outperform Naive Bayes.

In our current dataset, we've already seen evidence of this with features like `fgm`, `fga`, `pts`, `fta`, `3pa`, and `reb` being highly correlated. While we dropped some of these due to multicollinearity, the underlying statistical reality in basketball data remains that many metrics are interconnected. Therefore, when using Gaussian Naive Bayes, we should be mindful that its strong independence assumption is likely not met, which could affect the interpretation of its probability outputs and potentially its predictive power compared to models that handle feature dependencies better.

# Conclusion
## Model Reliability and Limitations: Insights for a Scouting Department

For a scouting department, understanding a predictive model's strengths and weaknesses is crucial for making informed decisions. Here's a summary of our Gaussian Naive Bayes model's performance and key considerations:

### Model Reliability (What the model does well):

*   **Good at Identifying Long-Term Potential (Precision for Class 1)**: The model shows a relatively high precision (0.8258) for predicting that a player *will* play 5+ years in the NBA. This means that when the model says a player will have a long career, it's correct about 83% of the time. This can be valuable for identifying prospects with strong indicators for longevity.
*   **Decent Overall Accuracy**: With an accuracy of 0.6903, the model correctly classifies about 69% of players regarding their career longevity. This suggests it has some predictive power.
*   **Effective at Identifying Players *Without* Long Careers (Recall for Class 0)**: The model has good recall (0.77) for identifying players who *will not* play 5+ years. This means it's good at catching those who are likely to have shorter careers.

### Model Limitations (Where the model struggles and what to consider):

*   **Misses Some Long-Term Players (Lower Recall for Class 1)**: The model's recall for players who *do* achieve 5+ years is 0.6450. This means that while it's accurate when it predicts longevity, it *misses* about 35.5% of players who actually go on to have long careers. A scout might overlook some good prospects if relying solely on this model.
*   **Independence Assumption May Be Unrealistic**: Gaussian Naive Bayes assumes that all player statistics are independent of each other (given career longevity). In reality, many basketball stats (like points, minutes played, field goals made, etc.) are highly correlated. This unrealistic assumption might lead the model to be overconfident in its probability predictions, even if its final classification is often correct. This means the 'why' behind a prediction might not be as robust as with other models.
*   **Potential for Imbalance Impact**: The target variable (`target_5yrs`) is imbalanced, with more players having long careers (831) than short ones (509). While the model still performs reasonably, highly imbalanced datasets can sometimes bias models towards the majority class. This could contribute to the lower recall for the minority class (players with short careers).
*   **Feature Complexity**: The current features are raw and engineered stats. While useful, the model might benefit from more sophisticated features or interactions that capture complex dynamics of player performance over time.

### Actionable Takeaways for Scouting:

*   **Use as a Screening Tool**: The model can serve as an effective first-pass screening tool to highlight players with strong longevity signals (high precision for Class 1) and to flag those with clear indicators of shorter careers (high recall for Class 0).
*   **Combine with Expert Judgment**: Due to the limitations, especially the potential to miss genuinely good prospects (lower recall for Class 1) and the simplifying assumptions, the model's output should always be combined with human scouting expertise, contextual information, and other qualitative assessments.
*   **Focus on High-Confidence Predictions**: For critical decisions, scouts might focus on predictions where the model expresses very high confidence (i.e., very high probability scores for either class).
*   **Consider Other Models**: For future iterations, exploring models that handle feature dependencies more explicitly (like Logistic Regression, SVMs, or tree-based models) could offer different perspectives and potentially improve performance metrics like recall for the positive class.

*   
