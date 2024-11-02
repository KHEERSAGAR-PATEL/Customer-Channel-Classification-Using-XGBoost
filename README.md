
# Customer Channel Classification Using XGBoost

## Project Overview

This project implements the **XGBoost** classifier using Python and **Scikit-Learn** to classify customers into two major channels: **Horeca (Hotel/Restaurant/Café)** and **Retail**. The objective is to leverage **boosting techniques** to accurately distinguish between these channels based on various customer features, which can have practical applications in targeted marketing, customer segmentation, and sales analysis.

The project achieves a notable **accuracy of 93.42%**, underscoring XGBoost's capability in handling tabular data with complex patterns.

## Project Objectives

- **Develop a robust classification model** that differentiates between Horeca and Retail customers.
- **Optimize model performance** through hyperparameter tuning using k-fold cross-validation.
- **Identify key predictive features** to understand factors influencing customer classification.

## Dataset and Preprocessing

The dataset includes features related to customer behavior and purchasing patterns, with labels initially encoded as `1` and `2` to represent different customer channels. These labels were **converted to binary values (0 and 1)** to ensure compatibility with the classifier and facilitate a straightforward interpretation of results.

Key steps involved:
1. **Data Cleaning**: Checking for null values and outliers.
2. **Label Encoding**: Encoding categorical variables and adjusting labels for binary classification.
3. **Data Splitting**: Splitting data into training and test sets to evaluate model performance.

## Model Training

The **XGBoost classifier** was trained on the processed dataset with the following configurations:
- **Objective**: Binary Classification (`binary:logistic`)
- **Hyperparameters**: 
    - `colsample_bytree = 0.3`
    - `learning_rate = 0.1`
    - `max_depth = 5`
    - `alpha = 10`
  
After model training, the classifier reached an impressive **accuracy score of 93.42%** on the test data, indicating strong generalization capabilities.

### Hyperparameter Tuning

To further enhance the model, we employed **k-fold cross-validation** with XGBoost’s `cv()` function, optimizing key hyperparameters and performing early stopping. This process validated the model's robustness and improved its predictive reliability.

## Feature Importance Analysis

To gain insights into the factors influencing classification, we analyzed **feature importance** using XGBoost's `plot_importance()` function. This method ranks features based on their impact within the boosting framework, with **"Grocery"** emerging as the most influential feature for distinguishing between customer channels. This insight provides valuable information for business decision-making, allowing us to focus on high-impact features for segmentation strategies.

## Results and Conclusion

The XGBoost classifier achieved a final accuracy score of **93.42%**, demonstrating its efficacy in customer classification tasks. The project confirms that:
1. **XGBoost’s boosting technique** is highly effective for customer classification with structured data.
2. **Hyperparameter tuning** and **cross-validation** can significantly enhance model performance.
3. The **feature importance** analysis highlights "Grocery" as a pivotal factor, suggesting it plays a major role in distinguishing between Horeca and Retail channels.

## Usage

To run this project:
1. Clone the repository and navigate to the project folder.
2. Install the necessary dependencies from `requirements.txt`.
3. Run the Jupyter Notebook `Customer_Channel_Classification.ipynb` to train the model and view results.

```bash
git clone <repository-link>
cd <repository-folder>
pip install -r requirements.txt
```

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
