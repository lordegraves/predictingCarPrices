# Predicting Car Prices

## Project Overview
The goal of this project is to develop a machine learning model that can accurately predict the market value of cars based on their characteristics. By analyzing a comprehensive dataset of car features and prices, we aim to provide valuable insights for buyers, sellers, and enthusiasts to understand what factors most significantly affect a car's price.

### Data Source
The dataset for this project was curated from a vast collection of datasets utilized for AI testing, stored on a private Google Drive. Due to privacy considerations, the direct link to the dataset is not available.

## Getting Started

### Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn

### Installation
Clone this repository to your local machine to get started:

## Workflow

### 1. Process the Data
Data preprocessing is crucial to prepare the raw data for modeling. This step includes:
- Handling missing values
- One-hot encoding categorical variables
- Normalizing or scaling numerical features
- Splitting the dataset into training and testing sets

### 2. Create Models
Several machine learning models were explored to predict car prices, including:
- **Linear Regression**
- **Decision Trees**
- **Random Forest Regressor**

### 3. Evaluate Models
Each model's performance was evaluated using the **R-squared metric** and **Mean Squared Error (MSE)** to compare their prediction accuracy on the test data.

### 4. Pick Most Promising Models
Based on initial evaluation metrics, the **Random Forest Regressor** was identified as the most promising model due to its superior balance of accuracy and generalization.

### 5. Tune Models
The Random Forest Regressor model was further tuned using `GridSearchCV` to find the optimal set of hyperparameters, enhancing its performance.

#### Best Parameters Found:
- `n_estimators`: 350
- `min_samples_split`: 2
- `min_samples_leaf`: 1
- `max_features`: None
- `max_depth`: None
- `bootstrap`: True

### 6. Evaluate Tuned Models
The tuned Random Forest model demonstrated significant improvement:
- **Train Score**: 0.9693505557318618
- **Test Score**: 0.7986109106211777

### 7. Create Pipeline
A pipeline was established to simplify the processing of future data, encapsulating data preprocessing and model prediction steps, ensuring new data can be easily assessed with the trained model.

## Further Improvements
To further increase the model's accuracy, the following strategies can be explored:
- Additional **feature engineering** to uncover more insights from the data.
- Experimenting with **more complex models** or **ensemble methods**.
- Using **cross-validation techniques** to fine-tune hyperparameters further.
- Investigating **potential data biases or outliers** that may affect model performance.

## Conclusion
This project demonstrates the effectiveness of machine learning in predicting car prices based on their features. The developed Random Forest model, after careful tuning, offers a robust solution for accurately estimating car values, aiding stakeholders in making informed decisions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
