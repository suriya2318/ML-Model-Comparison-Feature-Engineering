# Feature Engineering, Model Optimization & Performance Comparison (Enhanced House Price Prediction with Multi-Model Comparison)

## Project Objective
The goal of this project is to build an enhanced House Price Prediction system that goes beyond basic model training by applying professional-grade feature engineering, preprocessing techniques, and structured model comparison. This task focuses on how real-world Machine Learning engineers improve model performance by preparing data correctly, training multiple algorithms simultaneously, and selecting the best-performing model using measurable evaluation metrics. By the end of this project, the complete ML optimization workflow is demonstrated — from raw data preparation to justified model selection. 

## Dataset Used

•	California Housing Dataset (sklearn.datasets — 1990 U.S. Census) <a href="https://github.com/suriya2318/California-House-Price-Predictor/blob/main/California_Housing%20Datasets.csv"> Dataset</a>  

•	Feature Set: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

•	Target Variable: HousePrice (Median House Value in $100,000s)

•	Total Records: 20,640 housing block samples across California

## Key Business Questions (KPIs)

•	Which Machine Learning algorithm produces the most accurate house price predictions?

•	How much does feature scaling improve model stability and prediction accuracy?

•	What is the RMSE and R² score for each model on unseen test data?

•	Which model has the lowest average prediction error (MAE) in dollar terms?

•	Does Ridge Regression reduce overfitting compared to standard Linear Regression?

•	How well does a Decision Tree capture non-linear relationships in housing data?

•	Which model best balances accuracy and generalizability for deployment?

•	What percentage of house price variation does each model explain?

•	Which model should be selected for production use and why?

## Process

### Data Loading & Preparation
• Loaded the California Housing Dataset using sklearn.datasets.fetch_california_housing with as_frame=True.

• Renamed the target column from MedHouseVal to HousePrice for clarity.

• Verified all 20,640 rows — confirmed zero missing values across all 8 features.

• Reviewed column names, data types, and statistical distributions using df.info() and df.describe().

### Feature Engineering & Scaling
• Separated input features (X) from the target variable (y) before any transformation.

• Applied StandardScaler to all 8 input features to normalize them to mean=0 and std=1.

• Scaling was applied before the train-test split to ensure all features contribute equally to model learning regardless of their original numeric range.

• Without scaling, high-range features like Population would dominate the model unfairly over low-range features like HouseAge.

### Train-Test Split
• Split the scaled dataset into 80% training (16,512 rows) and 20% testing (4,128 rows) using train_test_split with random_state=42.

• Used the same split across all three models to ensure a fair and consistent performance comparison.

• Testing on unseen data prevents the model from simply memorizing training values.

### Multi-Model Training & Evaluation
• Trained three regression algorithms in a single automated loop for efficiency and consistency.

• Evaluated every model using three standard metrics: RMSE, R² Score, and MAE.

• Stored all results in a structured dictionary and converted to a comparison DataFrame for clean display.

• Selected the best-performing model based on highest R² and lowest RMSE on test data.

### Visualization & Reporting
• Built a bar chart comparing RMSE and R² scores across all three models side by side.

• Plotted Actual vs. Predicted scatter chart for the best-performing model.

• Generated a Residual Plot to validate that prediction errors are randomly distributed around zero.

• Saved the best model and scaler as pickle files using joblib for future deployment use.

## Methodology & Results
The project followed a structured and reproducible Machine Learning pipeline designed to ensure fair, objective, and professional model comparison. All three models were trained on identical scaled data using the same 80/20 train-test split with random_state=42, ensuring that performance differences are purely due to the algorithm itself and not data variation.

### Feature Scaling Result
• Before scaling: MedInc values ranged 0.5–15, Population values ranged 3–35,682 — a massive numeric imbalance that would cause high-range features to dominate model learning unfairly.

• After StandardScaler: all 8 features normalized to mean=0 and std=1 — every feature contributes equally to the model's learning process.

• Impact: Scaling improved linear model stability and ensured Ridge regularization penalized coefficients fairly across all features.

### Train-Test Split Result
• Total dataset: 20,640 rows split into 16,512 training rows (80%) and 4,128 testing rows (20%).

• All three models evaluated on the identical 4,128 unseen test rows — no data leakage, no memorization.

• random_state=42 ensures the experiment is fully reproducible by anyone running the notebook.

### Model Training Result
• All three models trained in a single automated loop — same input data, same evaluation metrics, same test set applied identically to every algorithm.

• Linear Regression converged immediately as expected for an ordinary least squares solution.

• Ridge Regression with alpha=1.0 applied L2 regularization — shrinking large coefficients to reduce overfitting risk.

• Decision Tree with max_depth=5 created up to 32 leaf nodes — deep enough to capture non-linear patterns while shallow enough to avoid overfitting.

## Model Performance Comparison Table

![](https://github.com/suriya2318/ML-Model-Comparison-Feature-Engineering/blob/main/Model%20Performance%20Comparison%20Table.png)

### Best Model Selected: Decision Tree Regressor
• Highest R² Score of 0.7517 — explains 75.17% of house price variation on unseen test data.

• Lowest RMSE of 0.6498 — average prediction error of approximately $64,980 per house.

• Lowest MAE of 0.4721 — median prediction is off by only $47,210 on average.

• Outperformed Linear Regression by +8.57% in R² Score and reduced RMSE by 0.1041 points.

• Ridge and Linear Regression produced nearly identical results (R² difference of only 0.0001), confirming that regularization provided minimal benefit on this particular dataset.

### Why Decision Tree Won
• House prices in California are heavily influenced by non-linear geographic and income interactions — for example, a high-income block near the coast is worth exponentially more than a high-income inland block.

• Linear models can only draw straight hyperplanes through the feature space and cannot capture these curved, threshold-based relationships.

• The Decision Tree splits data at specific feature thresholds — for example, blocks where MedInc > 5.0 AND Latitude < 35.5 get routed to a completely different prediction branch from similar-income inland blocks.

• This threshold-based splitting naturally mirrors how real estate pricing actually works in the market.

## Charts & Visualizations Overview

### Chart 1 — Model Comparison (RMSE)
• Side-by-side horizontal bar chart showing RMSE score for all three models simultaneously.

• Decision Tree bar is visibly shorter than both linear model bars, making the performance winner immediately obvious at a glance.

• Each bar labeled with its exact RMSE value for precise comparison.

• Color coded: Blue for Linear Regression, Teal for Ridge Regression, Amber for Decision Tree.

![Model Comparison](https://github.com/suriya2318/ML-Model-Comparison-Feature-Engineering/blob/main/Model%20Performance%20Comparison.png)

### Chart 2 — Model Comparison (R² Score)
• Side-by-side bar chart showing R² Score for all three models simultaneously.

• Decision Tree bar is visibly taller, confirming it explains significantly more price variation than both linear models.

• Paired with the RMSE chart in a single figure to give a complete two-metric performance overview in one visual.

![Model Comparison](https://github.com/suriya2318/ML-Model-Comparison-Feature-Engineering/blob/main/Model%20Performance%20Comparison.png)

### Chart 3 — Actual vs Predicted (Best Model)
• Scatter plot comparing the Decision Tree's predictions against real house prices on the 4,128 test rows.

• Red dashed diagonal line represents perfect prediction — dots closer to this line mean more accurate predictions.

• R² = 0.7517 annotation displayed directly on the chart for immediate context.

• Tighter clustering around the diagonal line compared to the linear model scatter confirms superior Decision Tree accuracy.

![Actual vs Predicted](https://github.com/suriya2318/ML-Model-Comparison-Feature-Engineering/blob/main/Visual%20performance%20Validation.png)

### Chart 4 — Residual Plot (Best Model)
• Plots the difference between actual and predicted values against predicted values for the Decision Tree.

• Red dashed horizontal line at zero — ideal residuals scatter randomly and symmetrically around this line.

• Mild positive residuals visible at higher predicted price ranges, indicating slight under-prediction of expensive properties.

• This pattern confirms that even the best model struggles with very high-end properties — a clear signal that ensemble methods like Random Forest are the logical next step.

![Residual Plot](https://github.com/suriya2318/ML-Model-Comparison-Feature-Engineering/blob/main/Visual%20performance%20Validation.png)

### Chart 5 — Feature Scaling Before vs After Comparison
• Side-by-side view of raw feature values versus scaled feature values for the first row of data.

• Clearly demonstrates how StandardScaler eliminates the massive numeric imbalance between features.

• Visually explains why scaling is a non-negotiable preprocessing step for any distance-sensitive or gradient-based ML algorithm.

## Models Overview

### Model 1: Linear Regression (Baseline)
• Standard ordinary least squares regression — finds the best-fitting straight-line relationship between all 8 features and the target price.

• Used as the baseline to measure how much improvement other models provide.

• Fast to train, highly interpretable, and suitable for linearly separable data.

• Limitation: Cannot capture non-linear patterns in the housing data.

### Model 2: Ridge Regression (Regularized Linear Model)
• An extension of Linear Regression that adds an L2 penalty term (alpha=1.0) to the loss function during training.

• The penalty shrinks large coefficients and reduces overfitting — especially useful when input features are correlated with each other.

• Produces more stable and generalizable coefficients than standard Linear Regression.

• Best used when multicollinearity exists between features such as AveRooms and AveBedrms.

### Model 3: Decision Tree Regressor (Non-Linear Model)
• A tree-based algorithm that splits the dataset into increasingly specific subgroups based on feature thresholds, capturing non-linear relationships.

• Trained with max_depth=5 to prevent overfitting while still learning complex patterns in the data.

• Does not require feature scaling to function correctly but was trained on scaled data for consistency.

• Capable of modeling interactions between features that linear models fundamentally cannot represent.

## Project Insights
• Feature scaling had a significant positive impact on model stability — without StandardScaler, features with large numeric ranges dominated the learning process unfairly.

• The Decision Tree Regressor outperformed both linear models by capturing non-linear relationships between features and house prices that straight-line models cannot represent.

• Ridge Regression produced nearly identical results to standard Linear Regression on this dataset, indicating that multicollinearity is not a major issue in the California Housing 
features.

• Both linear models (Linear and Ridge) plateau around R² = 0.60–0.67, confirming that house price prediction requires non-linear modeling for meaningful accuracy improvement.

• The Residual Plot for the best model shows mild heteroscedasticity at higher price ranges, suggesting that even the Decision Tree under-predicts very expensive properties.

• A structured automated training loop across all three models ensures fair comparison — same data split, same scaling, same evaluation metrics applied consistently.

• The model comparison table clearly demonstrates that RMSE and R² together give a more complete picture of model quality than either metric alone.

• Saving the best model with joblib enables direct deployment — the saved model can predict new house prices without retraining from scratch.

## Final Conclusion
This Feature Engineering, Model Optimization and Performance Comparison project successfully demonstrates how professional Machine Learning engineers approach model improvement in real-world projects. By applying StandardScaler for feature normalization, training three distinct regression algorithms (Linear Regression, Ridge Regression, and Decision Tree Regressor), and evaluating all models on identical unseen test data using RMSE, R², and MAE metrics, a fully objective and reproducible model selection process was achieved. The Decision Tree Regressor emerged as the best-performing model by capturing non-linear patterns in the California Housing data that linear models fundamentally cannot model. This project establishes the critical skills of preprocessing correctness, multi-model comparison, metric-based decision making, and professional model justification — all essential competencies for any production-level Machine Learning workflow. Future improvements include training Random Forest and Gradient Boosting models, applying cross-validation for more reliable performance estimates, and implementing hyperparameter tuning using GridSearchCV to further push model accuracy beyond the current benchmark.

