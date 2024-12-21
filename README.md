# LLM_Customer_Support_Ticket_Insights
 Predicting Ticket Resolutions Using Sentiment Analysis and Logistic Regression
This project focuses on developing a predictive model to classify ticket resolutions based on customer sentiment and ticket attributes. By leveraging a cleaned dataset containing customer demographic information, ticket priority, and sentiment scores, we trained a logistic regression model to evaluate the relationship between these variables and resolution outcomes. The insights and predictions were integrated into a Flask-based web application to render a REST API for seamless deployment and user interaction.

The project encompasses data preprocessing, exploratory data analysis, feature engineering, model training, API development using Flask, and performance evaluation. Advanced debugging and feature analysis were performed to refine the model and ensure interpretability, scalability, and actionable insights.

# Key Insights
1. Sentiment Analysis:

The correlation between sentiment score and resolution is weak (-0.0115), suggesting sentiment alone is not a strong predictor of resolution.
Sentiment score contributes negatively to resolution probability, as indicated by its coefficient (-0.178).

2. Class Imbalance:

A single resolution class (1518) dominates the dataset, causing class imbalance. Addressing this imbalance is critical for improving predictions across all resolution types.

3. Model Performance:

The initial logistic regression model achieved 65.91% accuracy for negative sentiment cases and 70.11% accuracy after introducing Sentiment Magnitude.
Model coefficients reveal that ticket priority (-1.095) has a stronger impact on resolution compared to sentiment score.

4. Feature Engineering:

Adding Sentiment Polarity and Sentiment Magnitude features slightly improved model performance.
Interaction terms (e.g., Sentiment Score × Ticket Priority) can further enhance predictions by capturing complex relationships.

5. Negative Sentiment Segment:

The model performs moderately well on negative sentiment cases (65.91% accuracy), but deeper analysis is needed to reduce misclassifications.

6. Visualization:
![priority_output.png](static%2Fpriority_output.png)
![satisfaction_output.png](static%2Fsatisfaction_output.png)
![sentiment_output.png](static%2Fsentiment_output.png)
![status_output.png](static%2Fstatus_output.png)
![wordcloud_output.png](static%2Fwordcloud_output.png)

The sentiment score distribution reveals clusters of negative, neutral, and positive scores, guiding feature engineering.
Visualizing predicted probabilities showed a clear downward trend for sentiment scores, validating the coefficient's impact.

7. Flask API:

A Flask-based REST API was developed to deploy the predictive model, enabling real-time ticket resolution predictions.
The API is designed for easy integration with existing customer support systems, providing predictions based on input features such as Customer Age, Ticket Priority, and Sentiment Score.

# Next Steps
1. Address class imbalance using oversampling, undersampling, or class-weighted modeling techniques.

2. Experiment with advanced algorithms like Random Forest or XGBoost for potential performance gains. 

3. Perform hyperparameter tuning and feature selection to optimize the model. 

4. Explore additional data sources or features (e.g., resolution time or customer feedback) for improved predictions. 

5. Extend the Flask app by integrating a frontend dashboard for visualizing predictions and insights interactively.

This project demonstrates how combining sentiment analysis, machine learning, and API development can streamline customer support processes by providing accurate and actionable ticket resolution predictions.







