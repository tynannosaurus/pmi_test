import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load your data
data = pd.read_csv('/Users/tynanp/Documents/pmi_2/data.csv')

# Prepare the feature matrix and target vector
X = data[['Max', 'Min', 'Average', 'Ambient', 'Humidity']]
y = data['PMI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree model with the best hyperparameters
best_tree_model = DecisionTreeRegressor(max_depth=7, min_samples_leaf=2, min_samples_split=2)
best_tree_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the trained model to a file
joblib.dump(best_tree_model, 'decision_tree_model.pkl')

