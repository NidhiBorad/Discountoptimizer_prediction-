import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import joblib

# Load data
products = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\PythonProject\data/products_large.csv')
transactions = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\PythonProject\data/transactions_large.csv')

# Merge data
df = transactions.merge(products, on='product_id')

# Encode categorical variables
le_category = LabelEncoder()
le_brand = LabelEncoder()
le_popularity = LabelEncoder()

df['category_enc'] = le_category.fit_transform(df['category'])
df['brand_enc'] = le_brand.fit_transform(df['brand'])
df['popularity_enc'] = le_popularity.fit_transform(df['popularity'])

# Feature selection
X = df[['base_price', 'discount_offered', 'category_enc', 'brand_enc', 'popularity_enc']]
y = df['converted']

# Check class imbalance
print(f"Class distribution in the target variable:\n{y.value_counts()}")

# Calculate class weight ratio for imbalanced classes
ratio = y.value_counts()[0] / y.value_counts()[1]
print(f"Class weight ratio: {ratio}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the XGBoost classifier with scale_pos_weight
xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=ratio)

# Hyperparameter grid for RandomizedSearchCV
# Adjust the hyperparameter grid
param_grid = {
    'n_estimators': [400, 500, 600],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0]
}

# Perform RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Train the model with the best parameters
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Save the tuned model
joblib.dump(best_model, r'C:\Users\lenovo\PycharmProjects\PythonProject\models/tuned_model.pkl')


# Cross-validation to check generalization
cv_scores = cross_val_score(best_model, X, y, cv=3, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean()}")

# Feature Importance Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=best_model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Discount vs Conversion Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='discount_offered', y='converted')
plt.title("Discount vs Conversion Rate")
plt.xlabel("Discount (%)")
plt.ylabel("Conversion Rate")
plt.tight_layout()
plt.savefig("discount_vs_conversion.png")
plt.show()

# Predicted Probability of Conversion (probability of class 1)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print(f"Predicted probability for the first 5 test samples:\n{y_pred_proba[:5]}")
