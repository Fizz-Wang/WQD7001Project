import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data from a CSV file
data = pd.read_csv('updated_imputed_file.csv')

# 2. Create a function to categorize models into specific categories
def categorize_model(model):
    if model == 'Model 3':
        return 'Model 3'
    elif model == 'Model Y':
        return 'Model Y'
    elif model == 'Model X':
        return 'Model X'
    elif model == 'Model S':
        return 'Model S'
    else:
        return 'Others'

# 3. Apply the categorization function to the 'Model' column
data['Model_Categorized'] = data['Model'].apply(categorize_model)

# 4. Convert the categorized models into numerical labels using LabelEncoder
label_encoder = LabelEncoder()
data['Model_encoded'] = label_encoder.fit_transform(data['Model_Categorized'])

# 5. Select features (X) and target (y)
#    - In this example, we're using 'Sale Price' and 'Electric Range' as features
#    - 'Model_encoded' is our target variable
X = data[['Sale Price', 'Electric Range']]
y = data['Model_encoded']

# 6. Split the dataset into training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Initialize the Random Forest Classifier
#    - n_estimators=100 means we will use 100 decision trees in the forest
#    - random_state=42 ensures reproducibility
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 8. Fit (train) the Random Forest model on the training set
rf_classifier.fit(X_train, y_train)

# 9. Predict on the testing set
y_pred = rf_classifier.predict(X_test)

# 10. Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 11. Print a detailed classification report
#     - Shows precision, recall, f1-score for each class
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 12. Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Use a heatmap for better visualization
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 13. Display feature importance for each feature used in the model
importances = rf_classifier.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.2f}")

# 14. Predict the model category for new, unseen data
#     - For example, a car with a sale price of 50,000 and an electric range of 450
new_data = pd.DataFrame({'Sale Price': [50000], 'Electric Range': [450]})
predicted_model_encoded = rf_classifier.predict(new_data)
predicted_model = label_encoder.inverse_transform(predicted_model_encoded)
print(f"Predicted Model: {predicted_model[0]}")

# save
joblib.dump(rf_classifier, 'model_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved.")