### train.py

# Import Libraries
import os
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Define main function
def main() -> None:
    # Download penguins dataset
    df = sns.load_dataset("penguins")
    print("Initial dataset shape:", df.shape)

    # Drop missing values
    df = df.dropna()

    ## Encode Catagorical Variables
    # Label encode the (species) target variable
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # One-hot encode (sex and island) input features
    df = pd.get_dummies(df, columns=["sex", "island"])

    # Split into X and y
    X = df.drop("species", axis=1)
    y = df["species"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Training set class distribution:")
    print(y_train.value_counts())

    # Define XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0
    )

    # Train model
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluation
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Print Results
    print("\nTrain classification report:")
    print(classification_report(y_train, train_pred, target_names=le.classes_))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Save outputs
    os.makedirs("app/data", exist_ok=True)

    # Save model
    model.save_model("app/data/model.json") 

    # Save label encoder 
    joblib.dump(le, "app/data/label_encoder.pkl")  

    # Save input columns
    joblib.dump(X.columns.tolist(), "app/data/columns.pkl")  
    print("Artifacts saved in app/data/")


if __name__ == "__main__":
    main()
