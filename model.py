import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_model(df):
    # Select features and target
    features = ["pclass", "sex", "age", "fare", "embarked"]
    target = "survived"
    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    # Preprocess: impute missing + one-hot encode categorical
    categorical = ["sex", "embarked"]
    numeric = ["pclass", "age", "fare"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train/test split (not really needed for demo, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    return model, features

def predict_survival(model, input_df, features):
    X = input_df[features]
    return int(model.predict(X)[0])
