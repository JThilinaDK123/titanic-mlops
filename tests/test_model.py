import pandas as pd
import seaborn as sns
from model import train_model, predict_survival

def test_model_training_and_prediction():
    df = sns.load_dataset("titanic").dropna(subset=["survived"])
    model, features = train_model(df)

    sample = pd.DataFrame([{
        "pclass": 3, "sex": "male", "age": 25, "fare": 7.25, "embarked": "S"
    }])

    pred = predict_survival(model, sample, features)
    assert pred in [0, 1]
