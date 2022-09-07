import pandas as pd
def get_numerical_columns(df: pd.DataFrame) -> list:
    numerical_columns = df.select_dtypes(include='number').columns.tolist()
    return numerical_columns

def get_categorical_columns(df: pd.DataFrame) -> list:
    categorical_columns = df.select_dtypes(
        include=['object']).columns.tolist()
    return categorical_columns