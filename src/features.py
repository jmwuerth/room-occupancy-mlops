import pandas as pd
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def extract_time_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts hour (0-23) and day_of_week (0-6) from a datetime column."""
    time_df = pd.DataFrame(index=df.index)
    dt_series = pd.to_datetime(df["date"])
    time_df["hour"] = dt_series.dt.hour
    time_df["day_of_week"] = dt_series.dt.day_of_week
    return time_df


def build_encoder() -> OneHotEncoder:
    """Creates a deterministic OneHotEncoder covering all 24 hours and 7 days."""
    categories = [list(range(24)), list(range(7))]
    return OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="ignore")


def transform_features(df: pd.DataFrame, encoder: OneHotEncoder, fit: bool = False) -> pd.DataFrame:
    """Combines numeric sensor telemetry with one-hot encoded datetime features."""
    time_cats = extract_time_categories(df)

    if fit:
        encoded_array = encoder.fit_transform(time_cats)
    else:
        encoded_array = encoder.transform(time_cats)

    encoded_cols = encoder.get_feature_names_out(["hour", "day_of_week"])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    numeric_df = df[FEATURE_COLS].astype(float)
    return pd.concat([numeric_df, encoded_df], axis=1)
