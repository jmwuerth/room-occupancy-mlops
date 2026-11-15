import os
import re
import pandas as pd

EXPECTED_COLUMNS = [
    "date",
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "Occupancy",
]


def _normalize_quotes(text: str) -> str:
    """Removes standard and curly quotes from raw text lines."""
    return text.replace('"', "").replace("“", "").replace("”", "").strip()


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Parses raw telemetry files, repairs split lines and dot-delimiters,
    and returns a strictly typed pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Telemetry file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as file:
        raw_lines = [_normalize_quotes(line) for line in file if line.strip()]

    if len(raw_lines) < 2:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    cleaned_records = []
    line_iter = iter(raw_lines[1:])  # Skip header row

    for line in line_iter:
        tokens = [t.strip() for t in line.split(",")]

        # Fix 1: Handle row split across two lines (e.g., '1,2015-02-11' + '14:48:00,...')
        if len(tokens) == 2 and re.match(r"^\d{4}-\d{2}-\d{2}$", tokens[1]):
            try:
                next_line = next(line_iter)
                next_tokens = [t.strip() for t in next_line.split(",")]
                tokens = [tokens[0], f"{tokens[1]} {next_tokens[0]}"] + next_tokens[1:]
            except StopIteration:
                continue

        # Fix 2: Repair dot-delimiter mutation merging time and temperature (14:50:00.21.7675)
        if len(tokens) > 1:
            dot_match = re.match(
                r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\.(\d+\.\d+)$", tokens[1]
            )
            if dot_match:
                tokens[1] = dot_match.group(1)
                tokens.insert(2, dot_match.group(2))

        # Strip leading row index if 8 columns exist (index + 7 data fields)
        if len(tokens) == 8:
            record = tokens[1:8]
        elif len(tokens) == 7:
            record = tokens
        else:
            continue

        cleaned_records.append(record)

    df = pd.DataFrame(cleaned_records, columns=EXPECTED_COLUMNS)

    # Coerce types and drop any corrupted rows that failed conversion
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")
    df = df.dropna()
    df["Occupancy"] = df["Occupancy"].astype(int)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    sample_df = load_and_clean_data("data/datatest2.txt")
    print(f"Cleaned rows: {len(sample_df)}")
    print(sample_df.head())
