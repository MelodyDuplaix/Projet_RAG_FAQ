import json
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=None)
def load_faq_data(path="data/faq-base.json"):
    """
    Loads the FAQ data from the specified JSON file into a pandas DataFrame.
    Uses caching to avoid reloading the file on subsequent calls.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        faq_df = pd.DataFrame(data["faq"])
        if 'id' not in faq_df.columns:
            raise ValueError("The FAQ data must contain an 'id' column.")
        return faq_df
    except FileNotFoundError:
        return pd.DataFrame()
    except (json.JSONDecodeError, ValueError) as e:
        # Log the error for debugging purposes if needed
        # print(f"Error loading FAQ data: {e}")
        return pd.DataFrame()
