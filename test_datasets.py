import pandas as pd
import numpy as np
import os

def create_messy_segmented():
    # Record spans 3 rows
    data = [
        ["ID", "Name", None, None],
        ["Date", "Score", None, None],
        ["Description", None, None, None],
        ["1", "Alice", None, None],
        ["2023-01-01", "95", None, None],
        ["Top scorer", None, None, None],
        ["2", "Bob", None, None],
        ["2023-01-02", "88", None, None],
        ["Runner up", None, None, None],
    ]
    df = pd.DataFrame(data)
    df.to_csv("messy_segmented.csv", index=False, header=False)

def create_messy_positional():
    # Shifted/sparse values
    data = [
        ["ID", "Name", "Score", "Date", None],
        ["1", "Alice", None, None, None],
        [None, None, "95", "2023-01-01", None],
        ["2", "Bob", None, None, None],
        [None, None, "88", "2023-01-02", None],
    ]
    df = pd.DataFrame(data)
    df.to_csv("messy_positional.csv", index=False, header=False)
    
def create_messy_streaming():
    # Random stream flow without strict boundaries
    data = [
        ["ID", "Name", "Age"],
        ["1", "Alice", "30", "2"],
        ["Bob", "25", "3", "Charlie"],
        ["35"]
    ]
    df = pd.DataFrame(data)
    df.to_csv("messy_streaming.csv", index=False, header=False)

if __name__ == "__main__":
    create_messy_segmented()
    create_messy_positional()
    create_messy_streaming()
    print("Created messy_segmented.csv, messy_positional.csv, messy_streaming.csv")
