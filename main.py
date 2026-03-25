import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "Area": np.random.randint(500, 5000, 200),
    "Bedrooms": np.random.randint(1, 6, 200),
    "Bathrooms": np.random.randint(1, 4, 200),
    "Location_Score": np.random.randint(1, 10, 200),
    "Age": np.random.randint(0, 30, 200),
}

df = pd.DataFrame(data)

df["Price"] = (
    df["Area"] * 300 +
    df["Bedrooms"] * 50000 +
    df["Bathrooms"] * 30000 +
    df["Location_Score"] * 40000 -
    df["Age"] * 10000
)

df.to_csv("housing_data.csv", index=False)

print("CSV file created!")
