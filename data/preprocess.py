from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/network_traffic/MachineLearningCVE")
OUT = Path("data/ids2018")

def main():

    csv_files = list(RAW.glob("*"))
    csv_files = [f for f in csv_files if f.is_file()]

    if len(csv_files) == 0:
        print("❌ No CSV files found.")
        return
 

    dfs = []

    for file in csv_files:
        print(f"Loading : {file.name}")
        df = pd.read_csv(file, low_memory=False)

        df.columns = df.columns.str.strip()

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    data = data.drop_duplicates()
    data = data.dropna()

    print(f"\nTotal Samples : {len(data):,}")

    train, temp = train_test_split(
        data,
        test_size=0.30,
        random_state=42,
        stratify=data["Label"]
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        stratify=temp["Label"]
    )

    OUT.mkdir(exist_ok=True)

    train.to_csv(OUT/"train.csv", index=False)
    val.to_csv(OUT/"val.csv", index=False)
    test.to_csv(OUT/"test.csv", index=False)

    print("\n==============================")
    print("Dataset Split Completed")
    print("==============================")
    print("Train :", len(train))
    print("Val   :", len(val))
    print("Test  :", len(test))

if __name__ == "__main__":
    main()