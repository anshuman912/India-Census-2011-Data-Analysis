"""Step-by-step Pandas analysis for the logistics dataset.

Run from the folder containing the outputs directory:
    python logistics_analysis.py
"""

from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "outputs" / "logistics_dataset_50000.csv"
ANALYSIS_DIR = BASE_DIR / "work" / "analysis_assets"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Step 1: Load the CSV f
    df = pd.read_csv(DATA_FILE)

    # Step 2: Inspect the dataset
    print("Dataset shape:", df.shape)
    print("\nFirst five records:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # Step 3: Clean the data
    df = df.drop_duplicates().dropna()
    numeric_columns = [
        "Distance_KM", "Delivery_Time_Hours", "Transportation_Cost",
        "Inventory_Level", "Demand"
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_columns)

    # Step 4: Create useful calculated fields
    df["Cost_per_KM"] = df["Transportation_Cost"] / df["Distance_KM"]
    df["Delay_Flag"] = (df["Delay"] == "Yes").astype(int)

    # Step 5: Descriptive statistics
    print("\nDescriptive statistics:\n", df[numeric_columns].describe().round(2))

    # Step 6: Warehouse-level analysis
    warehouse_summary = (
        df.groupby("Warehouse")
        .agg(
            Orders=("Order_ID", "count"),
            Average_Distance_KM=("Distance_KM", "mean"),
            Average_Delivery_Hours=("Delivery_Time_Hours", "mean"),
            Average_Transportation_Cost=("Transportation_Cost", "mean"),
            Delay_Rate=("Delay_Flag", "mean"),
        )
        .sort_values("Orders", ascending=False)
        .round(2)
    )
    warehouse_summary["Delay_Rate"] = warehouse_summary["Delay_Rate"] * 100
    print("\nWarehouse summary:\n", warehouse_summary)

    # Step 7: Destination-city analysis
    city_summary = (
        df.groupby("Destination_City")
        .agg(
            Orders=("Order_ID", "count"),
            Average_Cost=("Transportation_Cost", "mean"),
            Average_Delivery_Hours=("Delivery_Time_Hours", "mean"),
            Delay_Rate=("Delay_Flag", "mean"),
        )
        .sort_values("Orders", ascending=False)
        .round(2)
    )
    city_summary["Delay_Rate"] = city_summary["Delay_Rate"] * 100
    print("\nTop destination cities:\n", city_summary.head(10))

    # Step 8: Correlation analysis
    correlation = df[numeric_columns + ["Cost_per_KM", "Delay_Flag"]].corr().round(3)
    print("\nCorrelation matrix:\n", correlation)

    # Step 9: Save analysis tables for reuse
    warehouse_summary.to_csv(ANALYSIS_DIR / "warehouse_summary.csv")
    city_summary.to_csv(ANALYSIS_DIR / "city_summary.csv")
    correlation.to_csv(ANALYSIS_DIR / "correlation_matrix.csv")

    # Step 10: Save the key results used in the report
    metrics = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "average_distance": round(float(df["Distance_KM"].mean()), 2),
        "average_delivery_hours": round(float(df["Delivery_Time_Hours"].mean()), 2),
        "average_transportation_cost": round(float(df["Transportation_Cost"].mean()), 2),
        "overall_delay_rate": round(float(df["Delay_Flag"].mean() * 100), 2),
        "highest_delay_warehouse": warehouse_summary["Delay_Rate"].idxmax(),
        "highest_delay_rate": round(float(warehouse_summary["Delay_Rate"].max()), 2),
        "lowest_delay_warehouse": warehouse_summary["Delay_Rate"].idxmin(),
        "lowest_delay_rate": round(float(warehouse_summary["Delay_Rate"].min()), 2),
        "largest_order_warehouse": warehouse_summary["Orders"].idxmax(),
        "largest_order_count": int(warehouse_summary["Orders"].max()),
        "distance_cost_correlation": round(float(correlation.loc["Distance_KM", "Transportation_Cost"]), 3),
    }
    (ANALYSIS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nAnalysis completed. Files saved in:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()


