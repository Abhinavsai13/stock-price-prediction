import pandas as pd
from datetime import datetime

def run_merge(
    daily_file, yearly_file, output_file
):
    daily_df = pd.read_csv(daily_file, parse_dates=["Date"])
    yearly_df = pd.read_csv(yearly_file, index_col=0)

    def parse_year(col):
        try:
            return datetime.strptime(col, "Mar %Y")
        except:
            return None

    year_cols = {col: parse_year(col) for col in yearly_df.columns if "Mar" in col}
    yearly_df.columns = [year_cols.get(c, c) for c in yearly_df.columns]

    yearly_df_T = yearly_df.T
    yearly_df_T.index = pd.to_datetime(yearly_df_T.index)
    yearly_df_T = yearly_df_T.sort_index()

    merged_df = daily_df.copy()
    merged_df = merged_df.sort_values("Date")

    def map_year_to_date(date):
        fiscal_years = yearly_df_T.index
        prev_years = fiscal_years[fiscal_years <= date]
        if len(prev_years) == 0:
            return fiscal_years[0]
        return prev_years[-1]

    for metric in yearly_df_T.columns:
        merged_df[metric] = merged_df["Date"].apply(lambda d: yearly_df_T.loc[map_year_to_date(d), metric])

    merged_df.to_csv(output_file, index=False, float_format="%.4f")
    print(f"✅ Merged dataset saved as: {output_file}")
    print(f"Shape: {merged_df.shape}")
    print("Example rows:")
    print(merged_df.head(5))
    return output_file

if __name__ == "__main__":
    run_merge("ITC.NS_daily_technical.csv", "ITC_ML_Dataset.csv", "ITC.NS_daily_technical_merged.csv")
