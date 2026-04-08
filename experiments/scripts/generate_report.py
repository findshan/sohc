import pandas as pd
import os
import glob

def generate_report():
    # Find all comparison csv files
    csv_files = glob.glob('comparison_*.csv')
    if not csv_files:
        print("No comparison files found.")
        return

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("All dataframes are empty.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Filter out empty rows (where metrics are missing)
    full_df = full_df.dropna(subset=['recall@10'])
    
    if full_df.empty:
        print("No valid metric data found in CSVs yet.")
        return

    # Sort by dataset and model
    full_df = full_df.sort_values(['dataset', 'model'])

    # Create a nice markdown table
    # We focus on recall@10 and ndcg@10 as primary metrics
    metrics = ['recall@10', 'ndcg@10', 'mrr@10', 'hit@10']
    
    output = "# SOHCSASRec vs SASRec Performance Report\n\n"
    output += "| Dataset | Model | Recall@10 | NDCG@10 | MRR@10 | Hit@10 |\n"
    output += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for dataset in full_df['dataset'].unique():
        subset = full_df[full_df['dataset'] == dataset]
        for _, row in subset.iterrows():
            output += f"| {row['dataset']} | {row['model']} | {row['recall@10']:.4f} | {row['ndcg@10']:.4f} | {row['mrr@10']:.4f} | {row['hit@10']:.4f} |\n"
        
        # Calculate Improvement if both exist
        if len(subset) >= 2:
            try:
                sas = subset[subset['model'] == 'SASRec'].iloc[0]
                sohc = subset[subset['model'] == 'SOHCSASRec'].iloc[0]
                imp_recall = (sohc['recall@10'] - sas['recall@10']) / sas['recall@10'] * 100
                imp_ndcg = (sohc['ndcg@10'] - sas['ndcg@10']) / sas['ndcg@10'] * 100
                output += f"| **{dataset}** | **Improvement %** | **{imp_recall:+.2f}%** | **{imp_ndcg:+.2f}%** | | |\n"
            except Exception:
                pass
        output += "| --- | --- | --- | --- | --- | --- |\n"

    with open('experiment_report.md', 'w') as f:
        f.write(output)
    
    print("Report generated: experiment_report.md")

if __name__ == '__main__':
    generate_report()
