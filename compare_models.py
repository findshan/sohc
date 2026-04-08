import argparse
import os
import pandas as pd
from recbole.quick_start import run_recbole

def run_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, 'configs/sasrec_baseline.yaml')
    data_path = os.path.join(current_dir, 'dataset_example')
    
    models = ['SASRec', 'SOHCSASRec']
    results_list = []

    for model_name in models:
        print(f"\n" + "="*50)
        print(f">>> Running {model_name} on {args.dataset}...")
        print("="*50)
        
        config_dict = {
            'gpu_id': args.gpu,
            'epochs': args.epochs,
            'data_path': data_path,
            'checkpoint_dir': 'saved',
            'seed': 2024,
            'state': 'ERROR' # Reduce log clutter
        }

        try:
            res = run_recbole(
                model=model_name,
                dataset=args.dataset,
                config_file_list=[config_file],
                config_dict=config_dict
            )['test_result']
            
            results_list.append({
                'Model': model_name,
                'Recall@10': res['recall@10'],
                'NDCG@10': res['ndcg@10'],
                'MRR@10': res['mrr@10']
            })
        except Exception as e:
            print(f"Error running {model_name}: {e}")

    # Display results
    if results_list:
        df = pd.DataFrame(results_list)
        print("\n\n" + "#" * 40)
        print("# EXPERIMENT COMPARISON SUMMARY")
        print("#" * 40)
        print(df.to_markdown(index=False))
        
        if len(results_list) == 2:
            sas = results_list[0]
            sohc = results_list[1]
            imp_r = (sohc['Recall@10'] - sas['Recall@10']) / sas['Recall@10'] * 100
            imp_n = (sohc['NDCG@10'] - sas['NDCG@10']) / sas['NDCG@10'] * 100
            print(f"\nSOHC Improvement over SASRec:")
            print(f"Recall@10: {imp_r:+.2f}%")
            print(f"NDCG@10: {imp_n:+.2f}%")
        print("#" * 40)

if __name__ == '__main__':
    run_comparison()
