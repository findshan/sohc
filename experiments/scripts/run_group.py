import sys, os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from recbole.quick_start import run_recbole
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Comma separated dataset names')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--suffix', type=str, default='group', help='Log suffix')
    return parser.parse_args()

def run_group():
    args = parse_args()
    datasets = args.datasets.split(',')
    gpu_id = args.gpu
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/sasrec_baseline.yaml'))
    seed = 2024
    
    results_file = f'comparison_{args.suffix}.csv'
    all_results = []

    for dataset in datasets:
        dataset = dataset.strip()
        if not dataset: continue
        
        print(f"\n[GPU {gpu_id}] Processing: {dataset}")
        
        # Dataset specific field overrides
        extra_config = {}
        if 'diginetica' in dataset.lower():
            extra_config['USER_ID_FIELD'] = 'session_id'
            extra_config['load_col'] = {'inter': ['session_id', 'item_id', 'timestamp']}
        
        # For extremely large datasets or large item sets, we need smaller batch size to avoid OOM
        if 'ml-20m' in dataset.lower():
            extra_config['train_batch_size'] = 4096
            extra_config['eval_batch_size'] = 4096
        elif 'beauty' in dataset.lower() or 'yelp' in dataset.lower():
            extra_config['train_batch_size'] = 4096
            extra_config['eval_batch_size'] = 1024
        elif 'digital-music' in dataset.lower():
            # This dataset has a very large item set (456k), needs extreme batch size reduction
            extra_config['train_batch_size'] = 2048
            extra_config['eval_batch_size'] = 256

        # 1. Run SASRec
        try:
            print(f"-> Starting SASRec Baseline on {dataset}...")
            sas_res = run_recbole(
                model='SASRec',
                dataset=dataset,
                config_file_list=[config_file],
                config_dict={'seed': seed, 'gpu_id': gpu_id, **extra_config}
            )['test_result']
        except Exception as e:
            print(f"Error SASRec {dataset}: {e}")
            sas_res = {}

        # 2. Run SOHCSASRec
        try:
            print(f"-> Starting SOHCSASRec (Fixed) on {dataset}...")
            sohc_res = run_recbole(
                model='SOHCSASRec',
                dataset=dataset,
                config_file_list=[config_file],
                config_dict={'seed': seed, 'gpu_id': gpu_id, **extra_config}
            )['test_result']
        except Exception as e:
            print(f"Error SOHC {dataset}: {e}")
            sohc_res = {}

        # Metadata
        row_sas = {'dataset': dataset, 'model': 'SASRec', **sas_res}
        row_sohc = {'dataset': dataset, 'model': 'SOHCSASRec', **sohc_res}
        
        all_results.extend([row_sas, row_sohc])
        
        # Incremental save
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        print(f"Done {dataset}. Result comparison saved to {results_file}")

if __name__ == '__main__':
    run_group()
