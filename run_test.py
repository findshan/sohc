import argparse
import os
from recbole.quick_start import run_recbole

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--model', type=str, default='SOHCSASRec', help='Model name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, 'configs/sasrec_baseline.yaml')
    data_path = os.path.join(current_dir, 'dataset_example')
    
    # Overriding config for quick test
    config_dict = {
        'gpu_id': args.gpu,
        'epochs': 2,  # Short run for verification
        'train_batch_size': 2048,
        'data_path': data_path,
        'checkpoint_dir': 'saved',
        'seed': 2024
    }

    print(f"Starting verification run for {args.model} on {args.dataset}...")
    
    try:
        results = run_recbole(
            model=args.model,
            dataset=args.dataset,
            config_file_list=[config_file],
            config_dict=config_dict
        )
        print("\nVerification successful!")
        print("Test Result Summary:", results['test_result'])
    except Exception as e:
        print(f"\nVerification failed: {e}")

if __name__ == '__main__':
    run_test()
