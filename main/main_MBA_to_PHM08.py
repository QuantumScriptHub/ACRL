import os
import subprocess
import argparse

def main(args):
    """
    主控脚本，用于启动 MBA 到 PHM08 的领域自适应训练和评估任务。
    """
    print(f"Starting experiment: Source=MBA -> Target=PHM08")

    # --- 设置PYTHONPATH环境变量 ---
    path_to_acrl = '/workspace/ACRL'  
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = path_to_acrl + ":" + my_env.get("PYTHONPATH", "")
    
    # --- 训练命令 ---
    train_command = [
        'python', '-m', 'main.train',
        '--path_src', args.path_src,
        '--path_trg', args.path_trg,
        '--id_src', args.id_src,
        '--id_trg', args.id_trg,
        '--algo_name', 'acrl',
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--eval_batch_size', str(args.eval_batch_size),
        '--learning_rate', str(args.learning_rate),
        '--experiments_main_folder', args.experiments_main_folder,
        '--experiment_folder', args.experiment_folder,
        '--seed', str(args.seed)
    ]

    print("\n--- Running Training ---")
    print(f"Command: {' '.join(train_command)}")
    subprocess.run(train_command, env=my_env, check=True)
    print("--- Training Finished ---\n")

    # --- 评估命令 ---
    eval_command = [
        'python', '-m', 'main.eval',
        '--experiments_main_folder', args.experiments_main_folder,
        '--experiment_folder', args.experiment_folder,
        '--id_src', args.id_src,
        '--id_trg', args.id_trg
    ]
    
    print("--- Running Evaluation ---")
    print(f"Command: {' '.join(eval_command)}")
    subprocess.run(eval_command, env=my_env, check=True)
    print("--- Evaluation Finished ---")
    print("\nExperiment Complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master script for MBA to PHM08 Domain Adaptation experiment")

    parser.add_argument('--path_src', type=str, default='./datasets/MBA', help='Path to Source dataset (MBA)')
    parser.add_argument('--path_trg', type=str, default='./datasets/PHM08', help='Path to Target dataset (PHM08)')
    parser.add_argument('--id_src', type=str, default='mba', help='ID of Source dataset (placeholder)')
    parser.add_argument('--id_trg', type=str, default='phm08', help='ID of Target dataset (placeholder)')
    
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--experiments_main_folder', type=str, default='results')
    parser.add_argument('--experiment_folder', type=str, default='MBA_to_PHM08_exp', help='Folder to save experiment results')

    args = parser.parse_args()
    main(args)