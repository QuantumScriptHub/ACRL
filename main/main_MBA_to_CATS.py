import os
import subprocess
import argparse

def main(args):
    """
    主控脚本，用于启动 MBA 到 CATS 的领域自适应训练和评估任务。
    (版本：适配 'python -m' 模块化运行方式)
    """
    # 更新打印信息以反映新的目标域
    print(f"Starting experiment: Source=MBA -> Target=CATS")

    # --- 设置PYTHONPATH环境变量 ---
    path_to_acrl = '/workspace/ACRL'  
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = path_to_acrl + ":" + my_env.get("PYTHONPATH", "")
    
    # --- 1. 定义训练命令 (调用 train.py) ---
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
        '--num_channels_TCN', args.num_channels_TCN,
        '--hidden_dim_MLP', str(args.hidden_dim_MLP),
        '--weight_domain', str(args.weight_domain),
        '--weight_loss_pred', str(args.weight_loss_pred),
        '--experiments_main_folder', args.experiments_main_folder,
        '--experiment_folder', args.experiment_folder,
        '--seed', str(args.seed)
    ]

    # --- 2. 执行训练 ---
    print("\n--- Running Training ---")
    print(f"Command: {' '.join(train_command)}")
    subprocess.run(train_command, env=my_env, check=True)
    print("--- Training Finished ---\n")

    # --- 3. 定义评估命令 (调用 eval.py) ---
    eval_command = [
        'python', '-m', 'main.eval',
        '--experiments_main_folder', args.experiments_main_folder,
        '--experiment_folder', args.experiment_folder,
        '--id_src', args.id_src,
        '--id_trg', args.id_trg
    ]
    
    # --- 4. 执行评估 ---
    print("--- Running Evaluation ---")
    print(f"Command: {' '.join(eval_command)}")
    subprocess.run(eval_command, env=my_env, check=True)
    print("--- Evaluation Finished ---")
    print("\nExperiment Complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master script for MBA to CATS Domain Adaptation experiment")

    parser.add_argument('--path_src', type=str, default='./datasets/MBA', help='Path to Source dataset (MBA)')
    parser.add_argument('--path_trg', type=str, default='./datasets/CATS', help='Path to Target dataset (CATS)')
    parser.add_argument('--id_src', type=str, default='mba', help='ID of Source dataset (placeholder for MBA)')
    parser.add_argument('--id_trg', type=str, default='cats', help='ID of Target dataset (placeholder for CATS)')
    
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_channels_TCN', type=str, default='128-256-512', help='TCN channels (e.g., "64-64-64")')
    parser.add_argument('--hidden_dim_MLP', type=int, default=1024, help='MLP hidden dimension')
    parser.add_argument('--weight_domain', type=float, default=0.1, help='Weight of domain discriminator loss')
    parser.add_argument('--weight_loss_pred', type=float, default=1.0, help='Weight of Deep SVDD loss')
    
    parser.add_argument('--experiments_main_folder', type=str, default='results')
    parser.add_argument('--experiment_folder', type=str, default='MBA_to_CATS_exp', help='Folder to save experiment results')

    args = parser.parse_args()
    main(args)