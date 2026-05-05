import pandas as pd
import os

def clean_data(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return 0
    
    df = pd.read_csv(file_path)
    initial_count = len(df)
    
    # 过滤无效样本：超时 (latency_ms > 120000) 或 token_f1 == -1
    # 注意：有些 CSV 可能没有 token_f1 列，视具体情况而定
    mask = (df['latency_ms'] <= 120000)
    if 'token_f1' in df.columns:
        mask &= (df['token_f1'] != -1)
    
    df_clean = df[mask].copy()
    final_count = len(df_clean)
    
    df_clean.to_csv(output_path, index=False)
    print(f"File: {file_path}")
    print(f"  Initial: {initial_count}, Cleaned: {final_count}, Removed: {initial_count - final_count}")
    return final_count

if __name__ == "__main__":
    # 脚本在 rag-privacy-app 目录下运行
    base_dir = "."
    files = [
        ("results_cloud_50.csv", "results_cloud_50_clean.csv"),
        ("results_static_50.csv", "results_static_50_clean.csv"),
        ("results_dynamic_50.csv", "results_dynamic_50_clean.csv")
    ]
    
    for input_file, output_file in files:
        clean_data(os.path.join(base_dir, input_file), os.path.join(base_dir, output_file))
