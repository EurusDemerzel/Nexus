import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    files = {
        'Cloud-Only': 'results_cloud_50_clean.csv',
        'Static-Split': 'results_static_50_clean.csv',
        'Nexus-Dynamic': 'results_dynamic_50_clean.csv'
    }
    all_dfs = []
    for mode, file in files.items():
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['Mode'] = mode
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def plot_figures(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Token-F1 箱线图
    plt.figure()
    sns.boxplot(x='Mode', y='token_f1', data=df, palette='Set2')
    plt.title('Token-F1 Comparison Across Modes')
    plt.ylabel('Token-F1 Score')
    plt.xlabel('Execution Mode')
    plt.savefig(os.path.join(output_dir, 'paper_token_f1_box.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 延迟箱线图
    plt.figure()
    sns.boxplot(x='Mode', y='latency_ms', data=df, palette='Set2')
    plt.title('Latency Comparison Across Modes')
    plt.ylabel('Latency (ms)')
    plt.xlabel('Execution Mode')
    plt.yscale('log') # 延迟通常由于长尾效应建议用对数坐标或限制轴
    plt.savefig(os.path.join(output_dir, 'paper_latency_box.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 延迟 vs Token-F1 散点图
    plt.figure()
    sns.scatterplot(x='latency_ms', y='token_f1', hue='Mode', data=df, alpha=0.7, s=100)
    plt.title('Latency vs Token-F1 Trade-off')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Token-F1 Score')
    plt.savefig(os.path.join(output_dir, 'paper_scatter_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {output_dir}")

def generate_stats_table(df):
    stats = df.groupby('Mode').agg({
        'latency_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
        'token_f1': ['mean', 'std'],
        'exact_match': 'mean'
    })
    
    # 重命名列名以便展示
    stats.columns = [
        'Avg Latency (ms)', 'Median Latency (ms)', 'P95 Latency (ms)',
        'Token-F1 (Mean)', 'Token-F1 (Std)', 'EM Accuracy (%)'
    ]
    
    # EM 转换为百分比
    stats['EM Accuracy (%)'] = stats['EM Accuracy (%)'] * 100
    
    # 格式化
    formatted_stats = stats.round(2)
    print("\n### Experimental Results Summary Table")
    print(formatted_stats)
    
    # 生成 LaTeX
    print("\n### LaTeX Table Code")
    print(formatted_stats.to_latex())

if __name__ == "__main__":
    # 假设在 rag-privacy-app 目录下运行
    df = load_data()
    plot_figures(df, "paper/figures")
    generate_stats_table(df)
