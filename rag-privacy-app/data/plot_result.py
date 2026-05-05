from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CSV_CANDIDATES = [
	BASE_DIR / "results_dynamic_20.csv",
	BASE_DIR / "results_static_20.csv",
	BASE_DIR / "results_cloud_20.csv",
]


def load_available_results() -> pd.DataFrame:
	frames = []
	missing = []

	for path in CSV_CANDIDATES:
		if path.exists():
			frames.append(pd.read_csv(path))
		else:
			missing.append(path.name)

	if not frames:
		raise FileNotFoundError(
			f"未找到任何实验结果 CSV。请确认这些文件是否存在于 {BASE_DIR}: "
			+ ", ".join(p.name for p in CSV_CANDIDATES)
		)

	if missing:
		print(f"⚠️  跳过缺失文件: {', '.join(missing)}")

	return pd.concat(frames, ignore_index=True)


def main() -> None:
	df_all = load_available_results()

	plt.figure(figsize=(10, 6))
	sns.boxplot(x="mode", y="latency_ms", data=df_all)
	plt.tight_layout()
	plt.savefig(BASE_DIR / "latency_boxplot.png", dpi=200)
	plt.close()

	plt.figure(figsize=(10, 6))
	sns.barplot(x="mode", y="rouge_l", data=df_all, errorbar="sd")
	plt.tight_layout()
	plt.savefig(BASE_DIR / "rouge_barplot.png", dpi=200)
	plt.close()

	print("图表已保存")


if __name__ == "__main__":
	main()