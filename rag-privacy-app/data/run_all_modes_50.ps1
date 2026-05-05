# NEW: 批量运行三种模式（各 50 条）
python data/run_experiments.py --mode=cloud-only --limit 50 --output results_cloud_50.csv
python data/run_experiments.py --mode=static-split --limit 50 --output results_static_50.csv
python data/run_experiments.py --mode=nexus-dynamic --limit 50 --output results_dynamic_50.csv
