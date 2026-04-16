import requests

url = "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/FlashRAG_datasets.zip"
print("正在下载，请稍候...")
response = requests.get(url, stream=True)
if response.status_code != 200:
    print(f"下载失败，HTTP 状态码: {response.status_code}")
    exit(1)

total = int(response.headers.get('content-length', 0))
downloaded = 0
with open("FlashRAG_datasets.zip", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = (downloaded / total) * 100
                print(f"\r进度: {percent:.1f}%", end="")
print("\n下载完成！")