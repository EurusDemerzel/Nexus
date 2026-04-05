import torch
import pickle
import requests
from cryptography.fernet import Fernet
from common_model import get_split_models

# 加载端侧子模型
end_model, _ = get_split_models()
end_model.eval()

# 密钥（必须与云侧一致）
KEY = b'e5axOkMH3F-iYmuCvnKyNBAxiRQyaWtWxPoWSQcvPDc='
cipher = Fernet(KEY)

def load_image():
    """生成一个模拟的随机图像（28x28），实际可替换为真实图片"""
    # 随机生成，范围0~1
    img = torch.randn(1, 1, 28, 28)
    return img

def main():
    # 1. 加载输入
    x = load_image()
    print("输入形状:", x.shape)

    # 2. 端侧计算中间张量
    with torch.no_grad():
        intermediate = end_model(x)
    print("中间张量形状:", intermediate.shape)

    # 3. 序列化 + 加密
    intermediate_np = intermediate.numpy()
    data_bytes = pickle.dumps(intermediate_np)
    encrypted = cipher.encrypt(data_bytes)

    # 4. 发送给云侧
    url = "http://127.0.0.1:5000/predict"
    response = requests.post(url, data=encrypted)
    if response.status_code != 200:
        print("云侧请求失败")
        return

    # 5. 解密并反序列化结果
    encrypted_output = response.content
    decrypted_output = cipher.decrypt(encrypted_output)
    output_np = pickle.loads(decrypted_output)
    output_tensor = torch.tensor(output_np)

    # 6. 得到预测类别
    pred_class = torch.argmax(output_tensor, dim=1).item()
    print("云侧返回的原始输出:", output_tensor)
    print("预测类别:", pred_class)

if __name__ == '__main__':
    main()