import torch
import pickle
from cryptography.fernet import Fernet
from flask import Flask, request
from cryptography.fernet import Fernet
#print(Fernet.generate_key().decode()) 生成密钥
# 加载云侧子模型
from common_model import get_split_models
_, cloud_model = get_split_models()
cloud_model.eval()   # 推理模式

# 生成一个密钥（实际使用中应安全分发，这里演示为固定密钥）
# 注意：端侧和云侧必须使用相同的密钥
KEY = b'e5axOkMH3F-iYmuCvnKyNBAxiRQyaWtWxPoWSQcvPDc='
cipher = Fernet(KEY)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # 接收加密数据
    encrypted_data = request.data
    # 解密
    decrypted = cipher.decrypt(encrypted_data)
    # 反序列化为 numpy 数组，再转为 tensor
    arr = pickle.loads(decrypted)
    tensor = torch.tensor(arr, dtype=torch.float32)
    # 云侧计算
    with torch.no_grad():
        output = cloud_model(tensor)
    # 将输出转换为 numpy 并序列化
    output_np = output.numpy()
    output_bytes = pickle.dumps(output_np)
    # 加密后返回
    encrypted_output = cipher.encrypt(output_bytes)
    return encrypted_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)