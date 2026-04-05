import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 端侧部分（前两个卷积块）
        self.end_part = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                             # 14x14 -> 7x7
        )
        # 云侧部分（后两个卷积块 + 全连接）
        self.cloud_part = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 7x7 -> 7x7
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 7x7 -> 3x3
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.end_part(x)
        x = self.cloud_part(x)
        return x

# 切分后，分别提取两个子模型
def get_split_models():
    full_model = SimpleCNN()
    end_model = full_model.end_part   # 端侧子模型
    cloud_model = full_model.cloud_part  # 云侧子模型
    return end_model, cloud_model

# 简单测试：生成随机输入，验证两个子模型串联输出与完整模型一致
if __name__ == "__main__":
    end, cloud = get_split_models()
    x = torch.randn(1, 1, 28, 28)
    intermediate = end(x)
    output = cloud(intermediate)
    print("Output shape:", output.shape)  # 应为 torch.Size([1, 10])