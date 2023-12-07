import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# 加载数据
# 加载数据
features = np.load('features.npy')
labels = np.load('labels.npy')

# 将数据转换为 PyTorch 张量
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 重塑数据以适应一维卷积网络（batch_size, channels, length）
X_train = X_train.reshape(-1, 1, 1831)
X_test = X_test.reshape(-1, 1, 1831)

# 创建数据加载器
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class ImprovedCNN1D(nn.Module):
    def __init__(self):
        super(ImprovedCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (1831 // 8), 128)  # Adjusting to the new output size
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout for regularization

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)  # Applying dropout
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ImprovedCNN1D()
criterion = nn.BCEWithLogitsLoss()  # 适用于二分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30  # 或更多，根据需要

best_loss = float('inf')
best_model_path = 'best_model.pth'

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # 评估模式
    model.eval()
    total_loss = 0
    # 初始化用于跟踪正确预测的变量
    correct_predictions = 0
    total_predictions = 0

    # 关闭梯度计算
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 预测
            outputs = model(inputs)

            # 将输出转换为二元标签
            predicted_labels = torch.round(torch.sigmoid(outputs.squeeze()))

            # 更新正确预测的计数
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    # 计算准确率
    accuracy = correct_predictions / total_predictions
    print(f'Accuracy: {accuracy:.4f}')

    avg_loss = total_loss / len(test_loader)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')

    # # 如果这个周期的损失低于之前的最佳损失，则保存模型
    # if avg_loss < best_loss:
    #     best_loss = avg_loss
    #     torch.save(model.state_dict(), best_model_path)
    #     print(f'Saved Best Model at Epoch {epoch+1} with Loss: {avg_loss}')

# 将模型设置为评估模式
# model.eval()
#
# # 初始化用于跟踪正确预测的变量
# correct_predictions = 0
# total_predictions = 0
#
# # 关闭梯度计算
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         # 预测
#         outputs = model(inputs)
#
#         # 将输出转换为二元标签
#         predicted_labels = torch.round(torch.sigmoid(outputs.squeeze()))
#
#         # 更新正确预测的计数
#         correct_predictions += (predicted_labels == labels).sum().item()
#         total_predictions += labels.size(0)
#
# # 计算准确率
# accuracy = correct_predictions / total_predictions
# print(f'Accuracy: {accuracy:.4f}')
