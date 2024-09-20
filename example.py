import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
# 使用终端下载库 pip install numpy torch torchvision matplotlib tensorflow

# 定义网络结构 这个 Net 类定义了一个简单的四层神经网络，用于分类 MNIST 数据集中的手写数字。这个网络通过四个全连接层来学习输入图像的特征，并输出每个类别的预测概率。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义了第一个全连接层（也称为线性层），它将 784 个输入特征（28x28 像素的手写数字图片被展平成一个向量）映射到 64 个输出特征。
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)  # 定义了第二个全连接层，接收来自 fc1 的 64 个特征，并输出 64 个特征。
        self.fc3 = nn.Linear(64, 64)  # 定义了第三个全连接层，将 64 个输入特征映射到 64 个输出特征。
        self.fc4 = nn.Linear(64, 10)  # 定义了第四个全连接层，将 64 个输入特征映射到 10 个输出特征。

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


# 获取数据加载器
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = datasets.MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


# 评估模型
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            x = x.view(-1, 28 * 28).to(device)
            y = y.to(device)
            outputs = net(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
            n_total += y.size(0)
    return n_correct / n_total


# 主函数
def main():
    # 开始时间
    start_time = time.time()
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)  # 将模型移动到GPU
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 记录初始精度
    initial_accuracy = evaluate(test_data, net)
    print("初始精度 initial accuracy:", initial_accuracy)
    # Neolink.AI平台Tensorflow镜像镜像默认指定的tensorboard-logs地址
    logdir = "/root/tensorboard-logs"
    writer = tf.summary.create_file_writer(logdir)

    for epoch in range(2):
        for x, y in train_data:
            x = x.view(-1, 28 * 28).to(device)
            y = y.to(device)
            net.zero_grad()
            output = net(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            # 记录训练损失
            loss_value = loss.item()
            with writer.as_default():
                tf.summary.scalar('Loss/train', loss_value, step=epoch)

        accuracy = evaluate(test_data, net)
        print("迭代 epoch", epoch, "精度 accuracy:", accuracy)
        # 记录测试准确率
        with writer.as_default():
            tf.summary.scalar('Accuracy/train', accuracy, step=epoch)

    # 显示预测结果
    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net(x[0].view(-1, 28 * 28).to(device)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28).cpu().numpy())  # 将数据移回CPU
        plt.title("预测 prediction: " + str(int(predict)))
    plt.show()

    # 结束时间
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"函数运行时间: {elapsed_time} 秒")

    # 关闭SummaryWriter
    writer.close()

# coder：kana 2024.9.18
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
