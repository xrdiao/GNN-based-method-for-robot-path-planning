import time
import numpy as np
from tqdm import tqdm
import torch
from module_3d import MyGnn, NodeEncoder, MapEncoder
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


def check_succ(model_, data_test, data_map, y, sample):
    y_hat = model_(data_test, data_map)
    y_hat = y_hat.cpu()
    label_hat = torch.argmax(y_hat[sample], dim=1).cpu()
    label = torch.argmax(y[sample], dim=1).cpu()
    train_acc = np.array(torch.tensor(label_hat[:] == label[:], dtype=torch.int64)).flatten()

    # print(train_acc.sum())
    return train_acc.sum() / len(y)


# 模型拟合失败是谁的问题？1、数据集，2、模型，3、训练的过程
def my_train(model, opt, _loss, my_data, map, len_train, samples):
    assert my_data is not None
    model.train()
    success_rate = 0
    loss_all = 0
    index = 0
    for data, m in zip(my_data, map):
        opt.zero_grad()
        data.to(device)
        m.to(device)
        output = model(data.x, data.edge_index, m.x)
        label = data.y
        loss = _loss(output, label)
        loss.backward()

        loss_all += data.num_graphs * loss.item()
        opt.step()
        success_rate += check_succ(model_=model, data_test=data, data_map=m, y=label, sample=samples[index])
        index += 1

    return loss_all / len_train, success_rate / len_train


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = torch.load("data/train.pth")
    # test = torch.load("data/test.pth")
    maps = torch.load("data/map.pth")
    samples = np.load('data/samples.npy')

    # 超参量
    in_channel = 3 * 7  # feature的维度，在build中设置为6，可以进行重建
    out_channel = 3  # label的维度
    batch = 1
    encode_size = 64
    cuda = False  # 是否使用GPU，现在为测试阶段还没往那去

    model = MyGnn(in_channel=in_channel, out_channel=out_channel, batch=128, encode_size=encode_size, cuda=cuda)
    # model.load_state_dict(torch.load('model/encoder.pth'))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_function = torch.nn.BCELoss()
    loss_function.to(device)
    losses = []

    print('start train')

    for epoch in range(5):
        t1 = time.time()
        loss, success = my_train(model, opt, loss_function, train, maps, len(train.dataset), samples)
        losses.append(loss)
        t2 = time.time()
        print('epoch =', epoch + 1, 'loss =', loss, 'sucess_rate =', success, 'time =', t2 - t1)

    plt.plot(losses)
    # # print(model_n(train.dataset[0]))
    torch.save(model.state_dict(), 'model/encoder.pth')
    plt.show()
