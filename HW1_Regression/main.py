import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_selection import SelectKBest, f_regression

TRAIN_PATH = "./dataset/covid.train.csv"
TEST_PATH = "./dataset/covid.test.shuffle.csv"
MODE = "train"
BATCH_SIZE = 128
DEVICE = "cuda"
MAX_EPOCH = 3000


class Covid19Dataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.FloatTensor(x)
        self.ground_truth = torch.FloatTensor(y)

        # self._normalization()
        # self._get_most_relative(x, y, -10)
        self._standardization()

        self.dim = self.data.shape[1]

    def __getitem__(self, item):
        return self.data[item], self.ground_truth[item]

    def __len__(self):
        return len(self.data)

    def _standardization(self):
        mean = torch.mean(self.data[:, 40:], dim=0)
        std = torch.std(self.data[:, 40:], dim=0)
        self.data[:, 40:] = (self.data[:, 40:] - mean) / std

    def _normalization(self):
        maximum = torch.max(self.data[:, 40:])
        minimum = torch.min(self.data[:, 40:])
        self.data[:, 40:] = (self.data[:, 40:] - minimum) / (maximum - minimum)
        # denominator = torch.sqrt(torch.diag(torch.matmul(self.data, self.data.transpose(0, 1))))
        # self.data = torch.div(self.data.transpose(0, 1), denominator).transpose(0, 1)

    def _get_most_relative(self, x, y, k):
        bestfeatures = SelectKBest(score_func=f_regression, k=5)
        fit = bestfeatures.fit(x, y)
        index = fit.scores_.argsort()[k:]
        index = np.concatenate((np.arange(40), index))
        self.data = self.data[:, index]


class Covid19Model(nn.Module):
    def __init__(self, input_dim):
        super(Covid19Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.model(x).squeeze(1)

    def get_loss(self, pred, gt):
        return self.criterion(pred, gt)


def load_dataset(path):
    with open(path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])[:, 1:].astype(np.float64)
    return data


def split_dataset(data):
    np.random.shuffle(data[:])
    flag = int(data.shape[0] * .9)
    return data[:flag, :-1], data[:flag, -1], data[flag:, :-1], data[flag:, -1]


def train():
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.001, momentum=0.9)
    history = {'train': [], 'dev': []}
    min_mse = 1000
    early_stop_cnt = 0
    for epoch in range(MAX_EPOCH):
        train_model.train()
        for x, y in train_dataset:
            optimizer.zero_grad()                 # set gradient to zero
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = train_model(x)                 # forward pass (compute output)
            loss = train_model.get_loss(pred, y)  # compute loss
            loss.backward()                       # compute gradient (backpropagation)
            optimizer.step()                      # update model with optimizer
            history["train"].append(loss.detach().cpu().item())
        valid_mse = valid()
        if valid_mse < min_mse:
            # Save model if your model improved
            min_mse = valid_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(train_model.state_dict(), "model.pth")  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        history["dev"].append(valid_mse)
        # print("epoch {:4d}: loss = {:.4f}".format(epoch + 1, valid_mse))
        if early_stop_cnt > 200:
            print("early stop: epoch = {}".format(epoch))
            break
    print("iteration complete. min_mse = {:.4f}".format(min_mse))
    return min_mse, history


def valid():
    train_model.eval()
    total_loss = 0
    for x, y in valid_dataset:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            pred = train_model(x)
            loss = train_model.get_loss(pred, y)
        total_loss += loss.detach().cpu().item() * len(x)
    total_loss /= len(valid_dataset.dataset)
    return total_loss


def test():
    test_model.eval()
    preds = list()
    for x, _ in test_dataset:
        x = x.to(DEVICE)
        with torch.no_grad():
            pred = test_model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


# reference from NTU ML2021 Spring colab
# URL: https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb#scrollTo=BtE3b6JEH7rw
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


# reference from NTU ML2021 Spring colab
# URL: https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb#scrollTo=g0pdrhQAO41L
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == '__main__':
    raw = load_dataset(TRAIN_PATH)
    train_data, train_gt, valid_data, valid_gt = split_dataset(raw)
    train_dataset = DataLoader(Covid19Dataset(train_data, train_gt),
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               drop_last=False)
    valid_dataset = DataLoader(Covid19Dataset(valid_data, valid_gt),
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               drop_last=False)
    train_model = Covid19Model(train_dataset.dataset.dim).to(DEVICE)
    model_loss, model_loss_record = train()
    plot_learning_curve(model_loss_record, title='deep model')

    raw = load_dataset(TEST_PATH)
    test_dataset = DataLoader(Covid19Dataset(raw, np.array([-1 for i in range(len(raw))])),
                              batch_size=BATCH_SIZE,
                              drop_last=False)
    test_model = Covid19Model(test_dataset.dataset.dim).to(DEVICE)
    test_model.load_state_dict(torch.load("./model.pth", map_location="cpu"))
    pred = test()
    plot_pred(valid_dataset, test_model, DEVICE)
    save_pred(pred, "./output/pred.csv")
