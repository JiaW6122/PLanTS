import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = torch.nn.BatchNorm1d(self.input_size)
        self.nn = torch.nn.Linear(self.input_size, self.output_size)
        torch.nn.init.xavier_uniform_(self.nn.weight)

    def forward(self, x):
        x = self.normalize(x)
        logits = self.nn(x)
        return logits

def create_dataset(train_data, train_labels, test_data, test_labels, window_size=4,batch_size=100):
    train_data = np.transpose(train_data, (0, 2, 1)) 
    test_data = np.transpose(test_data, (0, 2, 1))
    n_train = int(0.8*len(train_data))
    n_valid = len(train_data) - n_train
    n_test = len(test_data)
    x_train, y_train = train_data[:n_train], train_labels[:n_train]
    x_valid, y_valid = train_data[n_train:], train_labels[n_train:]
    x_test = test_data
    y_test = test_labels

    datasets = []
    for set in [(x_train, y_train, n_train), (x_test, y_test, n_test), (x_valid, y_valid, n_valid)]:
        T = set[0].shape[-1]
        windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
        windows = np.concatenate(windows, 0)
        labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1)
        labels = np.round(np.mean(np.concatenate(labels, 0), -1))
        datasets.append(data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    trainset, testset, validset = datasets[0], datasets[1], datasets[2]
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

def train_classifier(model, classifier, train_loader, lr=0.001, ):
    classifier.train()
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        encoding=model.encode(x)
        prediction=classifier(encoding)
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        loss.backward()
        optimizer.step()
        y_all.append(y)
        prediction_all.append(prediction.detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c



def eval_state_prediction(model, train_data, train_labels, test_data, test_labels, window_size=4, n_states=6, encoding_size=256):
    # train_repr = model.encode(train_data)
    # test_repr = model.encode(test_data)
    model.eval()
    classifier=StateClassifier(input_size=encoding_size, output_size=n_states)
    train_loader, valid_loader, test_loader=create_dataset(train_data, train_labels, test_data, test_labels, window_size)
    












    
    
    