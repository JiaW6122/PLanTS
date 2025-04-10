import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import os
import matplotlib.pyplot as plt

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

def run_epoch(model, classifier, data_loader, is_train, lr=0.001):
    if is_train:
        classifier.train()
    else:
        classifier.eval()
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, (x,y) in enumerate(data_loader):
        y = y.to(device)
        # x = x.to(device)
        x=x.cpu().detach().numpy()
        x=np.transpose(x, (0, 2, 1)) 
        
        encoding=model.encode(x)
        # print(encoding.shape)
        encoding=torch.Tensor(encoding).to(device)
        prediction=classifier(encoding)
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        y_all.append(y.cpu().detach().numpy())
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c

def train_classifier(model, classifier, train_loader, valid_loader, lr, n_epochs=100):
    best_auc, best_acc, best_aupc, best_loss = 0, 0, 0, np.inf
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    for epoch in range(n_epochs):
        train_loss, train_acc, train_auc, train_auprc, _= run_epoch(model, classifier, train_loader, True, lr)
        test_loss, test_acc, test_auc, test_auprc, _ = run_epoch(model, classifier, False)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_loss<best_loss:
            best_auc = test_auc
            best_acc = test_acc
            best_loss = test_loss
            best_aupc = test_auprc
            state = {
                    'epoch': epoch,
                    'state_dict': classifier.state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            data_type="UCI_HAR"
            if not os.path.exists( './ckpt/classifier_test/%s'%data_type):
                os.mkdir( './ckpt/classifier_test/%s'%data_type)
            torch.save(state, './ckpt/classifier_test/%s/%s_checkpoint_%d.pth.tar'%(data_type, type))
    # Save performance plots
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses, label="train Loss")
    plt.plot(np.arange(n_epochs), test_losses, label="test Loss")

    plt.plot(np.arange(n_epochs), train_accs, label="train Acc")
    plt.plot(np.arange(n_epochs), test_accs, label="test Acc")
    plt.savefig(os.path.join("./plots/%s" % data_type, "classification_%s_%d.pdf"%(type)))
    return best_acc, best_auc, best_aupc



def eval_state_prediction(model, train_data, train_labels, test_data, test_labels, lr, window_size=4, n_states=6, encoding_size=256):
    # train_repr = model.encode(train_data)
    # test_repr = model.encode(test_data)
    model.eval()
    classifier=StateClassifier(input_size=encoding_size, output_size=n_states)
    train_loader, valid_loader, test_loader=create_dataset(train_data, train_labels, test_data, test_labels, window_size)
    ###train
    best_acc, best_auc, best_aupc = train_classifier(model, classifier, train_loader, valid_loader, lr)
    print('Best_acc:', best_acc,'Best_auc:', best_auc, 'Best_aupc:', best_aupc)

    ###test
    _, test_acc, test_auc, test_aupc,_ =run_epoch(model, classifier, test_loader, False)
    print('=======> Performance Summary:')
    print('State Prediction: \t Accuracy:\t AUC: \t AUPRC:'%
          (100 * test_acc, 100 * test_auc, test_aupc))
    












    
    
    