import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import os
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

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
    T = train_data.shape[-1]
    x_window = np.split(train_data[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window = np.concatenate(np.split(train_labels[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))

    x_window_test = np.split(test_data[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window_test = np.concatenate(np.split(test_labels[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window_test = torch.Tensor(np.concatenate(x_window_test, 0))
    y_window_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window_test]))

    testset = torch.utils.data.TensorDataset(x_window_test, y_window_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    shuffled_inds = list(range(len(x_window)))
    random.shuffle(shuffled_inds)
    x_window = x_window[shuffled_inds]
    y_window = y_window[shuffled_inds]
    n_train = int(0.7*len(x_window))
    X_train, X_test = x_window[:n_train], x_window[n_train:]
    y_train, y_test = y_window[:n_train], y_window[n_train:]

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    validset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=200, shuffle=False)

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
        
        encoding=model.encode(x, encoding_window='full_series')
        # print(encoding.shape)
        encoding=torch.Tensor(encoding).to(device)
        classifier=classifier.to(device)
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
        test_loss, test_acc, test_auc, test_auprc, _ = run_epoch(model, classifier, valid_loader, False)
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
            if not os.path.exists( 'ckpt/classifier_test/%s'%data_type):
                os.mkdir( 'ckpt/classifier_test/%s'%data_type)
            torch.save(state, 'ckpt/classifier_test/%s/%s_checkpoint.pth.tar'%(data_type, type))
    # Save performance plots
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses, label="train Loss")
    plt.plot(np.arange(n_epochs), test_losses, label="test Loss")

    plt.plot(np.arange(n_epochs), train_accs, label="train Acc")
    plt.plot(np.arange(n_epochs), test_accs, label="test Acc")
    plt.legend()

    plt.savefig(os.path.join("figures/%s" % data_type, "classification_%s.pdf"%(type)))
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
    print(f'State Prediction:\tAccuracy: {100 * test_acc:.2f}\tAUC: {100 * test_auc:.2f}\tAUPRC: {test_aupc:.2f}')


def tracking_encoding(model, data, labels, subject_id=1, window_size=4, sliding_gap=5):
    data = np.transpose(data, (0, 2, 1)) 
    sample = data[subject_id]
    label = labels[subject_id]
    T = data.shape[-1]
    encodings = []
    windows_label = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    model.eval()
    for t in range(window_size//2,T-window_size//2,sliding_gap):
        windows = sample[:, t-(window_size//2):t+(window_size//2)]
        windows_label.append((np.bincount(label[t-(window_size//2):t+(window_size//2)].astype(int)).argmax()))
        inputs=np.expand_dims(windows,axis=0)
        inputs=np.transpose(inputs, (0, 2, 1)) 
        encoding=model.encode(inputs, encoding_window='full_series')
        encoding=torch.from_numpy(encoding).to(torch.float)
        # print(encoding.squeeze(0).shape)
        encodings.append(encoding.squeeze(0))
    for t in range(window_size//(2*sliding_gap)):
        # fix offset
        encodings.append(encodings[-1])
        encodings.insert(0, encodings[0])
    # print(len(encodings))
    encodings = torch.stack(encodings, 0)
    # print(encodings.shape)
    pca = PCA(n_components=5)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())

    f, axs = plt.subplots(2)  # , gridspec_kw={'height_ratios': [1, 2]})
    f.set_figheight(10)
    f.set_figwidth(27)
    # print(sample.shape)
    for feat in range(min(sample.shape[0], 10)):
        sns.lineplot(x=np.arange(sample.shape[1]), y=sample[feat], ax=axs[0])

    axs[0].set_title('Time series Sample Trajectory', fontsize=30, fontweight='bold')
    axs[0].xaxis.set_tick_params(labelsize=22)
    axs[0].yaxis.set_tick_params(labelsize=22)
    axs[-1].xaxis.set_tick_params(labelsize=22)
    axs[-1].yaxis.set_tick_params(labelsize=22)
    axs[-1].set_ylabel('Encoding dimensions', fontsize=28)
    axs[0].margins(x=0)
    axs[0].grid(False)
    t_0 = 0
    for t in range(1, label.shape[-1]):
        if label[t]==label[t-1]:
            continue
        else:
            axs[0].axvspan(t_0, min(t+1, label.shape[-1]-1), facecolor=['y', 'g', 'b', 'r', 'c', 'm'][int(label[t_0])], alpha=0.5)
            t_0 = t
    axs[0].axvspan(t_0, label.shape[-1]-1 , facecolor=['y', 'g', 'b', 'r'][int(label[t_0])], alpha=0.5)
    axs[-1].set_title('Encoding Trajectory', fontsize=30, fontweight='bold')
    sns.heatmap(embedding.T, cbar=False, linewidth=0.5, ax=axs[-1], linewidths=0.05, xticklabels=False)
    f.tight_layout()
    plt.show()
    # sns.heatmap(encodings.detach().cpu().numpy().T, linewidth=0.5)
    plt.savefig(os.path.join("figures/UCI_HAR/embedding_trajectory_hm.pdf"))


    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'state':windows_label}#, 'label':windows_label}
    df = pd.DataFrame(data=d)
    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    # sns.jointplot(x="f1", y="f2", data=df, kind="kde", size='time', hue='label')
    sns.scatterplot(x="f1", y="f2", data=df, hue="state")
    plt.show()
    plt.savefig(os.path.join("figures/UCI_HAR/embedding_trajectory_scatter.pdf"))











    
    
    