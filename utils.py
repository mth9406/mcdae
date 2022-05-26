import torch 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Dataset
class MultiTaskDataset(Dataset):
    """
    MultiTaskDataset
    """
    def __init__(self, X, Y):
        super().__init__()
        self.X, self.Y = X, Y
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

# normalization
def normalize_train(X_train):
    cache = {'mean':None, 'std':None}
    cache['mean'] = X_train.mean(axis=0)
    cache['std'] = X_train.std(axis=0)
    X_train_norm = (X_train-cache['mean'])/cache['std']
    return X_train_norm, cache

def normalize_test(X_test, cache):
    return (X_test-cache['mean'])/cache['std']

def load_elec_data(file= './data/elec/train.csv', test_size= 0.3, val_size= 0.1, normalize= True):
    '''
    Load electronic power anomaly detection data.
    file: a path to the data
    test_size: test_size of "train_test_split"
    normalize: normalize data if True 
    '''
    df = pd.read_csv(file, encoding='cp949')
    # df = df.drop('index', axis= 1)
    print(df.info())
    print('-'*20)

    X, y = df.iloc[:, :8], df.iloc[:, -3:]
    
    # make the categorical values numerical
    # normal: 0, caution: 1 and abnormal: 2
    for col in y.columns:
        y.loc[:, col] = y.loc[:, col].map(
            {'정상':0, '주의':1, '경고':2}
        )

    X, y = torch.FloatTensor(X.values), torch.LongTensor(y.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify= y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify= y_train)
    cache = None

    if normalize:
        X_train, cache = normalize_train(X_train)
        X_val = normalize_test(X_val, cache)
        X_test = normalize_test(X_test, cache)
    else:
        print("--(!) The data is not normalized")

    return X_train, X_val, X_test, y_train, y_val, y_test, cache

def div0( a, b, fill=np.nan ):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
    if np.isscalar( c ):
        return c if np.isfinite( c ) \
            else fill
    else:
        c[ ~ np.isfinite( c )] = fill
        return c

# model evaluation
def evaluate(cm, n_targets):
    """
    Evaluates a model. 
    cm: confusion matrix
    returns accuracy, precision, recall and F1-score.
    """
    # cm
    # column: predicted class
    # row: true label
    acc, rec, prec, f1 = np.zeros(n_targets),  np.zeros(n_targets),\
                         np.zeros(n_targets),  np.zeros(n_targets)

    for i in range(n_targets):
        diag = np.diag(cm[i])
        n_samples = np.sum(cm[i], axis= 1)
        n_preds = np.sum(cm[i], axis= 0)
        # accuracy
        acc[i] = np.sum(diag)/np.sum(cm[i])
        # recall
        rec_ = div0(diag, n_samples, 1)
        rec[i] = np.sum(rec_*n_samples)/np.sum(n_samples)
        # precision
        prec_ = div0(diag, n_preds, 1)
        prec[i] = np.sum(prec_*n_samples)/np.sum(n_samples)
        # f1-score
        f1_ = div0(2*rec_*prec_,(rec_+prec_), 0)
        f1[i] = np.sum(f1_*n_samples)/np.sum(n_samples)

    return acc, rec, prec, f1

class EarlyStopping(object):

    def __init__(self, 
                patience: int= 10, 
                verbose: bool= False, delta: float= 0,
                path = './'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta # significant change
        self.path = os.path.join(path, 'latest_checkpoint.pth.tar')
        self.best_score = None
        self.early_stop= False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer):
        
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
            self.counter = 0


    def save_checkpoint(self, val_loss, ckpt_dict):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        
        torch.save(ckpt_dict, self.path) 
        self.val_loss_min = val_loss