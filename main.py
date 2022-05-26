import os
import csv
import argparse
import ast
import sys
import pickle
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models import *

def arg_as_list(s:str):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a list")
    return v

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--dataset_path', default='data/elec/train.csv', 
                help='dataset directory path: data/elec/train.csv')
parser.add_argument('--test_size', type= float, default= 0.3,
                help= 'test size when splitting the data (float type)')
parser.add_argument('--val_size', type= float, default= 0.1, 
                help= 'validation size when splitting the data (float type)')
parser.add_argument('--normalize', action='store_true', 
                help= 'normailze data')

# Training args
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--recon_penalty', type= float, default= 1e-2, help= 'penalty term of the reconstruction error')
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--model_path', type=str, default='./', help='a path to sava a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')

# Model configs
parser.add_argument('--model_type', type= int, default=0, 
                help= 'model type: 0 = single encoder layer, 1 = double encoder layer')
parser.add_argument('--in_features', type=int, default=8, 
                help= "input features")
parser.add_argument('--latent_dim', type=int, default=4, 
                help= "latent vector dimension")
parser.add_argument('--n_targets', type=int, default=3, 
                help= "the number of targets (tasks)")
parser.add_argument('--n_labels', type=arg_as_list, default=[3, 3, 3], 
                help= "list of the number of labels of each target (task)")
parser.add_argument('--drop_p', type= float, default= 0.2,
                help= "drop out rate in the encoder and decoder layer")
parser.add_argument('--noise_scale', type= float, default= 1e-2, 
                help= 'scale noise')

# Test, Validation configs
# todo if needed.
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

def main(): 
    # load data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, cache = load_elec_data(args.dataset_path, args.test_size, args.val_size, args.normalize)
    
    train_data = MultiTaskDataset(X_train, y_train)
    valid_data = MultiTaskDataset(X_val, y_val)
    test_data = MultiTaskDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)    
    print("Loading data done!")
    
    # model 
    if args.model_type == 0:
        model = MultiClassifictaionTaskModel(args.in_features, args.latent_dim, args.n_targets, 
                        args.n_labels, drop_p= args.drop_p, noise_scale= args.noise_scale).to(device)
    elif args.model_type == 1:
        model = MultiClassifictaionTaskModel2(args.in_features, args.latent_dim, args.n_targets, 
                        args.n_labels, drop_p= args.drop_p, noise_scale= args.noise_scale).to(device)
    else: 
        print('the model is not ready yet...')
        sys.exit()

    # train the model
    optimizer = optim.Adam(model.parameters(), args.lr)
    crossEntropies = []
    for i in range(args.n_targets):
        # compute class weights
        w = compute_class_weight(class_weight='balanced', classes= np.unique(y_train[:, i]), y= y_train[:,i].numpy())
        w = torch.FloatTensor(w).to(device)
        crossEntropies.append(nn.CrossEntropyLoss(w))
    # crossEntropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path
    )

    logs = {
        'tr_loss':[],
        'valid_loss':[]
    }
    num_batches = len(train_loader)
    print('Start training...')
    for epoch in range(args.epoch):
        # to store losses per epoch
        tr_loss, valid_loss = 0, 0
        # a training loop
        for batch_idx, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device) 

            model.train()
            # feed forward
            loss = 0
            with torch.set_grad_enabled(True):
                y_hats, x_hat = model(x)
                for i, y_hat in enumerate(y_hats):
                    loss += crossEntropies[i](y_hat, y[:, i])
                loss += args.recon_penalty * mse(x_hat, x)
            
            # backward 
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store the d_tr_loss
            tr_loss += loss.detach().cpu().item()

            if (batch_idx+1) % args.print_log_option == 0:
                print(f'Epoch [{epoch+1}/{args.epoch}] Batch [{batch_idx+1}/{num_batches}]: \
                    loss = {loss.detach().cpu().item()}')

        # a validation loop 
        for batch_idx, (x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            
            model.eval()
            loss = 0
            with torch.no_grad():
                y_hats, x_hat = model(x)
                for i, y_hat in enumerate(y_hats):
                    loss += crossEntropies[i](y_hat, y[:, i])
                loss += args.recon_penalty * mse(x_hat, x)
            valid_loss += loss.detach().cpu().item()
        
        # save current loss values
        tr_loss, valid_loss = tr_loss/len(train_loader), valid_loss/len(valid_loader)
        logs['tr_loss'].append(tr_loss)
        logs['valid_loss'].append(valid_loss)

        print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {tr_loss:.6f}, validation loss= {valid_loss:.6f}')
        early_stopping(valid_loss, model, epoch, optimizer)

        if early_stopping.early_stop:
            break     
    
    print("Training done! Saving logs...")
    log_path= os.path.join(args.model_path, 'training_logs')
    os.makedirs(log_path, exist_ok= True)
    log_file= os.path.join(log_path, 'training_logs.csv')
    with open(log_file, 'w', newline= '') as f:
        wr = csv.writer(f)
        n = len(logs['tr_loss'])
        rows = np.array(list(logs.values())).T
        wr.writerow(list(logs.keys()))
        for i in range(1, n):
            wr.writerow(rows[i, :])
        # a validation loorp 
    
    labels = np.array([np.arange(n_label) for n_label in args.n_labels])
    cms = [np.zeros((n_label, n_label)) for n_label in args.n_labels] #
    print("Testing the model...")
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            y_hats = model.predict(x).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            for i in range(args.n_targets):
                cms[i] += confusion_matrix(y[:, i], y_hats[:, i], labels= labels[i])
    acc, rec, prec, f1 = evaluate(cms, n_targets= args.n_targets)            
    
    print("Test done!")
    for i in range(args.n_targets):
        disp = ConfusionMatrixDisplay(confusion_matrix=cms[i])
        disp.plot()
        cm_file = os.path.join(args.model_path, f"confusion_matrix_target{i}.png")
        plt.savefig(cm_file)
        plt.show()
        print(f"정확도 (accuracy): {acc[i]:.2f}")
        print(f"재현율 (recall): {rec[i]:.2f}")
        print(f"정밀도 (precision): {prec[i]:.2f}")
        print(f"F1 score: {f1[i]:.2f}")
        print()
                  
if __name__ == '__main__':
    main()