import os
import random
import codecs
import numpy as np
import math
import argparse

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import torch
from torch import optim, cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

import models.resnet
import models.densenet


parser = argparse.ArgumentParser(description='PyTorch code: Multi-step Gaussian Noise Detector')
parser.add_argument('--model', required=True, help='resnet | densenet')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--batch_size', type=int, default=10000, metavar='N', help='batch size of data loader')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--multigpu', default=False, help='use data parallel or not')
parser.add_argument('--noise_step', type=int, default=10, help='the number of noise magnitude steps')
parser.add_argument('--noise_N', type=int, default=30, help='the number of noise images generated from each step')
args = parser.parse_args()
#args = parser.parse_args(args=[]) # for jupyter
print(args)

class MyBiLSTM(nn.Module):
    def __init__(self, feature_size, hidden_dim, n_layers):
        super(MyBiLSTM, self).__init__()

        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_output = 2
        self.lstm = nn.LSTM(feature_size, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, self.n_output)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim, device=x.device))
        c_0 = Variable(torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim, device=x.device))
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, (hidden, internal state))
        y = self.fc(torch.cat([hn[0],hn[1]],dim=1))
        return y


def create_loader(robustness_AEs, clean, step, cv): # step 
    clean = [[clean[i][0][args.classNum*j:args.classNum*(j+1)].tolist() for j in  step] for i in range(len(clean))]
    AEs = [[robustness_AEs[i][0][args.classNum*j:args.classNum*(j+1)].tolist() for j in step] for i in range(len(robustness_AEs))]
    
    # 5 cross validation
    N = int(len(clean)/5) 
    clean_tr, AEs_tr = clean[:N*cv]+ clean[N*(cv+1):], AEs[:N*cv]+ AEs[N*(cv+1):]
    clean_val, AEs_val = clean[N*cv : int(N*(cv+0.5))], AEs[N*cv : int(N*(cv+0.5))]
    clean_ts, AEs_ts = clean[int(N*(cv+0.5)):N*(cv+1)], AEs[int(N*(cv+0.5)):N*(cv+1)]
    
    train_data = torch.Tensor(clean_tr + AEs_tr)
    valid_data = torch.Tensor(clean_val + AEs_val)
    test_data = torch.Tensor(clean_ts + AEs_ts)
    
    train_labels = torch.Tensor([[1,0] for _ in range(len(clean_tr))]  + [[0,1] for _ in range(len(AEs_tr))])
    valid_labels = torch.Tensor([[1,0] for _ in range(len(clean_val))]  + [[0,1] for _ in range(len(AEs_val))])
    test_labels = torch.Tensor([[1,0] for _ in range(len(clean_ts))]  + [[0,1] for _ in range(len(AEs_ts))])

    dataset_train = TensorDataset(train_data, train_labels)
    dataset_valid = TensorDataset(valid_data, valid_labels)
    dataset_test = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(dataset_train, batch_size=min(len(train_data), args.batch_size), shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=min(len(valid_data), args.batch_size), shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=min(len(test_data), args.batch_size), shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train_LSTM(LSTMmodel, train_loader, valid_loader): 
    criterion = F.cross_entropy
    optimizer = optim.Adam(LSTMmodel.parameters(), lr=0.005)
    stop_count = 0
    max_acc = 0
    record_loss_train = []
    
    #train
    for i in range(2001):
        #train
        LSTMmodel.train()
        torch.manual_seed(2023)
        for j, (x, t) in enumerate(train_loader):
            opt = LSTMmodel(x.to(args.device))
            loss = criterion(opt, t.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # eval
        with torch.no_grad(): 
            LSTMmodel.eval()
            correct = 0
            l = 0
            for j, (x, t) in enumerate(valid_loader):
                opt = LSTMmodel(x.to(args.device)) 
                correct+= (torch.argmax(opt,dim=-1).cpu() == torch.argmax(t,dim=-1).cpu()).sum() 
                l += len(opt) 

            # early stopping
            acc = correct/l
            if max_acc <= acc:
                max_acc = acc
                best = LSTMmodel.state_dict()
                stop_count = 0 
            else:
                stop_count += 1
                
            if stop_count > 10:
                break
                
    return best, max_acc
    

def train_sequance(clean, AEs, step, cv): 
    train_loader, valid_loader, test_loader = create_loader(clean, AEs, step, cv) 

    # define detection model
    feature_size, n_hidden, n_layers  = args.classNum, 256*int(math.log10(args.classNum)), 1
    LSTMmodel = MyBiLSTM(feature_size, n_hidden, n_layers).to(args.device)
    if args.multigpu:
        LSTMmodel = torch.nn.DataParallel(LSTMmodel) # make parallel
        torch.backends.cudnn.benchmark = True

    temp_dict, temp_acc = train_LSTM(LSTMmodel, train_loader, valid_loader) #これまでの best dict

    return temp_dict, temp_acc 



def calculate_AUROC(best_dict, test_loader):
    #bestモデルの読み込み
    feature_size, n_hidden, n_layers  = args.classNum, 256*int(math.log10(args.classNum)), 1
    LSTMmodel = MyBiLSTM(feature_size, n_hidden, n_layers).to(args.device)
    if args.multigpu:
        LSTMmodel = torch.nn.DataParallel(LSTMmodel) # make parallel
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    LSTMmodel.load_state_dict(best_dict) 
    LSTMmodel.eval()
    
    Score=[]
    Labels = []
    with torch.no_grad():
        for j, (x, t) in enumerate(test_loader):
            opt = LSTMmodel(x.to(args.device)).cpu()
            Sx = nn.Softmax(dim=1) #softmax
            Score += Sx(opt).cpu()
            Labels += t
            
    Score = [Score[i][0] for i in range(len(Score))]
    Labels = [Labels[i][0] for i in range(len(Labels))]
        
    # calculate AUROC 
    roc = roc_curve(Labels, Score) # roc
    score = roc_auc_score(Labels, Score) # auroc
    return score


def serch_best_step(model, dataset, cv, TXT):
    np.random.seed(seed=2023)
    torch.manual_seed(2023)
    
    ## データの読み込み (特徴量データ)
    data_path = "robustness/" + model +"_"+ dataset + "_randomN" + str(args.noise_N) + "_step" + str(args.noise_step)
    [robustness_train, robustness_FGSM_clean, robustness_PGD_clean, robustness_BIM_clean,
     robustness_DeepFool_clean, robustness_CW_clean, robustness_FGSM, robustness_PGD, robustness_BIM, 
     robustness_DeepFool, robustness_CW] = torch.load(data_path)

    AEs_list = [robustness_BIM_clean, robustness_CW_clean, robustness_DeepFool_clean, robustness_FGSM_clean, 
            robustness_PGD_clean, robustness_BIM, robustness_CW, robustness_DeepFool, robustness_FGSM, 
            robustness_PGD]
    
    AUROC_list =[]
    final_step_l =[]
    

    for i in range(5): # each AE
        print(model,dataset,cv,i, end = "",file=codecs.open(TXT, 'a', 'utf-8'))
        best_acc = 0 # best ceckpoint of LSTM
        candidate_step = [i+1 for i in range(args.noise_step)] # 0.1~1
        final_step = [0] #  "0" means without gaissian noise
        
        while True: 
            print("final_step", final_step, file=codecs.open(TXT, 'a', 'utf-8'))
            local_best_acc_l = []
            end_flg = True
            
            for j in range(len(candidate_step)): #候補全部試す
                temp_step = final_step + [candidate_step[j]] #
                temp_step.sort()
                print(candidate_step[j], end = " ",file=codecs.open(TXT, 'a', 'utf-8'))
                local_best_dict, local_best_acc  = train_sequance(AEs_list[i], AEs_list[i+5], temp_step, cv)
                local_best_acc_l.append(local_best_acc)
                
                if local_best_acc > best_acc: #一番良かったやつを保存
                    best_dict = local_best_dict
                    best_acc = local_best_acc
                    idx = j
                    end_flg = False
            
            Max_kouho_acc = max(local_best_acc_l) #一番良かったやつを選択
            
            if end_flg: #何の更新もなかった時
                break 
                
            else: # Max_kouho_acc == best_acc:
                #print("idx",idx, Max_kouho_acc, end = "", file=codecs.open(TXT, 'a', 'utf-8'))
                final_step.append(candidate_step.pop(idx))  #満を侍して追加
                local_best_acc_l.pop(idx) #
                final_step.sort() #一応ソート
                
            if len(candidate_step) == 0: # all use
                break 
                
        print("final_step", final_step, file=codecs.open(TXT, 'a', 'utf-8'))
        _, _, test_loader = create_loader(AEs_list[i], AEs_list[i+5], final_step, cv) 
        score = calculate_AUROC(best_dict, test_loader)
        print(score, file=codecs.open(TXT, 'a', 'utf-8'))
        AUROC_list.append(score)  
        final_step_l.append(final_step)
        
    return AUROC_list, final_step_l


def main():
    args.classNum = 100 if args.dataset == "cifar100" else 10  # Number of classes
    AUROC_cv = []
    TXT = 'result/log_SV_LSTM_' + args.model + "_" + args.dataset +"_randomN" + str(args.noise_N) + "_step" + str(args.noise_step) + '.txt'
    for cv in range(5): # 5-cross validation
        AUROC_score, final_step = serch_best_step(args.model, args.dataset, cv, TXT) 
        AUROC_cv.append(AUROC_score) # 5*5
    
    torch.save([final_step, AUROC_cv], 'result/SV_LSTM_'+ args.model + "_" + args.dataset +"_randomN" + str(args.noise_N) + "_step" + str(args.noise_step) + '.pt')
            
if __name__ == '__main__':
    main()

