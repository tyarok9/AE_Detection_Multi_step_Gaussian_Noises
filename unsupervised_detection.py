import os
import random
import numpy as np
import argparse

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import svm
import models.resnet
import models.densenet
import torch

parser = argparse.ArgumentParser(description='PyTorch code: Multi-step Gaussian Noise Detector')
parser.add_argument('--model', required=True, help='resnet | densenet')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--batch_size', type=int, default=10000, metavar='N', help='batch size for data loader')
parser.add_argument('--multigpu', default=False, help='use data parallel or not')
parser.add_argument('--noise_step', type=int, default=10, help='the number of noise magnitude steps')
parser.add_argument('--use_noise_step', type=list, default=[0,1,2], help='noise steps used in unsupervised detection')
parser.add_argument('--noise_N', type=int, default=30, help='the number of noise images generated from each step')
args = parser.parse_args()


def train_OCSVM(robustness_train, step):
    train_data = [[robustness_train[i][0][args.classNum*j:args.classNum*(j+1)].tolist() for j in  step] for i in range(len(robustness_train))]
    train_data = torch.tensor(train_data).view(-1, args.classNum*len(step))

    clf = svm.OneClassSVM(nu = 0.25, kernel='rbf', gamma = 'auto')
    clf.fit( train_data[:20000] )
    return clf


def calculate_AUROC(clf, clean, robustness_AEs, step):
    clean = torch.tensor([[clean[i][0][args.classNum*j:args.classNum*(j+1)].tolist() for j in  step] for i in range(len(clean))]).view(-1,args.classNum*len(step))
    AEs = torch.tensor([[robustness_AEs[i][0][args.classNum*j:args.classNum*(j+1)].tolist() for j in step] for i in range(len(robustness_AEs))]).view(-1,args.classNum*len(step))
    
    label_clean = [1 for _ in range(len(clean))] 
    label_AEs = [0 for _ in range(len(AEs))]
    label_robustness = label_clean + label_AEs
    
    pred_clean = clf.decision_function(clean)
    pred_AEs =  clf.decision_function(AEs)
    
    pred_clean_AEs = pred_clean.tolist()
    pred_clean_AEs.extend(pred_AEs.tolist()) 
    roc = roc_curve(label_robustness, pred_clean_AEs) # roc
    score = roc_auc_score(label_robustness, pred_clean_AEs) # auroc
    
    return score


def train_equance(model, dataset, step):
    
    np.random.seed(seed=2023)
    torch.manual_seed(2023)
    
    ##  load robustness data
    data_path = 'robustness/' + model + '_' + dataset +'_randomN' + str(args.noise_N) + '_step' + str(args.noise_step)
    
    
    [robustness_train, robustness_FGSM_clean, robustness_PGD_clean, robustness_BIM_clean,
     robustness_DeepFool_clean, robustness_CW_clean, robustness_FGSM, robustness_PGD, robustness_BIM, 
     robustness_DeepFool, robustness_CW] = torch.load(data_path)

    train_data = robustness_train
    AEs_list = [robustness_BIM_clean, robustness_CW_clean, robustness_DeepFool_clean, robustness_FGSM_clean, 
            robustness_PGD_clean, robustness_BIM, robustness_CW, robustness_DeepFool, robustness_FGSM, 
            robustness_PGD]
    
    # OCSVM
    clf = train_OCSVM(train_data, step)
    
    AUROC_list =[]
    for i in range(5):
        score = calculate_AUROC(clf, AEs_list[i], AEs_list[i+5], step)
        if score < 0.5:
            score = calculate_AUROC(clf, AEs_list[i+5], AEs_list[i], step)
        AUROC_list.append(score)
        
    return AUROC_list
        
def main():
    AUROC = []
    print("start:",args.model, args.dataset)
    args.classNum = 100 if args.dataset == 'cifar100' else 10
    AUROC_score = train_equance(args.model, args.dataset, args.use_noise_step)
    AUROC.append(AUROC_score)
    torch.save(AUROC, 'result/USV_OCSVM_'+ args.model + "_" + args.dataset + "_randomN" + str(args.noise_N) + "_step" + str(args.noise_step) + '.pt')
            
if __name__ == '__main__':
    main()

