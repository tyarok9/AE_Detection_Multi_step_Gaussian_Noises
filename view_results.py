from decimal import *
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch code: Multi-step Gaussian Noise Detector')
parser.add_argument('--model_list', type=list, default=['densenet','resnet'], help='models list of showing result')
parser.add_argument('--data_list', type=list, default=['cifar10','cifar100','svhn'], help='datasets list')
parser.add_argument('--noise_step', type=int, default=10, help='the number of noise magnitude steps')
parser.add_argument('--noise_N', type=int, default=30, help='the number of noise images generated from each step')
parser.add_argument('--train_method', type=str, default= 'both', help='supervised | unsupervised | both')
args = parser.parse_args()

def data_print(l): 
    for d in l:
        d = float(d)*100
        d = Decimal(str(d))
        d = d.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
        print(d,  end=' '*(6-len(str(d))) + '| ')

def print_sv():
    print('Supervised')
    print(' '*19, end = '| ')
    for ae in ['BIM','CW','DF','FGSM','PGD', 'AVG']:
        print(ae, end =' '*(6-len(str(ae))) + '| ' )
    print()
        
    for i in range(len(args.model_list)):
        for j in range(len(args.data_list)):
            data = torch.load('result/SV_LSTM_'+ args.model_list[i] + '_' + args.data_list[j] + '_randomN' + str(args.noise_N) + '_step' + str(args.noise_step) + '.pt')
            
            if j==0:
                print(args.model_list[i], end = ' '*(9-len(args.model_list[i]))+'| ')
            else:
                print(' '*8, end = ' | ')
            
            print(args.data_list[j], end = ' '*(8-len(args.data_list[j]))+ '| ')
            
            cv_mean = torch.tensor(data[1]).mean(dim=0) 
            ae_mean = cv_mean.mean()
            data_print(cv_mean)
            data_print([ae_mean])
    
            print()
    print()

def print_usv():
    print('Unsupervised')
    print(' '*19, end = '| ')
    for ae in ['BIM','CW','DF','FGSM','PGD', 'AVG']:
        print(ae, end =' '*(6-len(str(ae))) + '| ' )
    print()
    
    #USVdata = torch.load('result/USV_OCSVM' + '_randomN' + str(args.noise_N) + '_step' + str(args.noise_step) + '.pt'')
    
    for i in range(len(args.model_list)):
        for j in range(len(args.data_list)):
            data = torch.load('result/USV_OCSVM_' + args.model_list[i] + '_' + args.data_list[j] + '_randomN' + str(args.noise_N) + '_step' + str(args.noise_step) + '.pt')
            if j==0:
                print(args.model_list[i], end = ' '*(9-len(args.model_list[i]))+'| ')
            else:
                print(' '*8, end = ' | ') 
                
            print(args.data_list[j], end = ' '*(8-len(args.data_list[j]))+ '| ')
            #print(data[0])
            data_print(data[0])
            data_print([torch.tensor(data).mean()])
            
            print()

def main():
    if args.train_method == 'supervised':
        print_sv()
    elif args.train_method == 'unsupervised':
        print_usv()
    else:
        print_sv()
        print_usv()

if __name__ == '__main__':
    main()
