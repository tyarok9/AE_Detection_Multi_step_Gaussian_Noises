import argparse 
import random
import numpy as np
import seaborn as sns
import os

import models
import models.densenet
import models.resnet

import torch
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description='PyTorch code: Multi-step Gaussian Noise Detector')
parser.add_argument('--batch_size', type=int, default=10000, metavar='N', help='batch size for data loader')
parser.add_argument('--model', required=True, help='resnet | densenet')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--device', type=str, default="cuda:0", help='devie')
parser.add_argument('--multigpu', type=bool, default=False, help='use data parallel or not')
parser.add_argument('--noise_step', type=int, default=10, help='the number of noise magnitude steps')
parser.add_argument('--noise_N', type=int, default=30, help='the number of noise images generated from each step')
args = parser.parse_args()
#args = parser.parse_args(args=['--dataset','cifar10','--model','resnet'])
print(args)

def load_AEs(model, dataset, ae_dir):
    def align_dataset(AEs, clean, label, noisy):
        return [[AEs[i], clean[i], label[i], noisy[i]] for i in range(len(AEs))] 
    # clean
    clean_data = torch.load("AEs/"+ ae_dir + "/clean_data_"+ ae_dir + "_clean.pth")
    label = torch.load("AEs/" + ae_dir + "/label_"+ ae_dir + "_clean.pth")
    noisy_data = torch.load("AEs/"+ ae_dir + "/noisy_data_"+ ae_dir + "_clean.pth")
    clean = align_dataset(clean_data, clean_data, label, noisy_data)

    #BIM
    AEs = torch.load("AEs/"+ ae_dir + "/adv_data_"+ ae_dir + "_BIM.pth")
    clean_data = torch.load("AEs/"+ ae_dir + "/clean_data_"+ ae_dir + "_BIM.pth")
    label = torch.load("AEs/" + ae_dir + "/label_"+ ae_dir + "_BIM.pth")
    noisy_data = torch.load("AEs/"+ ae_dir + "/noisy_data_"+ ae_dir + "_BIM.pth")
    BIM = align_dataset(AEs, clean_data, label, noisy_data)

    #CW
    AEs = torch.load("AEs/"+ ae_dir + "/adv_data_"+ ae_dir + "_CWL2.pth")
    clean_data = torch.load("AEs/"+ ae_dir + "/clean_data_"+ ae_dir + "_CWL2.pth")
    label = torch.load("AEs/" + ae_dir + "/label_"+ ae_dir + "_CWL2.pth")
    noisy_data = torch.load("AEs/"+ ae_dir + "/noisy_data_"+ ae_dir + "_CWL2.pth")
    CW = align_dataset(AEs, clean_data, label, noisy_data)

    #DeepFool
    AEs = torch.load("AEs/"+ ae_dir + "/adv_data_"+ ae_dir + "_DeepFool.pth")
    clean_data = torch.load("AEs/"+ ae_dir + "/clean_data_"+ ae_dir + "_DeepFool.pth")
    label = torch.load("AEs/" + ae_dir + "/label_"+ ae_dir + "_DeepFool.pth")
    noisy_data = torch.load("AEs/"+ ae_dir + "/noisy_data_"+ ae_dir + "_DeepFool.pth")
    DeepFool = align_dataset(AEs, clean_data, label, noisy_data)
    
    #FGSM
    AEs = torch.load("AEs/"+ ae_dir + "/adv_data_"+ ae_dir + "_FGSM.pth")
    clean_data = torch.load("AEs/"+ ae_dir + "/clean_data_"+ ae_dir + "_FGSM.pth")
    label = torch.load("AEs/" + ae_dir + "/label_"+ ae_dir + "_FGSM.pth")
    noisy_data = torch.load("AEs/"+ ae_dir + "/noisy_data_"+ ae_dir + "_FGSM.pth")
    FGSM = align_dataset(AEs, clean_data, label, noisy_data)

    #PGD
    AEs = torch.load("AEs/"+ ae_dir + "/adv_data_"+ ae_dir + "_PGD100.pth")
    clean_data = torch.load("AEs/"+ ae_dir + "/clean_data_"+ ae_dir + "_PGD100.pth")
    label = torch.load("AEs/" + ae_dir + "/label_"+ ae_dir + "_PGD100.pth")
    noisy_data = torch.load("AEs/"+ ae_dir + "/noisy_data_"+ ae_dir + "_PGD100.pth")
    PGD = align_dataset(AEs, clean_data, label, noisy_data)
    
    return clean, BIM, CW, DeepFool, FGSM, PGD


def get_train(model, dataset):
    if model == "densenet":
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)), ])
        args.clamp_min = -1.98888885975
        args.clamp_max = 2.12560367584
    else: #resnet
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        args.clamp_min = -2.42906570435
        args.clamp_max = 2.75373125076
        
        
    if dataset == "cifar10":
        temp = datasets.CIFAR10('./data', train=True, download=True, transform=in_transform)
    elif dataset == "cifar100":
        temp = datasets.CIFAR100('./data', train=True, download=True, transform=in_transform)
    elif dataset == "svhn":
        temp = datasets.SVHN('./data', split='train', download=True, transform=in_transform)
    else:
        raise Exception('Set correct dataset name')

    train_clean = [[temp[i][0], temp[i][0], torch.tensor(temp[i][1]), 0] for i in range(len(temp))]
    return train_clean

#define model
def get_model(ae_dir, device):
    pre_trained_net = './pre_trained/' + ae_dir + '.pth'
    ClassNum = 100 if args.dataset == "cifar100" else 10
    
    if args.model == "densenet":
        model = models.densenet.DenseNet3(100, int(ClassNum))
        model.load_state_dict(torch.load(pre_trained_net, map_location=device))
    else: # resnet
        model = models.resnet.ResNet34(int(ClassNum))
        model.load_state_dict(torch.load(pre_trained_net, map_location=device))
    return model


def get_robustness( model, device, dataset, noise_SD, AEs_flag):
    def add_gaussian_noise(noise_images, AEs_flag, AEs, clean_data, noise_image, model):
        Zn_list = [] 
        for i in range(len(noise_images)):
            N = args.noise_N
            Zn = 0 
            
            for _ in range(N):
                if AEs_flag:
                    noise_data = AEs + noise_images[i][_]
                else:
                    noise_data = clean_data + noise_images[i][_]
                noise_data = torch.clamp(noise_data, args.clamp_min, args.clamp_max)
    
                Zn += model(noise_data)   
            Zn = Zn / N 
            noise_pred = torch.argmax(Zn, dim=-1).cpu().numpy() 
            Zn_list.append(Zn.cpu())
     
        return Zn_list, noise_pred
    
    def concat_opt(Zc, Zn_list): # concat
        try:
            o1 = Zc.tolist()
        except:
            pass
        o2 = Zn_list
        
        for i in range(len(o2)):
            o1 = o1 + o2[i].tolist()  
        return torch.tensor(np.array(o1)).float()

    # create gauusian noise image
    noise_image = []
    np.random.seed(seed=2023) # 2022 in oroginal paper
    for i in range(len(noise_SD)):
        img = [torch.from_numpy( np.random.normal(0,noise_SD[i],np.shape(dataset[0][0])).astype(np.float32)).clone().to(device) for _ in range(args.noise_N)]
        noise_image.append(img)

    torch.manual_seed(2023)  
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False) 
    robustness_list = []
    robustness = []
        
    with torch.no_grad(): 
        for AEs, clean_data, labels, noisy in loader: 
            AEs, labels, clean_data = AEs.to(device), labels.to(device), clean_data.to(device)
            
            if AEs_flag:
                Zc = model(AEs)
                output_clean = model(clean_data)
                init_pred_Zc = torch.argmax(Zc, dim=-1).cpu().numpy()  
                init_pred_clean = torch.argmax(output_clean, dim=-1).cpu().numpy() 
                
            else:
                Zc = model(clean_data)
                init_pred_Zc = torch.argmax(Zc, dim=-1).cpu().numpy()  
                init_pred_clean = torch.argmax(Zc, dim=-1).cpu().numpy() 

            # create detection features
            Zn_list, noise_pred = add_gaussian_noise(noise_image, AEs_flag, AEs, clean_data, noise_image, model)

            
            # align data
            for i in range(len(Zc)):
                # skip
                if AEs_flag: 
                    if init_pred_clean[i] != labels[i] or init_pred_Zc[i] == labels[i]:  
                        continue
                else:  # not AEs
                    if init_pred_clean[i] != labels[i] : 
                        continue
                
                robustness_all_noise = concat_opt(Zc[i], [Zn_list[j][i] for j in range(len(Zn_list))]) 
                robustness.append( [ robustness_all_noise.cpu(), labels[i].item(), init_pred_Zc[i].item() ] ) 

    return robustness


def main():
    os.makedirs("robustness", exist_ok=True)
    os.makedirs("result", exist_ok=True)
    print("start:",  args.model, args.dataset)
    ae_dir = args.model + "_" + args.dataset 
    clean, BIM, CW, DeepFool, FGSM, PGD = load_AEs(args.model, args.dataset, ae_dir)
    train_clean = get_train(args.model, args.dataset)

    device = args.device
    model = get_model(ae_dir, device)
    # set device
    if args.multigpu:
        print("Start DataParallel")
        model = torch.nn.DataParallel(model) 
        torch.backends.cudnn.benchmark = True
    model.to(device)
    model.eval()
    
    noisesd = [0.1*(i+1) for i in range(args.noise_step)] # Variations in the magnitude of noise

    robustness_BIM_clean = get_robustness( model, device, BIM, noisesd, AEs_flag = False )
    robustness_BIM = get_robustness( model, device, BIM, noisesd, AEs_flag = True )
    robustness_CW_clean = get_robustness( model, device, CW, noisesd, AEs_flag = False )
    robustness_CW = get_robustness( model, device, CW, noisesd, AEs_flag = True )
    robustness_DeepFool_clean = get_robustness( model, device, DeepFool, noisesd, AEs_flag = False )
    robustness_DeepFool = get_robustness( model, device, DeepFool, noisesd, AEs_flag = True )
    robustness_FGSM = get_robustness( model, device, FGSM, noisesd, AEs_flag = True )
    robustness_FGSM_clean = get_robustness( model, device, FGSM, noisesd, AEs_flag = False )
    robustness_PGD_clean = get_robustness( model, device, PGD, noisesd, AEs_flag = False )
    robustness_PGD = get_robustness( model, device, PGD, noisesd, AEs_flag = True )
    robustness_train = get_robustness( model, device, train_clean, noisesd, AEs_flag = False )

    # save
    os.makedirs(robustness, )
    dir_name = "robustness/" + args.model +"_"+ args.dataset + "_randomN" + str(args.noise_N) + "_step" + str(args.noise_step)
    torch.save([robustness_train, robustness_FGSM_clean, robustness_PGD_clean, robustness_BIM_clean,
                robustness_DeepFool_clean, robustness_CW_clean, robustness_FGSM, robustness_PGD, robustness_BIM, 
                robustness_DeepFool, robustness_CW], dir_name)
    
    print("finish:",  args.model, args.dataset)


if __name__ == '__main__':
    main()

