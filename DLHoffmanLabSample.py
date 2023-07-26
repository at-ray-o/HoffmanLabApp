#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:12:24 2021

@author: atreyadey
"""

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import Process, freeze_support


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) #/ torch.sum(mask)
        return result

class contactsDataset(Dataset):
    def __init__(self, X):
        'Initialization'
        self.X = X

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X

    transform = T.Compose([
        T.ToTensor()])

class coordinatesDataset(Dataset):
    def __init__(self, X):
        'Initialization'
        self.X = X

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        currentCoord = self.X[index]
        X = self.transform(currentCoord)
        return X

    transform = T.Compose([
        T.ToTensor()])

class croppedDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ims):
        'Initialization'
        self.ims = ims
    def __len__(self):
            'Denotes the total number of samples'
            return len(self.ims)
    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            image = self.ims[index]
            X = self.transform(image)
            return X
    transform = T.Compose([
        #T.ToPILImage(),
        #T.CenterCrop(0.75 * 64),
        #T.Resize(image_size),
        #T.RandomResizedCrop(image_size),
        #T.RandomHorizontalFlip(),
        T.ToTensor()])

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_rep = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_rep2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_rep = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_rep2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_rep = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_rep2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_rep = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_rep2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5_rep = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv5_rep2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6_rep = nn.Conv2d(512, 512, 3, padding=1)
        self.conv6_rep2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv7_rep = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7_rep2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8_rep = nn.Conv2d(128, 128, 3, padding=1)
        self.conv8_rep2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_rep = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_rep2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 1, 3, padding=1)

        self.bnorm64 = nn.BatchNorm2d(64)
        self.bnorm64u = nn.BatchNorm2d(64)
        self.bnorm128 = nn.BatchNorm2d(128)
        self.bnorm128u = nn.BatchNorm2d(128)
        self.bnorm256 = nn.BatchNorm2d(256)
        self.bnorm256u = nn.BatchNorm2d(256)
        self.bnorm512 = nn.BatchNorm2d(512)
        self.bnorm512u = nn.BatchNorm2d(512)
        self.bnorm1024 = nn.BatchNorm2d(1024)
        self.bnorm1024u = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.2)

        self.expelu = nn.ELU()
        #Decoder
        self.t_conv5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(64, 1, 2, stride=2)
        self.zeroPad = nn.ConstantPad2d((1, 0, 1, 0),0)
        self.softplus = nn.Softplus()
    def forward(self, x):
        #x = nn.ZeroPad2d(4)                        #Pad with 4 zeros on all sides
        x = torch.sigmoid(self.conv1(x))            #1,(256,256) channel -> 64,(256,256) channel
        x = torch.sigmoid(self.conv1_rep(x))        #64,(256,256) channel -> 64,(256,256) channel
        x = torch.sigmoid(self.conv1_rep2(x))        #64,(256,256) channel -> 64,(256,256) channel
        x = self.bnorm64(x)                         #batch normalization 64
        x = F.mish(self.drop(x))                    #dropout
        a = F.mish(x)                               #save the data
        x = self.pool(x)                            #Shrink by pool 2
        x = F.mish(self.conv2(x))                   #64,(128,128) channel -> 128,(128,128) channel
        x = F.mish(self.conv2_rep(x))               #128,(128,128) channel -> 128,(128,128) channel
        x = F.mish(self.conv2_rep2(x))               #128,(128,128) channel -> 128,(128,128) channel
        b = F.mish(x)                               #save the data
        x = self.bnorm128(x)                        #batch normalization 128
        x = F.mish(self.drop(x))                    #dropout
        x = self.pool(x)                            #Shrink by pool by 2
        x = F.mish(self.conv3(x))                   #128,(64,64) channel -> 256,(64,64) channel
        x = F.mish(self.conv3_rep(x))               #256,(64,64) channel -> 256,(64,64) channel
        x = F.mish(self.conv3_rep2(x))               #256,(64,64) channel -> 256,(64,64) channel
        c = F.mish(x)                               #save the data
        x = self.bnorm256(x)                        #batch normalization 256
        x = F.mish(self.drop(x))                    #dropout
        x = self.pool(x)                            #Shrink by pool by 2
        x = F.mish(self.conv4(x))                   #256,(32,32) channel -> 512,(32,32) channel
        x = F.mish(self.conv4_rep(x))               #512,(32,32) channel -> 512,(32,32) channel
        x = F.mish(self.conv4_rep2(x))               #512,(32,32) channel -> 512,(32,32) channel
        x = self.bnorm512(x)                        #batch normalization 512
        x = F.mish(self.drop(x))                    #dropout
        d = F.mish(x)                               #save the data
        x = self.pool(x)                            #Shrink by pool by 2
        x = F.mish(self.conv5(x))                   #512,(16,16) channel -> 1024,(16,16) channel
        x = F.mish(self.conv5_rep(x))               #1024,(16,16) channel -> 1024,(16,16) channel
        x = F.mish(self.conv5_rep2(x))               #1024,(16,16) channel -> 1024,(16,16) channel
        x = self.bnorm1024(x)                       #batch normalization 1024
        x = F.mish(self.drop(x))                    #dropout
        x = F.mish(self.t_conv5(x))                 #Up 1024,(16,16) channel -> 512,(32,32) channel
        x = self.bnorm512u(x)                       #batch normalization bnorm512
        x = torch.cat((x,d),1)                      #concatenate 512,(32,32) channel -> 1024,(32,32) channel
        x = F.mish(self.conv6(x))                  #1024,(32,32) channel -> 512,(32,32) channel
        x = F.mish(self.conv6_rep(x))              #512,(32,32) channel -> 512,(32,32) channel
        x = F.mish(self.conv6_rep2(x))              #512,(32,32) channel -> 512,(32,32) channel
        x = F.mish(self.t_conv4(x))                 #Up 512,(32,32) channel -> 256,(64,64) channel
        x = self.bnorm256u(x)                       #batch normalization bnorm256
        x = torch.cat((x,c),1)                      #concatenate 256,(64,64) channel -> 512,(64,64) channel
        x = F.mish(self.conv7(x))                   #512,(64,64) channel -> 256,(64,64) channel
        x = F.mish(self.conv7_rep(x))               #256,(64,64) channel -> 256,(64,64) channel
        x = F.mish(self.conv7_rep2(x))               #256,(64,64) channel -> 256,(64,64) channel
        x = F.mish(self.t_conv3(x))                 #Up 256,(64,64) channel -> 128,(128,128) channel
        x = self.bnorm128u(x)                       #batch normalization 128
        x = F.mish(self.drop(x))                    #dropout
        x = torch.cat((x,b),1)                      #concatenate 128,(128,128) channel -> 256,(128,128) channel
        x = F.mish(self.conv8(x))                   #256,(128,128) channel -> 128,(128,128) channel
        x = F.mish(self.conv8_rep(x))               #128,(128,128) channel -> 128,(128,128) channel
        x = F.mish(self.conv8_rep2(x))               #128,(128,128) channel -> 128,(128,128) channel
        x = torch.sigmoid(self.t_conv2(x))          #Up 128,(128,128) channel -> 64,(256,256) channel
        x = self.bnorm64u(x)                        #batch normalization 64
        x = torch.cat((x,a),1)                      #concatenate 64,(256,256) channel -> 128,(256,256) channel
        x = F.mish(self.drop(x))                    #dropout
        x = F.mish(self.conv9(x))                   #128,(256,256) channel -> 64,(256,256) channel
        x = F.mish(self.conv9_rep(x))                   #128,(256,256) channel -> 64,(256,256) channel
        x = F.mish(self.conv9_rep2(x))                   #128,(256,256) channel -> 64,(256,256) channel
        x = F.mish(self.conv10(x))                   #64,(256,256) channel -> 1,(256,256) channel

        
        L_rank, L_eucl = 0,0    #Optional training, deleted for this code
        dist = x
        contacts = ((torch.tanh(-dist+4)+1)*100+0.5)  #log(9)=2.1972245773362196
        overlap = ((torch.tanh(-dist+0.81)+1)*100+0.5)
        overlap = torch.sum(torch.flatten(overlap))
        return L_rank, L_eucl, contacts, dist, overlap

def show_images(images, epoch, nmax=9):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=3).permute(1, 2, 0))

def show_batch(dl, nmax=9):
    for images in dl:
        #print(np.shape(images))
        show_images(images, nmax)
        break

def print_batch(dl, epoch, nmax=3):
    #show_batch(contacts_dl)
    save_image(dl.detach()[:nmax], './Vaporwave_Out_Images/Autoencoder_image{}.png'.format(epoch))

def make_dir():
    image_dir = 'Vaporwave_Out_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def save_decod_img(img, epoch):
    #print(np.shape(img))
    img = img.view(img.size(0), 1, 64, 64)
    #print(np.shape(img))
    #input()
    save_image(img[0], './Vaporwave_Out_Images/Autoencoder_image{}.png'.format(epoch))

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def training(model, contacts_dl, f_train, f_test, Epochs, weight_euc, weight_avg, model_dir_save):
    train_loss = []
    test_loss = []
    for epoch in range(Epochs):
        if epoch%10==0:
            torch.save(model.state_dict(),model_dir_save)
        running_loss = 0.0
        running_loss_dist = 0.0
        for data in (contacts_dl):
            contacts = data[:,0:1,:,:]
            dist = data[:,1:2,:,:]
            avgdist = data[:,2:3,:,:]
            device = get_device()
            contacts = contacts.to(device)
            dist = dist.to(device)
            avgdist = avgdist.to(device)
            optimizer.zero_grad()
            L_rank, L_eucl, outputs, dist_outputs, overlap = model(contacts)
            #loss_dist_avg = criterion(1/dist_outputs[:,:,2:254,2:254], 1/dist[:,:,2:254,2:254])
            # print(dist_outputs.size())
            # input()
            dist_loss = criterion(dist_outputs[:,:,2:254,2:254], dist[:,:,2:254,2:254]) #compute distance prediction loss
            loss = (dist_loss)  #+ weight_euc/10.0*L_eucl + weight_euc/10.0*L_rank + weight_avg/10.0*overlap
            #print(weight_euc/10.0,weight_avg/10.0)
            #input()

            loss.backward() #Use autoencoder loss for the network
            #torch.nn.utils.clip_grad_norm_(model.parameters(),0.01)
            #print(model.conv8.weight.grad)
            optimizer.step()
            running_loss += loss.item()
            running_loss_dist += dist_loss.item()

        loss = running_loss / len(contacts_dl)
        loss_dist = running_loss_dist / len(contacts_dl)
        train_loss.append(loss)
        test_loss.append(loss_dist)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss),file=f_train,flush=True)
        print('Epoch {} of {}, Test Loss: {:.3f}'.format(
            epoch+1, Epochs, loss_dist),file=f_test,flush=True)

        if epoch % 2 == 0:
            dummy = outputs.cpu().data
            plt.imsave("Autoencoder_contact"+str(epoch)+".png",(dummy[9,0,:,:]))
            dummy = dist_outputs.cpu().data
            plt.imsave("Autoencoder_distance"+str(epoch)+".png",(dummy[9,0,:,:]))
            np.save("Distance_"+str(epoch)+".npy",(dummy[9,0,:,:]))
            #save_decod_img(outputs.cpu().data, epoch)

    return train_loss

def run(n_epochs, weight_euc, weight_avg, model_dir_save):
    torch.multiprocessing.freeze_support()
    #torch.autograd.set_detect_anomaly(True)
    print('Training Started')
    device = get_device()
    model.to(device)
    make_dir()
    f_train = open("Lossfile_train.txt", "w")
    f_test = open("Lossfile_test.txt", "w")
    train_loss = training(model, contacts_dl, f_train, f_test, n_epochs, weight_euc, weight_avg, model_dir_save)

def evaluation(model, batch_size, contacts_dl,  weight_euc, weight_avg):
    train_loss = []
    test_loss = []

    running_loss = 0.0
    running_loss_dist = 0.0
    iter = 0
    for data in (contacts_dl):
        contacts = data[:,0:1,:,:]
        dist = data[:,1:2,:,:]
        avgdist = data[:,2:3,:,:]
        L_rank, L_eucl, outputs, dist_outputs, rgLoss = model(dist)
        loss = criterion(contacts, outputs)      #compute autoencoder loss
        loss_dist_avg = criterion(avgdist,dist_outputs)
        dist_loss = criterion(dist_outputs[:,:,3:253,3:253], dist[:,:,3:253,3:253]) #compute distance prediction loss
        loss = dist_loss
        running_loss += loss.item()
        running_loss_dist += dist_loss.item()
        dummy = dist_outputs.cpu().data
        np.save("Eval_Distance_"+str(iter)+".npy",dummy)
        dummy = dist.cpu().data
        np.save("Eval_Distance_Real"+str(iter)+".npy",dummy)
        iter += 1

    loss = running_loss / len(contacts_dl)
    loss_dist = running_loss_dist / len(contacts_dl)
    train_loss.append(loss)
    test_loss.append(loss_dist)
    print('Train Loss: {:.3f}'.format(loss))
    print('Test Loss: {:.3f}'.format(loss_dist))



if __name__ == '__main__':
    """
    Paramaters:
    Is train?
    Number of epochs
    Learning rate
    Weight decay
    Optional Euclidean Weight
    Batch size
    Training structures range start
    Training structures range end
    Directory Name load
    Directory Name save
    Structure directory
    """
    train = True
    n_epochs = 30
    l_rate = 0.001
    w_decay=0.001
    weight_euc = 0.0
    weight_avg = 0.0
    batch_size = 10
    trainStartStruct = 0
    trainEndStruct = 1000
    model_dir_load = "SavedStateDictionary/2p00Supervised.pth"
    model_dir_save = "SavedStateDictionary/2p00Supervised.pth"
    dir = "D:/Work/ContactMapToStructure/LJSimulations/Trajectories/ThetaSolventLD250mers/MdCodeCpp2p00/out/"
    CONTACTS_DIR = dir+"DistancesSP.npy"
    print("Loop from top")
    X_train = np.load(CONTACTS_DIR)
    print("Shape of training data: ",X_train.shape)
    print("Data type: ",type(X_train))
    X_train = X_train[trainStartStruct:trainEndStruct,0:256,0:256,:]
    rg = np.sum(X_train[:,0:256,0:256,1])/(2*256*256*1000)
    print("rg =",(rg))
    X_train[:,0:256,0:256,0] = np.power(X_train[:,0:256,0:256,0],1.9)
    dist_avg = np.mean(X_train[:,0:256,0:256,1],axis=0)
    dist_avg_arr = np.repeat(dist_avg[np.newaxis,:, :], np.shape(X_train)[0], axis=0)
    dist_avg_arr = np.expand_dims(dist_avg_arr, axis=3)
    X_train = np.concatenate((X_train, dist_avg_arr), axis=3)

    print("Shape of training data: ",X_train.shape)
    print("Data type: ",type(X_train))
    plt.imsave("Real_distance_Example.png",(X_train[9,:,:,1]))
    plt.imsave("Real_contact_Example.png",(X_train[9,:,:,0]))
    plt.imsave("Avg_distance_Example.png",(dist_avg))


    cropped_dataset = croppedDataset(X_train)
    contacts_dl = DataLoader(cropped_dataset, batch_size, shuffle=False, num_workers=3, pin_memory=True)

    model = Autoencoder()
    
    n_breaks = 10
    if train==True:
        #model.load_state_dict(torch.load(model_dir_load))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=l_rate, weight_decay=w_decay)
        run(n_epochs, weight_euc, weight_avg, model_dir_save)
        print("Saved")
        torch.save(model.state_dict(),model_dir_save)
    else:
        #Perform calculations
        criterion = nn.MSELoss()
        model.load_state_dict(torch.load(model_dir_load))
        evaluation(model, batch_size, contacts_dl, weight_euc, weight_avg)
