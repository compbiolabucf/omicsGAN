import torch
from torch import nn
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from torch import linalg as LA
import torch.utils.data
from torch.utils.data.dataset import Dataset
import copy
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import random
from math import floor
import torch.nn.functional as F
from torch.nn import init
from sklearn.metrics import roc_auc_score
from functools import reduce
from sklearn.metrics import accuracy_score
np.seterr(divide='ignore', invalid='ignore')
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn import svm
import sys
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import argparse

torch.manual_seed(111)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def feature_selection(X, y):
  data_label1 = np.asarray([X[i] for i in range(len(y)) if y[i] == 1])
  data_label0 = np.asarray([X[i] for i in range(len(y)) if y[i] == 0])
  p = ttest_ind(data_label1, data_label0)[1]
  keep_ttest_index = np.argsort(p)[0:200] #np.where(p < .001)[0]
  return keep_ttest_index

def load_data(path):
  data = pd.read_csv(path,delimiter=',',index_col=0)
  cols = data.columns.tolist()
  data = np.log1p(data)
  data.loc[:, 'var'] = data.loc[:, cols].var(axis=1)
  drop_index = data[data['var'] < 0].index.tolist()
  data.drop(index=drop_index, inplace=True)
  X = data[cols]

  return X

def SVM(X_train, y_train, X_test, y_test):

  clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
  y_pred = clf.predict_proba(X_test)[:,1]
  
  y_pred_bin = np.copy(y_pred)
  y_pred_bin[y_pred_bin < 0.5] = 0
  y_pred_bin[y_pred_bin >= 0.5] = 1

  return roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred_bin),f1_score(y_test, y_pred_bin)


def prediction(mRNA_value,GAN_epoch,labels,update=None):
  X = np.array(mRNA_value).astype(float)

  trial=50
  AUC_all=[]
  ACC_all=[]

  features=[]
  for col in range(labels.shape[1]):
    y = labels[:,col]
    AUC = []
    ACC = []
    for i in range(trial):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
      target = feature_selection(X_train, y_train)
      X_train = X_train[:, target]
      X_test = X_test[:, target] 
      auc, acc ,f1 = SVM(X_train, y_train, X_test, y_test)
      AUC.append(auc)
      ACC.append(f1)
    AUC_all.append(AUC)
    ACC_all.append(np.mean(ACC)) 

  AUC_all = np.array(AUC_all).transpose()
  if update==1:
    print('AUC for real omics2:', np.mean(AUC_all,axis=0))
  '''
  if GAN_epoch==0:
    print('AUC for real data:', np.mean(AUC_all,axis=0))
  else:
    print('AUC for generated data at epoch',GAN_epoch,':', np.mean(AUC_all,axis=0))
  '''
  return np.mean(AUC_all,axis=0)


class Discriminator(nn.Module):
    def __init__(self,n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            #nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Dropout(0.3),
    
            nn.Linear(128, 1),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self,n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            
            #nn.Linear(768, 1024),
            #n.BatchNorm1d(1024),
            #nn.ReLU(),

            #nn.Linear(1024, 768),
            #nn.BatchNorm1d(768),
            #nn.ReLU(),
            
            #nn.Linear(768, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),
            
            nn.Linear(768, n_input),
        )

    def forward(self, x):
        output = self.model(x)
        
        return output


def omics2(update,omics1,omics2,adj_file,label):
    print('Generating miRNA update '+str(update))


    mRNA = load_data('mRNA.csv')
    miRNA = pd.read_csv('miRNA.csv',index_col=0,delimiter=',')
    adj = pd.read_csv('bipartite_targetscan_gene.csv',index_col=0)
    xy, x_ind, y_ind = np.intersect1d(mRNA.columns,miRNA.columns,return_indices=True)
    _, x_ind1, y_ind1 = np.intersect1d(miRNA.index,adj.columns,return_indices=True)
    xy1, x_ind2, y_ind2 = np.intersect1d(mRNA.index.values,adj.index.values,return_indices=True)

    mRNA = mRNA.iloc[:,x_ind]
    miRNA = miRNA.iloc[:,y_ind]
    mRNA = mRNA.iloc[x_ind2,:]
    miRNA = miRNA.iloc[x_ind1,:]
    adj = adj.iloc[:,y_ind1]
    adj = adj.iloc[y_ind2,:]
    mRNA = mRNA.fillna(0)
    miRNA = miRNA.fillna(0)

    adj[adj==1] = -1
    adj[adj==0] = 1

    data = pd.read_csv(label, delimiter=',',index_col=0)
    xy, x_ind, y_ind = np.intersect1d(mRNA.columns,data.index,return_indices=True)
    mRNA = mRNA.iloc[:,x_ind]
    miRNA = miRNA.iloc[:,x_ind]
    y= data.iloc[y_ind,:].astype(str)
    #y[y=='Positive']=1
    #y[y=='Negative']=0
    labels=np.array(y).astype(np.float32)

    sample_name = miRNA.columns
    feature_name = miRNA.index

    mRNA = np.array(mRNA).transpose().astype(np.float32)
    miRNA = np.array(miRNA).transpose().astype(np.float32)
    adj = np.array(adj).astype(np.float32)

    X_0 = torch.from_numpy(miRNA).to(device)

    if update>1:
        mRNA_file_name = 'omics1_'+str(update-1)+'.csv'
        miRNA_file_name = 'omics2_'+str(update-1)+'.csv'

        miRNA = pd.read_csv(miRNA_file_name,delimiter=',',index_col=0)
        miRNA = np.array(miRNA).astype(np.float32)
        mRNA = pd.read_csv(mRNA_file_name,delimiter=',',index_col=0)
        mRNA = np.array(mRNA).astype(np.float32)

    n_input_miRNA=np.size(miRNA,1)
    sample_size = np.size(miRNA,0)

    C=np.sqrt(np.outer(np.sum(np.absolute(adj),0),np.sum(np.absolute(adj),1)))
    adj=np.divide(adj,C.transpose())

    miRNA_train_data=torch.from_numpy(miRNA)
    adj=torch.from_numpy(adj)

    miRNA_train_labels = torch.zeros(sample_size)

    miRNA_train_set = [(miRNA_train_data[i], miRNA_train_labels[i]) for i in range(sample_size)]

    batch_size=sample_size
    miRNA_train_loader = torch.utils.data.DataLoader(miRNA_train_set, 
                                  batch_size=batch_size, shuffle=False)
           
    discriminator = Discriminator(n_input_miRNA).to(device)
    generator = Generator(n_input_miRNA).to(device)

    lr_D = 5e-6
    lr_G = 5e-5
    num_epochs = 10000
    critic_ite = 5
    weight_clip = 0.01

    optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
    optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=lr_G)

    start = timer() 
    best=None
    final_real_data = []
    final_gen_data = []
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(miRNA_train_loader):

            mRNA_train_data = mRNA[n*batch_size:(n+1)*batch_size,:]
            mRNA_train_data=torch.from_numpy(mRNA_train_data)
            latent_value = torch.matmul(mRNA_train_data,adj)
            real_samples = real_samples.to(device)
            latent_value = latent_value.to(device)
  
            #### training the discriminator
            for n_critic in range(critic_ite):

                generated_samples = generator(latent_value)

                discriminator.zero_grad()
                output_discriminator_real = discriminator(real_samples)
                output_discriminator_fake = discriminator(generated_samples)

                loss_discriminator = torch.mean(output_discriminator_fake)-torch.mean(output_discriminator_real)
                loss_discriminator.backward(retain_graph=True)
                optimizer_discriminator.step()

                for p in discriminator.parameters():
                    p.data.clamp_(-weight_clip,weight_clip)
            
            #### Training the generator
            generator.zero_grad()
            
            output_discriminator_fake = discriminator(generated_samples)
            loss_generator = -torch.mean(output_discriminator_fake) + .001* LA.norm((X_0 - generated_samples), 2)

            loss_generator.backward()
            optimizer_generator.step()
                
            if epoch ==0:
                auc = prediction(real_samples.cpu().detach().numpy(),epoch,labels,update)
               # sys.exit()
            elif epoch % 100==99:
                auc = prediction(generated_samples.cpu().detach().numpy(),epoch,labels)
                filename = 'omics2_'+str(update)+'.csv'
                
                if best is None:
                  best = auc
                  dd = pd.DataFrame(generated_samples.cpu().detach().numpy(),index=sample_name,columns=feature_name)
                  dd.to_csv(filename)
                  
                elif np.mean(auc)>np.mean(best):
                  best = auc
                  dd = pd.DataFrame(generated_samples.cpu().detach().numpy(),index=sample_name,columns=feature_name)
                  dd.to_csv(filename)
                  
    return best




