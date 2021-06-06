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


if len(sys.argv)!=2:
  print('input: code.py update')# gpu')
  sys.exit()

update = int(sys.argv[1])
gpu_num = 0#int(sys.argv[2])
torch.manual_seed(111)
device = torch.device('cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)
print('source_'+str(update))


def feature_selection(X, y):
  data_label1 = np.asarray([X[i] for i in range(len(y)) if y[i] == 1])
  data_label0 = np.asarray([X[i] for i in range(len(y)) if y[i] == 0])
  p = ttest_ind(data_label1, data_label0)[1]
  keep_ttest_index = np.argsort(p)[0:200] #np.where(p < .001)[0]
  return keep_ttest_index

def load_data(path):
  data = pd.read_csv(path,delimiter='\t',index_col=0)
  cols = data.columns.tolist()
  data = np.log1p(data)
  data.loc[:, 'var'] = data.loc[:, cols].var(axis=1)
  drop_index = data[data['var'] < .01].index.tolist()
  data.drop(index=drop_index, inplace=True)
  X = data[cols]

  return X

def SVM(X_train, y_train, X_test, y_test):

  clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
  #clf = RandomForestClassifier(n_estimators=150, random_state=0,min_samples_split=5).fit(X_train, y_train)
  y_pred = clf.predict_proba(X_test)[:,1]
  
  y_pred_bin = np.copy(y_pred)
  y_pred_bin[y_pred_bin < 0.5] = 0
  y_pred_bin[y_pred_bin >= 0.5] = 1

  return roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred_bin),f1_score(y_test, y_pred_bin)


def prediction(mRNA_value,labels,GAN_epoch):
  X = np.array(mRNA_value).astype(float)

  trial=100
  AUC_all=[]
  ACC_all=[]

  features=[]
  for col in range(1):
    y = labels
    AUC = []
    ACC = []
    for i in range(trial):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
      if np.size(np.unique(y_test))==1:
        continue
      target = feature_selection(X_train, y_train)
      X_train = X_train[:, target]
      X_test = X_test[:, target] 
      auc, acc ,f1 = SVM(X_train, y_train, X_test, y_test)
      AUC.append(auc)
      ACC.append(f1)
      
    AUC_all.append(AUC)
    ACC_all.append(np.mean(ACC)) 
    print(np.mean(auc))
  AUC_all = np.array(AUC_all).transpose()
  #np.savetxt('/home/ahmed/Desktop/miRNA-mRNA/Breast_cancer/temp_folder/auc/mRNA_1_4_auc.csv',AUC_all,delimiter=',',fmt='%s')
  if GAN_epoch==0:
    print('AUC for real data:', np.mean(AUC_all,axis=0), np.mean(ACC_all,axis=0))
  else:
    print('AUC for generated data at epoch',GAN_epoch,':', np.mean(AUC_all,axis=0))

  return np.mean(AUC_all,axis=0)


class Discriminator(nn.Module):
    def __init__(self,n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            #nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            #nn.Dropout(0.3),
            
            #nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Dropout(0.3),
    
            nn.Linear(256, 1),
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



mRNA = load_data('mRNA.csv')

spectral_count = pd.read_csv('spectral_count.tsv',delimiter='\t',index_col=0)
tf_sample_names = []

new = []
for i,x in enumerate(spectral_count.columns):
  if 'Spectral Counts' in x:
    names=x.split(':')
    names = [x[0:10] for x in names]
    names= ['TCGA-'+str(x) for x in names]
    l = len(names)
    for ii in range(l):
      new.append(spectral_count.iloc[:,i])
    tf_sample_names.extend(names)

new=np.array(new)
tf_sample_names, x_ind, y_ind = np.intersect1d(mRNA.columns,tf_sample_names,return_indices=True)
tf = new[y_ind,:].transpose()
mRNA = mRNA.iloc[:,x_ind]
miRNA = pd.DataFrame(tf, index=spectral_count.index, columns=tf_sample_names)

adj = pd.read_csv('bipartite_rbp.csv',index_col=0)
print(adj)
print(adj.shape)
sys.exit()
adj = adj.T

xy, x_ind, y_ind = np.intersect1d(mRNA.columns,miRNA.columns,return_indices=True)
_, x_ind1, y_ind1 = np.intersect1d(miRNA.index,adj.columns,return_indices=True)
xy1, x_ind2, y_ind2 = np.intersect1d(mRNA.index,adj.index,return_indices=True)

mRNA = mRNA.iloc[:,x_ind]
miRNA = miRNA.iloc[:,y_ind]
mRNA = mRNA.iloc[x_ind2,:]
miRNA = miRNA.iloc[x_ind1,:].transpose()
adj = adj.iloc[:,y_ind1]
adj = adj.iloc[y_ind2,:]


mRNA = mRNA.fillna(0)
miRNA = miRNA.fillna(0)

adj[adj==1] = -1
adj[adj==0] = 1

data = pd.read_csv('brca_tcga_pan_can_atlas_2018_clinical_data.tsv', delimiter='\t',usecols=[2,4])
data.dropna(subset=['Neoplasm Disease Stage American Joint Committee on Cancer Code'],inplace=True)

xy, x_ind, y_ind = np.intersect1d(mRNA.columns,data.iloc[:,0],return_indices=True)
mRNA = mRNA.iloc[:,x_ind].values
miRNA = miRNA.iloc[x_ind,:].values
y= data.iloc[y_ind,1].astype(str)

labels=np.zeros(np.shape(y))
ind3=np.where(y=='STAGE IV')[0]
ind4=np.where(y=='STAGE IIIB')[0]
ind2=np.where(y=='STAGE IIIA')[0]
#ind1=np.where(y=='STAGE III')[0]
ind5=np.where(y=='STAGE IIB')[0]
#ind6=np.where(y=='STAGE IIA')[0]
ind=reduce(np.union1d, (ind2,ind3,ind4,ind5))#ind1,ind2,
labels[ind]=1
y=labels.astype(np.float32)

mRNA = np.array(mRNA).transpose().astype(np.float32)
miRNA = np.array(miRNA).astype(np.float32)
adj = np.array(adj).astype(np.float32)

X_0 = torch.from_numpy(miRNA).to(device)

if update>1:
  mRNA_file_name = 'mRNA_'+str(update-1)+'.csv'
  miRNA_file_name = 'miRNA_'+str(update-1)+'.csv'

  mRNA = pd.read_csv(mRNA_file_name,delimiter=',',header=None)
  mRNA = np.array(mRNA).astype(np.float32)
  miRNA = pd.read_csv(miRNA_file_name,delimiter=',',header=None)
  miRNA = np.array(miRNA).astype(np.float32)

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

        """
        real_mean = torch.mean(real_samples,1)
        real_std = torch.std(real_samples,1)

        fake_mean = torch.mean(latent_value,1)
        fake_std = torch.std(latent_value,1)

        real_samples = ((real_samples.t() - real_mean) / real_std).t()
        latent_value = ((latent_value.t() - fake_mean) / fake_std).t()
        """   
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
            
        # Show loss
        #if epoch % 10 == 0 and n == 0:
        #if epoch % 10 == 0:
        #print(f"Epoch: {epoch} Loss D.: {loss_discriminator}, Loss G.: {loss_generator}")
        if epoch ==0:
            auc = prediction(real_samples.cpu().detach().numpy(),y,epoch)
           # sys.exit()
        elif epoch % 300==299:
            auc = prediction(generated_samples.cpu().detach().numpy(),y,epoch)
            #prediction(latent_value.cpu().detach().numpy(),y,epoch)
            #sys.exit()
            filename = 'miRNA_'+str(update)+'.csv'
            
            if best is None:
              best = np.mean(auc)
              best_epoch = epoch
              np.savetxt(filename,generated_samples.cpu().detach().numpy(),delimiter=',',fmt='%s')

            elif np.mean(auc)>np.mean(best):
              best = np.mean(auc)
              best_epoch = epoch
              np.savetxt(filename,generated_samples.cpu().detach().numpy(),delimiter=',',fmt='%s')
          
print(best)
print(best_epoch)




