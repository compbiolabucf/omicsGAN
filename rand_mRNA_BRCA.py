import torch
from torch import nn
from sklearn.metrics import accuracy_score
import math
#import matplotlib.pyplot as plt
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
import os 
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn import svm
import sys
from functools import reduce
from sklearn.metrics import f1_score

torch.manual_seed(111)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
serial = 1    ###### number of random network 1-10
update = 1    ###### number of updates, starts from 1

print('mRNA_'+str(serial)+'_'+str(update))

class Drug_Dataset(Dataset):
    def __init__(self, input_file, target_file, transform=None, split=None):
        
        self.transform = transform  # data transform
        self.split = split  # dataset split for train/val/test

        # need to use astype convert the dtype from object to float32
        self.input_data = input_file.astype(
            np.float32, copy=False)
        self.target_data = target_file.astype(
            np.float32, copy=False)
        
        
    def __getitem__(self,idx):

        sample = {'input_data': self.input_data[:, idx],
                  'target_data': self.target_data[idx, :]}

        if self.transform:
            sample = self.transform(sample)   

        return sample['input_data'], sample['target_data']
    
    def __len__(self):
        return len(self.target_data)
    
  
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return {'input_data': torch.from_numpy(sample['input_data']),
                'target_data': torch.from_numpy(sample['target_data'])}


def data_split(dataset, split_ratio=[0.8, 0.1, 0.1], shuffle=False, manual_seed=None):

    length = dataset.__len__()
    indices = list(range(0, length))

    assert (sum(split_ratio) == 1), "Partial dataset is not used"

    if manual_seed is None:
        manual_seed = random.randint(1, 10000)
        
    if shuffle == True:
        random.seed(manual_seed)
        random.shuffle(indices)    

    breakpoint_train = floor(split_ratio[0] * length)
    breakpoint_val = floor(split_ratio[1] * length)

    idx_train = indices[:breakpoint_train]
    idx_val = indices[breakpoint_train:breakpoint_train+breakpoint_val]
    idx_test = indices[breakpoint_train+breakpoint_val:]

    return idx_train, idx_val, idx_test

def train(model, criterion, optimizer, train_loader):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        return loss


def valid(model, val_loader):

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):

            output = model(data)
            auc = roc_auc_score(target,output)

    return auc


class Net_final(nn.Module):

    def __init__(self, n_input,n_hidden, n_output):
        super(Net_final, self).__init__()
        
        self.hidden1 = nn.Linear(n_input,n_hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.sigmoid1 = nn.ReLU()
        self.dout1 = nn.Dropout(0.2)

        self.hidden2 = nn.Linear(n_hidden, 100, bias=False)
        self.bn2 = nn.BatchNorm1d(100)
        self.sigmoid2 = nn.ReLU()
        self.dout2 = nn.Dropout(0.2)

        self.hidden3 = nn.Linear(100, n_output, bias=False)
        self.sigmoid3=nn.Sigmoid()
        
    def forward(self, x):
        
        out = self.hidden1(x)
        out = self.bn1(out)
        out = self.sigmoid1(out)
        out = self.dout1(out)
        out = self.hidden2(out)
        out = self.bn2(out)
        out = self.sigmoid2(out)
        out = self.dout2(out)
        out = self.hidden3(out)
        out = self.sigmoid3(out)
        
        return out


def prediction1(mRNA_value,mRNA_sample,GAN_epoch):
  mRNA_value = np.array(mRNA_value).astype(float).transpose()
  labels= pd.read_csv('brca_clinical.csv',delimiter=',')
  labels=np.array(labels)
  label_name=labels[:,0]
  labels=labels[:,1:]
  labels[labels=='Positive']=1
  labels[labels=='Negative']=0

  xy, x_ind, y_ind = np.intersect1d(mRNA_sample, label_name, return_indices=True)
  labels=labels[y_ind,:]
  mRNA_value_final=mRNA_value[:,x_ind]
  #print(mRNA_value_final.shape)

  trial=50
  result2=[]
  result1=[]
  col1=np.array([0,3])
  dataset_transform = transforms.Compose([ToTensor()])

  for col in col1:
    for n_trial in range(trial):
      input_array=mRNA_value_final
      N_input =  np.size(mRNA_value_final,0) 
      N_hidden=  150 
      N_target = 1
      N_EPOCHS = 100
      LR = 0.01  

      label=labels[:,col]
      label=np.expand_dims(label, axis=1)

      dataset = Drug_Dataset(input_array,label,transform=dataset_transform)
      
      model = Net_final(N_input,N_hidden, N_target)
      Criterion = nn.BCELoss()

      Optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)
      scheduler = torch.optim.lr_scheduler.ExponentialLR(Optimizer, 0.99, last_epoch=-1)
      train_idx, val_idx, test_idx = data_split(dataset, split_ratio=[0.6, .2, 0.2],
                                                  shuffle=True, manual_seed=n_trial)

      ind_0=np.where(label[train_idx]==0)[0]
      ind_1=np.where(label[train_idx]==1)[0]  

      ind_0 = np.array(train_idx)[ind_0]
      ind_1 = np.array(train_idx)[ind_1]

      if np.size(ind_0)>np.size(ind_1):
        ind_11=np.random.choice(ind_1,np.size(ind_0))
        new_train_idx = np.concatenate([ind_11,ind_0])
      else:
        ind_00=np.random.choice(ind_0,np.size(ind_1))
        new_train_idx = np.concatenate([ind_00,ind_1])

      train_sampler = SubsetRandomSampler(new_train_idx)
      val_sampler = SubsetRandomSampler(val_idx)
      test_sampler = SubsetRandomSampler(test_idx)

      train_loader_final = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=new_train_idx.__len__(),
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True)

      val_loader_final = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=val_idx.__len__(),
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True)

      test_loader_final = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=test_idx.__len__(),
                                                  sampler=test_sampler,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

      best_val=None
  
      for epoch in range(N_EPOCHS):

        train_loss = train(model, Criterion, Optimizer, train_loader_final)
        val_auc = valid(model, val_loader_final)
  
        if best_val is None:
          is_best = True
          best_val = val_auc
        else:
          is_best = val_auc > best_val
          best_val = max(val_auc, best_val)

        if is_best:  # make a copy of the best model
          model_best = copy.deepcopy(model)
    
        scheduler.step()

      auc = valid(model_best, test_loader_final)   
      result1.append(auc)
      #print(n_trial,' ',auc)   
      
  result1=np.reshape(result1,(np.size(col1),-1))
  #print(pd.DataFrame(mRNA_value))
  if GAN_epoch==0:
    print('Average AUC (STD) for real data:', np.mean(result1,axis=1), '(',np.std(result1,axis=1),')')
  else:
    print('Average AUC (STD) for generated data at epoch',GAN_epoch,':', np.mean(result1,axis=1), '(',np.std(result1,axis=1),')')


def SVM(X_train, y_train, X_test, y_test):

  clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
  #clf = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train,y_train)
  y_pred = clf.predict_proba(X_test)[:,1]
  
  y_pred_bin = np.copy(y_pred)
  y_pred_bin[y_pred_bin < 0.5] = 0
  y_pred_bin[y_pred_bin >= 0.5] = 1

  return roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred_bin),f1_score(y_test, y_pred_bin)


def prediction(mRNA_value,mRNA_sample,GAN_epoch):
  X = np.array(mRNA_value).astype(float)
  labels= pd.read_csv('brca_clinical.csv',delimiter=',')
  labels=np.array(labels)
  label_name=labels[:,0]
  labels=labels[:,1:]
  labels[labels=='Positive']=1
  labels[labels=='Negative']=0

  xy, x_ind, y_ind = np.intersect1d(mRNA_sample, label_name, return_indices=True)
  labels=labels[y_ind,:].astype(int)
  X=X[x_ind,:]
 
  trial=50
  AUC_all=[]
  ACC_all=[]

  features=[]
  for col in np.array([0,3]):
    y = labels[:,col]
    AUC = []
    ACC = []
    for i in range(trial):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

      auc, acc ,f1 = SVM(X_train, y_train, X_test, y_test)
      AUC.append(auc)
      ACC.append(f1)
    AUC_all.append(AUC)
    ACC_all.append(np.mean(ACC)) 
  print(np.size(y_test))
  AUC_all = np.array(AUC_all).transpose()
  #np.savetxt('/home/ahmed/Desktop/miRNA-mRNA/Breast_cancer/temp_folder/auc/mRNA_1_4_auc.csv',AUC_all,delimiter=',',fmt='%s')
  if GAN_epoch==0:
    print('AUC for real data:', np.mean(AUC_all,axis=0), np.mean(ACC_all,axis=0))
  else:
    print('AUC for generated data at epoch',GAN_epoch,':', AUC_all)



def adj_matrix2():
  miRNA = pd.read_csv('miRNA.csv',delimiter=',',header=None)
  miRNA=np.array(miRNA)
  mRNA = pd.read_csv('mRNA_value.csv',delimiter=',')
  mRNA=np.array(mRNA)
  d1=miRNA[1:,1].astype(float)
  feature = pd.read_csv('mRNA_feature.csv',delimiter=',')
  feature=np.array(feature)
  sample = pd.read_csv('mRNA_sample.csv',delimiter=',')
  sample=np.array(sample)


  data = pd.read_excel('/home/ahmed/Desktop/miRNA-mRNA/Breast_cancer/bipartite_targetscan.xlsx',header=None)
  data=np.transpose(np.array(data))
  final_miRNA=data[1:,0]
  final_mRNA=data[0,1:]
  adj=data[1:,1:].astype(np.int8)
  
  adj[adj == 1] = -1
  adj[adj==0]=1
  print('data loaded')
  
  xy, x_ind, y_ind = np.intersect1d(miRNA[0,:],sample,return_indices=True)
  _, x_ind1, y_ind1 = np.intersect1d(miRNA[:,0],final_miRNA,return_indices=True)
  xy1, x_ind2, y_ind2 = np.intersect1d(feature.astype('<U15'),final_mRNA.astype('<U15'),return_indices=True)
  mRNA=mRNA[:,y_ind].astype(np.float32)
  miRNA=miRNA[:,x_ind]
  mRNA=mRNA[x_ind2,:]
  adj=adj[:,y_ind2]
  miRNA=miRNA[x_ind1,:].astype(np.float32)
  adj=adj[y_ind1,:].astype(np.float32)
  mRNA=np.exp2(mRNA)-.001

  
  return adj,mRNA,miRNA,xy,xy1



class Discriminator(nn.Module):
    def __init__(self,n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
    
            nn.Linear(128, 1),
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
            
            nn.Linear(768, n_input),
        )

    def forward(self, x):
        output = self.model(x)
        
        return output



miRNA = pd.read_csv('miRNA_adj2.csv',delimiter=',',header=None)
mRNA = pd.read_csv('mRNA_adj2.csv',delimiter=',',header=None)
adj = pd.read_csv('adj2.csv',delimiter=',',header=None)
sample_name = pd.read_csv('sample_827.csv',header=None)
miRNA=np.array(miRNA).astype(np.float32)
mRNA=np.array(mRNA).astype(np.float32)
adj=np.array(adj).astype(np.float32)
sample_name=np.array(sample_name)

mRNA=np.log1p(mRNA)
var = np.var(mRNA,axis=1)
ind = np.argsort(var)[::-1][0:200]
mRNA = mRNA[ind,:].transpose()
adj = adj[:,ind]
X_0 = torch.from_numpy(mRNA).to(device)

miRNA_file_name = 'miRNA_'+str(update-1)+'.csv'
mRNA_file_name = 'mRNA_'+str(update-1)+'.csv'

if update>1:
  miRNA = pd.read_csv(miRNA_file_name,delimiter=',',header=None)
  miRNA = np.array(miRNA).astype(np.float32).transpose()
  mRNA = pd.read_csv(mRNA_file_name,delimiter=',',header=None)
  mRNA = np.array(mRNA).astype(np.float32)

n_input_mRNA=np.size(mRNA,1)
sample_size = np.size(mRNA,0)

C=np.sqrt(np.outer(np.sum(np.absolute(adj),0),np.sum(np.absolute(adj),1)))
adj=np.divide(adj,C.transpose())

mRNA_train_data=torch.from_numpy(mRNA)
adj=torch.from_numpy(adj)

mRNA_train_labels = torch.zeros(sample_size)

mRNA_train_set = [(mRNA_train_data[i], mRNA_train_labels[i]) for i in range(sample_size)]

batch_size=sample_size
mRNA_train_loader = torch.utils.data.DataLoader(mRNA_train_set, 
                              batch_size=batch_size, shuffle=False)
       
discriminator = Discriminator(n_input_mRNA).to(device)
generator = Generator(n_input_mRNA).to(device)

lr_D = 5e-5
lr_G = 5e-5
num_epochs = 3000
critic_ite = 5
weight_clip = 0.01

optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=lr_G)

final_real_data = []
final_gen_data = []
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(mRNA_train_loader):

        miRNA_train_data = miRNA[:,n*batch_size:(n+1)*batch_size]
        miRNA_train_data=torch.from_numpy(miRNA_train_data)
        latent_value = torch.matmul(miRNA_train_data.t(),adj)
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
        loss_generator = -torch.mean(output_discriminator_fake) + .0001* LA.norm((X_0 - generated_samples), 2)

        loss_generator.backward()
        optimizer_generator.step()
            
        #print(f"Epoch: {epoch} Loss D.: {loss_discriminator}, Loss G.: {loss_generator}")
        if epoch ==0:
            prediction1(real_samples.cpu().detach().numpy(),sample_name,epoch)
            #print('pass')

        elif epoch % 300==299:
            prediction1(generated_samples.cpu().detach().numpy(),sample_name,epoch)
            filename = 'mRNA_'+str(serial)+'_'+str(update)+'_'+str(epoch)+'.csv'
            np.savetxt(filename,generated_samples.cpu().detach().numpy(),delimiter=',',fmt='%s')


