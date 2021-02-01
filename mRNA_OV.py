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

from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn import svm
import sys
from functools import reduce
print(torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
update = 4
print('mRNA_'+str(update))

def SVM(X_train, y_train, X_test, y_test):
  clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
  #clf = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train,y_train)
  y_pred = clf.predict_proba(X_test)[:,1]
  
  y_pred_bin = np.copy(y_pred)
  y_pred_bin[y_pred_bin < 0.5] = 0
  y_pred_bin[y_pred_bin >= 0.5] = 1

  return roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred_bin)


def prediction(mRNA_value,mRNA_sample,GAN_epoch):
  #mRNA_value = pd.DataFrame(mRNA_value)
  X = np.array(mRNA_value).astype(float)
  #print(mRNA_value)
  #sys.exit()
  #X=np.log1p(mRNA_value)
 
  data = pd.read_csv('ov_tcga_clinical_data.tsv', delimiter='\t',usecols=[2,8])
  data.dropna(subset=['Neoplasm American Joint Committee on Cancer Clinical Group Stage'],inplace=True)
  xy, x_ind, y_ind = np.intersect1d(mRNA_sample,data.iloc[:,0],return_indices=True)
  X = X[x_ind,:]
  y= data.iloc[y_ind,1].astype(str)

  labels=np.zeros(np.shape(y))
  ind3=np.where(y=='Stage IV')[0]
  ind4=np.where(y=='Stage IIIC')[0]
  ind=reduce(np.union1d, (ind3,ind4))#ind1,ind2,
  labels[ind]=1
  y=labels

  trial=50
  AUC_all=[]
  ACC_all=[]

  features=[]
  for col in range(1):
  
    AUC = []
    ACC = []
    for i in range(50):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    #target = feature_selection(X_train, y_train)
    #features.append()
    #print(X_train.shape)
    #X_train = X_train[:, target]
    #X_test = X_test[:, target]

    #print(X_train.shape)
    #sys.exit()
      auc, acc = SVM(X_train, y_train, X_test, y_test)
      AUC.append(auc)
      ACC.append(acc)
    AUC_all.append(np.mean(AUC))
    ACC_all.append(np.mean(ACC)) 

  np.savetxt('/home/ahmed/Desktop/miRNA-mRNA/Ovarian_cancer/temp_folder/mRNA_auc.csv',AUC,delimiter=',',fmt='%s')
  if GAN_epoch==0:
    print('AUC for real data:', AUC_all)
  else:
    print('AUC for generated data at epoch',GAN_epoch,':', AUC_all)

  return AUC_all



def adj_matrix2():
  miRNA = pd.read_csv('miRNA.csv',delimiter=',',header=None)

  miRNA=np.array(miRNA)
  data = pd.read_csv('mRNA.csv',delimiter=',',header=None)
  data=np.array(data)
  mRNA=data[1:,1:]
  feature = data[1:,0]
  sample = data[0,1:]

  data = pd.read_excel('bipartite_targetscan.xlsx',header=None)
  data=np.transpose(np.array(data))
  final_miRNA=data[1:,0]
  final_mRNA=data[0,1:]
  adj=data[1:,1:].astype(np.int8)
  """
  aa=np.sum(adj,axis=1)
  ind=np.where(aa>600)[0]
  adj=np.delete(adj,ind,axis=0)
  final_miRNA=np.delete(final_miRNA,ind)
  
  temp=np.where(np.sum(np.absolute(adj),0)==0)[0]
  adj=np.delete(adj,temp,axis=1)
  final_mRNA=np.delete(final_mRNA,temp)
  """
  adj[adj == 1] = -1
  adj[adj==0]=1
  print('data loaded')

  xy, x_ind, y_ind = np.intersect1d(miRNA[0,:],sample,return_indices=True)
  _, x_ind1, y_ind1 = np.intersect1d(miRNA[:,0],final_miRNA,return_indices=True)
  xy1, x_ind2, y_ind2 = np.intersect1d(feature.astype('<U15'),final_mRNA.astype('<U15'),return_indices=True)
  mRNA=mRNA[:,y_ind].astype(float)
  miRNA=miRNA[:,x_ind]
  mRNA=mRNA[x_ind2,:]
  adj=adj[:,y_ind2]
  miRNA=miRNA[x_ind1,:].astype(float)
  adj=adj[y_ind1,:]
  mRNA=np.exp2(mRNA)-.001
  
  #mRNA=np.log2(mRNA+1)
  #mRNA=mRNA/sum(mRNA)
  miRNA=np.nan_to_num(miRNA)
  #miRNA=np.exp2(miRNA)-.001

  return adj,mRNA,miRNA,xy,xy1



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
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        return loss


def valid(model, val_loader):

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            output = model(data)
            output = output.cpu().numpy()
            target = target.numpy()
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
  #print(np.shape(mRNA_value))
  data = pd.read_csv('ov_tcga_clinical_data.tsv', delimiter='\t',usecols=[2,8])
  data.dropna(subset=['Neoplasm American Joint Committee on Cancer Clinical Group Stage'],inplace=True)
  xy, x_ind, y_ind = np.intersect1d(mRNA_sample,data.iloc[:,0],return_indices=True)
  mRNA_value_final = mRNA_value[:,x_ind]
  y= data.iloc[y_ind,1].astype(str)

  labels=np.zeros(np.shape(y))
  ind3=np.where(y=='Stage IV')[0]
  ind4=np.where(y=='Stage IIIC')[0]
  ind=reduce(np.union1d, (ind3,ind4))#ind1,ind2,
  labels[ind]=1

  trial=50
  result2=[]
  result1=[]
  col1=np.array([0])
  dataset_transform = transforms.Compose([ToTensor()])

  for col in col1:
    for n_trial in range(trial):
      input_array=mRNA_value_final
      N_input =  np.size(mRNA_value_final,0) 
      N_hidden=  150 
      N_target = 1
      N_EPOCHS = 100
      LR = 0.01  

      label=labels
      label=np.expand_dims(label, axis=1)

      dataset = Drug_Dataset(input_array,label,transform=dataset_transform)
      
      model = Net_final(N_input,N_hidden, N_target).to(device)
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
      result1.append(best_val)
      #print(n_trial,' ',auc)   
      
  result1=np.reshape(result1,(np.size(col1),-1))
  #print(pd.DataFrame(mRNA_value))
  if GAN_epoch==0:
    print('Average AUC (STD) for real data:', np.mean(result1,axis=1), '(',np.std(result1,axis=1),')')
  else:
    print('Average AUC (STD) for generated data at epoch',GAN_epoch,':', np.mean(result1,axis=1), '(',np.std(result1,axis=1),')')

  return np.mean(result1,axis=1)



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
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, n_input),
        )

    def forward(self, x):
        output = self.model(x)
        
        return output


miRNA = pd.read_csv('miRNA_adj2.csv',delimiter=',',header=None)
mRNA = pd.read_csv('mRNA_adj2.csv',delimiter=',',header=None)
adj = pd.read_csv('adj2.csv',delimiter=',',header=None)
sample_name = pd.read_csv('sample_415.csv',header=None)
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

if update>1:
  mRNA_file_name = 'mRNA_'+str(update-1)+'.csv'
  miRNA_file_name = 'miRNA_'+str(update-1)+'.csv'

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
       

best_loss_discriminator=0
num_epochs = 900
critic_ite = 5
weight_clip = 0.01

for lr_D in np.array([1e-7]):
  for lr_G in np.array([1e-4]):
    for alpha in np.array([1]):
      discriminator = Discriminator(n_input_mRNA).to(device)
      generator = Generator(n_input_mRNA).to(device)
      optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
      optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=lr_G)
      auc_value = []
      value=[]
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
          loss_generator = -torch.mean(output_discriminator_fake) + alpha* LA.norm((X_0 - generated_samples), 2)

          loss_generator.backward()
          optimizer_generator.step()

          if loss_discriminator<best_loss_discriminator:
            best_loss_discriminator = loss_discriminator
            best_mRNA = generated_samples.cpu().detach().numpy()

           # print(f"Epoch: {epoch} Loss D.: {loss_discriminator}, Loss G.: {loss_generator}")
          if epoch ==0:
            auc = prediction1(real_samples.cpu().detach().numpy(),sample_name,epoch)

          elif epoch%300==299:
            filename ='mRNA_'+str(update)+'_'+str(epoch)+'.csv'
            auc = prediction1(generated_samples.cpu().detach().numpy(),sample_name,epoch)
            np.savetxt(filename,generated_samples.cpu().detach().numpy(),delimiter=',',fmt='%s')

