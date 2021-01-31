import numpy as np 
import pandas as pd 
from scipy.stats import ttest_ind
import sys

file='mRNA_adj2.csv'
print(file)
mRNA = pd.read_csv(file,delimiter=',',header=None)
mRNA = np.array(mRNA)
var = np.var(mRNA,axis=1)
ind = np.argsort(var)[::-1][0:200]
if file=='mRNA_adj2.csv':
	mRNA = mRNA[ind,:].transpose()
mRNA=np.log1p(mRNA)
mRNA=np.nan_to_num(mRNA)
print(mRNA.shape)

sample_name = pd.read_csv('sample_827.csv',header=None)
sample_name=np.array(sample_name)

labels= pd.read_csv('brca_clinical.csv',delimiter=',')
labels=np.array(labels)
label_name=labels[:,0]
labels=labels[:,1:]
labels[labels=='Positive']=1
labels[labels=='Negative']=0

xy, x_ind, y_ind = np.intersect1d(sample_name, label_name, return_indices=True)
labels=labels[y_ind,:].astype(int)
X=mRNA[x_ind,:]
p_total=[]
for i in ([0,2]):
	y=labels[:,i]
	data_label1 = np.asarray([X[i] for i in range(len(y)) if y[i] == 1])
	data_label0 = np.asarray([X[i] for i in range(len(y)) if y[i] == 0])
	p = ttest_ind(data_label1, data_label0)[1]
	#print(p)
	p_total.append(p)
	keep_ttest_index = np.where(p < 1e-2)
	print(np.size(keep_ttest_index))
print(np.shape(p_total))
