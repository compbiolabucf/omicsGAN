import os
import sys
import pandas as pd 
import numpy as np 

total_update = int(sys.argv[1])
mRNA_file = sys.argv[2]
miRNA_file = sys.argv[3]
adj_file = sys.argv[4]

for i in range(1, total_update+1):
	os.system('python3 -W ignore BRCA_mRNA.py '+str(i)+' '+mRNA_file+' '+miRNA_file+' '+adj_file)
	os.system('python3 -W ignore BRCA_miRNA.py '+str(i)+' '+mRNA_file+' '+miRNA_file+' '+adj_file)

best_mRNA = pd.read_csv('best_mRNA.txt',header=None)
keep_mRNA = np.argsort(best_mRNA.values,axis=0)[::-1][0][0]

best_miRNA = pd.read_csv('best_miRNA.txt',header=None)
keep_miRNA = np.argsort(best_miRNA.values,axis=0)[::-1][0][0]

for i in range(1, total_update+1):
	if i!=keep_mRNA:
		os.remove("mRNA_BRCA"+str(i)+".csv") 

	if i!=keep_miRNA:
		os.remove("miRNA_BRCA"+str(i)+".csv")

os.remove('best_mRNA.txt')
os.remove('best_miRNA.txt')