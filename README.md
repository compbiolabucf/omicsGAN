# omicsGAN
Sample datasets for cancer phenotype prediction are available through this link. https://drive.google.com/drive/folders/11Q0cIobQbraS7ig-38VEs-2IanL4cIPN?usp=sharing 

Users need to download all data necessary for a cancer analysis into the same folder as the codes. Updated omics datasets will be saved in the same folder as well. 
<!---If more than one cancer types are to be analyzed simulataneously, they must be stored in different directories as some datasets have duplicate names.--->
Omics datasets should be in feature by sample format and interaction netowrk should be in first omics data by second omics data format.

**Command:** omicsGAN.py, total number of update(K), first omics dataset, second omics dataset, interaction network \
**Sample command:** omicsGAN.py 5 mRNA.csv miRNA.csv bipartite_targetscan_gene.csv 

## Required Python packages
- Numpy
- Pandas
- sklearn
- PyTorch (pytorch version >=1.5.0, torchvision version >=0.6.0)

### **Framework**
![Image description](https://github.com/compbiolabucf/omicsGAN/blob/main/netflow-1.png)





<!---
## **BRCA_mRNA.py**
The framework to update mRNA expression for breast cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. 

## **BRCA_miRNA.py**
The framework to update miRNA expression for breast cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. 

## **rand_mRNA_BRCA.py**
The framework to update mRNA expression for breast cancer (update k) using a random network. It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data and **serial** that indicates the number of random network being used (serial=[1,2,...10]). It generates multiple synthetic data at multiple epochs and saves them at the current directory. As there are multiple phenotypes to be predicted, based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   

## **rand_miRNA_BRCA.py**
The framework to update miRNA expression for breast cancer (update k) using a random network. It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data and **serial** that indicates the number of random network being used (serial=[1,2,...10]). It generates multiple synthetic data at multiple epochs and saves them at the current directory. As there are multiple phenotypes to be predicted, based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   


## **OV_mRNA.py**
The framework to update mRNA expression for ovarian cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. 

## **OV_miRNA.py**
The framework to update miRNA expression for ovarian cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. 

## **LUAD_mRNA.py**
The framework to update mRNA expression for lung cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. 

## **LUAD_miRNA.py**
The framework to update miRNA expression for lung cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. 

--->
## **omicsGAN.py**
Users only need to run this code for generating synthetic data through omicsGAN using command line arguments mentioned above.  

## **omics1.py**
updates the first omics data

## **omics2.py**
updates the second omics data





