# omicsGAN
Necessary datasets for all codes are available through the link below. \
https://drive.google.com/drive/folders/11Q0cIobQbraS7ig-38VEs-2IanL4cIPN?usp=sharing \
prefix of the code names are the cancer types and suffix tells whether it updates the mRNA or miRNA. Codes with prefix "RBP_" are for transcription factor-gene expression integration. \
All codes perform one update for one omics profile at a time. Users need to run the two codes for two omics profile of a cancer type in a sequential manner. For example, for lung cancer phenotype prediction, run each of LUAD_mRNA.py and LUAD_miRNA.py once that will provide the first update for both omics data. Then we can start the second update by running the same two codes once again. Command for running all the codes takes two argument in the following format \
**Code_name update** (e.g., LUAD_mRNA.py 1)\
Users need to download all data necessary for a cancer analysis into the same folder as the codes. If both cancer types are to be analyzed simulataneously, they must be stored in different directories as some datasets have duplicate names.   

## Required Python packages
- Numpy
- Pandas
- sklearn
- PyTorch (pytorch version >=1.5.0, torchvision version >=0.6.0)

### **Framework**
![Image description](https://github.com/compbiolabucf/omicsGAN/blob/main/netflow-1.png)





## **mRNA_BRCA.py**
The framework to update mRNA expression for breast cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. It generates multiple synthetic data at multiple epochs and saves them at the current directory. As there are multiple phenotypes to be predicted, based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   

## **miRNA_BRCA.py**
The framework to update miRNA expression for breast cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. It generates multiple synthetic data at multiple epochs and saves them at the current directory. As there are multiple phenotypes to be predicted, based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   

## **rand_mRNA_BRCA.py**
The framework to update mRNA expression for breast cancer (update k) using a random network. It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data and **serial** that indicates the number of random network being used (serial=[1,2,...10]). It generates multiple synthetic data at multiple epochs and saves them at the current directory. As there are multiple phenotypes to be predicted, based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   

## **rand_miRNA_BRCA.py**
The framework to update miRNA expression for breast cancer (update k) using a random network. It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data and **serial** that indicates the number of random network being used (serial=[1,2,...10]). It generates multiple synthetic data at multiple epochs and saves them at the current directory. As there are multiple phenotypes to be predicted, based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   



## **mRNA_OV.py**
The framework to update mRNA expression for ovarian cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_x^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. It generates multiple synthetic data at multiple epochs and saves them at the current directory. Based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data.   

## **miRNA_OV.py**
The framework to update miRNA expression for ovarian cancer (update k). It takes <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k-1)}"> and 
<img src="https://render.githubusercontent.com/render/math?math=H_x^{(k-1)}"> as input and generates <img src="https://render.githubusercontent.com/render/math?math=H_y^{(k)}"> . User has to define the value of variable **update** that represents the value of k in the generated data. It generates multiple synthetic data at multiple epochs and saves them at the current directory. Based on the printed validation AUC, user has to choose the best epoch and the corrosponding sythetic data. 







