# omicsGAN
BRCA_data and OV_data contains the data necessary for breast cancer and ovarian cancer respectively. Processed mRNA expression file can be downloaded from the links below:\
BRCA: https://drive.google.com/file/d/19BKm9OOaHwqSBAJ1gbIjVyL_J9HH2SEN/view?usp=sharing \
OV: https://drive.google.com/file/d/1qH3VYvJAAxiqXpo9y25l-n7ISg7GDsqH/view?usp=sharing  \
codes with suffix "_BRCA" are for breast cancer analysis and "_OV" for ovarian cancer analysis. \
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







