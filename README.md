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

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">




## **mRNA_BRCA.py**
The framework to update mRNA expression for breast cancer (update k). 
