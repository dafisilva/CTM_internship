import os 
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from PIL import Image

num_classes=4;
divs=5;

PATH=r'C:\Users\dafis\Desktop\CTM_SUMMER\Dataset'
classes=['Negative for Intraepithelial malignancy','Low squamous intra-epithelial lesion', 'High squamous intra-epithelial lesion',  'Squamous cell carcinoma']
directories=os.listdir(PATH)
X=[]
Y=[]

for folder in directories:
  
  sub_path=os.path.join(PATH,folder)
  imgs_name=[x for x in os.listdir(sub_path) if x.endswith('.jpg')]
  imgs=[os.path.join(sub_path,x) for x in imgs_name]
  labels=[classes.index(folder)for i in range(len(imgs_name))]
  X+=imgs;
  Y+=labels;

X,Y=np.array(X),np.array(Y)


skf=StratifiedKFold(n_splits=divs,shuffle=True);

data_dict=[{'train':(X[tr],Y[tr]),'test':(X[ts],Y[ts])} for tr,ts in skf.split(X,Y)]

    

pickle.dump(data_dict,open(r'C:\Users\dafis\Desktop\CTM_SUMMER\Pickle/data.p','wb'))






