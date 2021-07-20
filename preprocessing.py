import os 
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from PIL import Image

num_classes=4;
divs=5;

PATH=r'/content/drive/MyDrive/CTM_Dataset/Dataset'
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

#without test set:
skf=StratifiedKFold(n_splits=divs,shuffle=True,random_state=1234)
data_dict=[{'train':(X[tr],Y[tr]),'test':(X[ts],Y[ts])} for tr,ts in skf.split(X,Y)]

#with test set:
skf2=StratifiedKFold(n_splits=2,shuffle=True,random_state=1234)
new_data_dict=[]
for x in data_dict:
  tr,tv = skf2.split(x['train'][0],x['train'][1])
  new_data_dict.append({'train':(x['train'][0][tr[0]],x['train'][1][tr[0]]),'val':(x['train'][0][tv[0]],x['train'][1][tv[0]]),'test':x['test']})


    

pickle.dump(data_dict,open(r'/content/drive/MyDrive/CTM_Dataset/Pickle/data.p','wb'))
