
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self,pickle_root,fold,stage,transform=None):
        
        self.transform=transform;
        self.X,self.Y=pickle.load(open(pickle_root,"rb"))[fold][stage]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        image=Image.open(self.X[idx])
        label=self.Y[idx]
        
        if self.transform:
            image=self.transform(image);
        return image,label
    
      
transform={'train':transforms.Compose(
    [
     transforms.ToPILImage(),
     transforms.Resize((224,224)),
     transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ColorJitter(saturation=(0.5, 2.0)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ]),
    'test':transforms.Compose(
    [
     transforms.ToPILImage(),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ])}


root=r'C:\Users\dafis\Desktop\CTM_SUMMER\Pickle\data.p'
batch_size=32

dataset={x:CustomDataset(root,0,x,transform[x]) for x in ['train','test']}
dataloader={x:DataLoader(dataset[x],batch_size,shuffle=True,num_workers=2) for x in ['train','test']}

