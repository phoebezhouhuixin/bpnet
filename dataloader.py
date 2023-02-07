import pickle
import numpy as np
from torch.utils.data import Dataset
import os

class BPdatasetv1(Dataset):
    def __init__(self, i, train = False, val = False):
        if train == True:
            dt = pickle.load(open(os.path.join('data','train4.p'),'rb'))
            self.input = np.swapaxes(dt['X_train'],1,2).astype('float32')
            self.output = np.swapaxes(dt['X_train'],1,2).astype('float32')
        elif val == True:
            dt = pickle.load(open(os.path.join('data','val4.p'),'rb'))
            self.input = np.swapaxes(dt['X_val'],1,2).astype('float32')
            self.output = np.swapaxes(dt['X_val'],1,2).astype('float32')
            
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        inp = self.input[idx]
        out = self.output[idx]
        return inp, out
    
class BPdatasetv2(Dataset):
    def __init__(self, i, train = False, val = False, test=False): # i think i was the fold number; now fixed at 4 which is the best fold
        if train == True:
            dt = pickle.load(open(os.path.join('data','train4.p'),'rb'))
            self.input = np.swapaxes(dt['X_train'],1,2).astype('float32')
            self.output = np.swapaxes(dt['Y_train'],1,2).astype('float32')
        elif val == True:
            dt = pickle.load(open(os.path.join('data','val4.p'),'rb'))
            self.input = np.swapaxes(dt['X_val'],1,2).astype('float32')
            self.output = np.swapaxes(dt['Y_val'],1,2).astype('float32')
        elif test == True:
            dt = pickle.load(open(os.path.join('data','test.p'),'rb'))
            self.input = np.swapaxes(dt['X_test'],1,2).astype('float32')
            self.output = np.swapaxes(dt['Y_test'],1,2).astype('float32')
        print(f"Initialised dataset object: input len {len(self.input)} ({self.input.shape}), output len {len(self.output)} ({self.output.shape})")
            
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        inp = self.input[idx]
        out = self.output[idx]
        return inp, out 

if __name__ == "__main__":
    with open("data/meta.p", "rb") as f:
        meta = pickle.load(f)
    print(f"meta: {type(meta)}, max ABP {meta['max_abp']}, min ABP {meta['min_abp']}")
    # ds = BPdatasetv2(0, train = True, val = False,  test = True)



