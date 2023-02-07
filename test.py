import pickle
from tqdm import tqdm
import torch
from model import Unet
from dataloader import *

bs = 256

model = Unet((256,1,1250)).cuda()
path = 'model/model.pt'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])

pick_path = 'output_val.p'

# test = torch.utils.data.DataLoader(BPdatasetv2(0, train = False, val = False,  test = True), batch_size=bs)
test = torch.utils.data.DataLoader(BPdatasetv2(0, train = False, val = True,  test = False), batch_size=bs)

temp1 = []
model.eval()
with torch.no_grad():
    for idx,(inputs,labels) in tqdm(enumerate(test),total=len(test),  disable=True):
        # print(f"inputs.shape: {inputs.shape} ({type(inputs)},  labels.shape: {labels.shape} ({type(labels)})")
        # inputs torch.Tensor shape (256,1,1250) and labels torch.Tensor shape (256,1,1250)
        
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs_v= model(inputs).cuda()
        # print(f"{idx} outputs_v.shape {outputs_v.shape}")
        temp1.extend(outputs_v)
        # if idx==9:
        #     break

temp1 = torch.stack(temp1)   
print(f"temp1.shape {temp1.shape}") # [(256*N, 1,1250)]
with open(pick_path,'wb') as f:
    pickle.dump(temp1.cpu().detach().numpy(), f)