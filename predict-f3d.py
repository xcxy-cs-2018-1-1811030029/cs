import sys
import numpy as np
#from torch.utils.data import DataLoader
from attsigmod import UNet
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def dataProcess(sx):
    std = np.std(sx)
    mea = np.mean(sx)
    sx = (sx-mea)/std
    return sx

def predict(model,inputs):
    inputs = inputs[np.newaxis,np.newaxis,:,:,:] #np.newaxis的作用是增加一个维度
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
    outputs = model(inputs).cpu().detach().numpy()
    return outputs

def predictGPU(model,inputs):
    inputs = inputs[np.newaxis,np.newaxis,:,:,:]
    print(inputs.shape)
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(cudaName)
    outputs = model(inputs).cpu().detach().numpy()
    return outputs

filename = './data/prediction/f3d/gxl.dat'

n1,n2,n3 = 512, 384, 128
v3d = np.fromfile(filename,dtype=np.single).reshape((n1,n2,n3))
v3d=dataProcess(v3d)
v3d = np.transpose(v3d)


cudaName = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpth = 'train'
is_loaded = True

res = np.zeros(v3d.shape)
v3d = v3d[:128,:128,:128]

model = UNet(in_channels=1, out_channels=1).to(cudaName)
#loaded_file = './check/model_min_%s.pth' % cpth
loaded_file = './check/model.pth'

if is_loaded:

    m = torch.load(loaded_file)['net']
    try:
        model.load_state_dict(m)
    except RuntimeError:
        print("The data was obtained through multi GPU training, so it was re read")
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in m.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model = model.to(cudaName)

    res = predictGPU(model, v3d)[0, 0, :, :, :]
    #res=np.transpose(res)
    # res[res>=0.3]=1
    # res[res<0.3]=0

fig = plt.figure(figsize=(12,12))
plt.subplot(1, 2, 1)
imgplot1 = plt.imshow(v3d[:,29,:],cmap=plt.cm.gray,aspect=1)
plt.xlabel('Traces',fontsize=15)
plt.ylabel('Times(ms)',fontsize=15)
plt.subplot(1, 2, 2)
imgplot2 = plt.imshow(res[:,29,:],cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.xlabel('Traces',fontsize=15)
plt.ylabel('Times(ms)',fontsize=15)
plt.show()
