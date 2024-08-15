import torch
import torch.nn
import timm
from utils.net import Net
import os

if not os.path.exists('./torchscript'):
    os.makedirs('./torchscript')


model=Net('convnext_pico.d1_in1k',num_class=9,pretrained=False,embeddingdim=256,mode='pred')
model.load_state_dict(torch.load('./pt/best.pt',map_location='cpu'))
model.eval()



x = torch.randn(1,3,112,112,requires_grad=False)

script_module = torch.jit.trace(model, x)

print('script_module.graph: ')
print(script_module.graph)

script_module.save('./torchscript/clsslib.pt')