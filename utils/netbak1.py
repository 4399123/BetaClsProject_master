import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import os
from collections import OrderedDict, defaultdict

def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict

def load_state_dict(checkpoint_path, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        # _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        # _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=True, strict=False):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

class classifier(nn.Module):
    def __init__(self, in_ch, num_classes,embeddingdim):
        super(classifier, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_ch, embeddingdim)
        self.fc2 = nn.Linear(embeddingdim, num_classes)

    def forward(self, x):
        H,W=x.size()[2:]
        inputsz=np.array([H,W])
        outputsz = np.array([1, 1])
        stridesz = np.floor(inputsz / outputsz).astype(np.int32)
        kernelsz = inputsz - (outputsz - 1) * stridesz
        x=F.avg_pool2d(x,kernel_size=list(kernelsz),stride=list(stridesz))
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feature=F.relu(x)
        out = self.fc2(feature)
        return feature,out


class Net(nn.Module):
    def __init__(self,model_name ,num_class,pretrained=False,mode='train',embeddingdim=512):
        super(Net, self).__init__()
        model = timm.create_model(model_name, pretrained=pretrained)
        # model = timm.create_model(model_name, pretrained=pretrained,img_size=112)
        # load_checkpoint(model, './utils/premodels/convnext_pico.pth')
        fc_in_ch=list(model.named_modules())[-1][1].in_features
        self.backbone =nn.Sequential(*list(model.children())[:-2])
        self.classifier = classifier(fc_in_ch, num_class,embeddingdim)
        self.mode=mode

    def forward(self, x):
        x = self.backbone(x)
        feature,out= self.classifier(x)
        if self.mode=='train':
            return feature,out
        elif self.mode=='eval':
            return feature, out
        else:
            index = torch.argmax( out, dim=1)
            with torch.no_grad():
                score=torch.softmax(out, dim=1)[0][index]
                score=torch.tensor(score, dtype=torch.float32)
            index = torch.tensor(index, dtype=torch.float32)
            return index,score

if __name__=="__main__":
    net=Net('convnext_pico.d1_in1k',num_class=9,pretrained=False,embeddingdim=128,mode='pred')
    # summary(net,(3,224,224))
    net.eval()
    in_ten = torch.randn(3, 3,224, 224)
    index,score=net(in_ten)
    print(index)
    print(score)