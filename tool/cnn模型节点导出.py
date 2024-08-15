import torch
import timm
from torchsummary import summary
import time

x = torch.randn(1, 3, 224, 224)
model = timm.create_model('resnext50_32x4d')
preds = model(x)
print('preds shape: {}'.format(preds.shape))

all_feature_extractor = timm.create_model('convnext_tiny_in22ft1k', features_only=True)
all_features = all_feature_extractor(x)
print('All {} Features: '.format(len(all_features)))
for i in range(len(all_features)):
    print('feature {} shape: {}'.format(i, all_features[i].shape))


out_indices = [1, 2, 3]
selected_feature_extractor = timm.create_model('convnext_tiny_in22ft1k', features_only=True, out_indices=out_indices)
selected_features = selected_feature_extractor(x)
print('Selected Features: ')
for i in range(len(out_indices)):
    print('feature {} shape: {}'.format(out_indices[i], selected_features[i].shape))


# m1 = timm.create_model('resnet50', pretrained=False, num_classes=0, global_pool='').to('cuda')
# summary(m1,(3,224,224))
#
# time.sleep(2)
#
# m2 = timm.create_model('resnet50', pretrained=False, features_only=True).to('cuda')
# summary(m2,(3,224,224))

# o = m1(torch.randn(1, 3, 224, 224))
# print(f'无分类层、无全局池化层 输出: {o.shape}')

# m2 = timm.create_model('resnet50', pretrained=False, num_classes=0).to('cpu')
#
# o = m2(torch.randn(1, 3, 224, 224))
# print(o.shape[-1])
timm.create_model('convnext_tiny_in22ft1k', pretrained=True)