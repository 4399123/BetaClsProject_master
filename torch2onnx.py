import torch
import torch.nn
from utils.net import Net
import onnx
from onnxsim import simplify
import onnxoptimizer

# model=Net('convnext_pico.d1_in1k',num_class=11,pretrained=False,embeddingdim=1024,mode='pred')
model=Net('tf_efficientnet_b5.ns_jft_in1k',num_class=15,pretrained=False,embeddingdim=1024,mode='pred')
model.load_state_dict(torch.load('./pt/12ppm_n4/efficientnet_b5.pt',map_location='cpu'))
model.eval()

input_name = 'input'
output_name = 'output'
output_name1 = 'output1'


x = torch.randn(1,3,112,112,requires_grad=False)

torch.onnx.export(model, x, './onnx/best.onnx', input_names=[input_name], output_names=[output_name,output_name1], verbose=False,
                  dynamic_axes={
                        input_name:{0:'batch_size'},
                        output_name:{0:'batch_size'},
                        output_name1: {0: 'batch_size'}
                  })

print('step 1 ok')
model = onnx.load('./onnx/best.onnx')
newmodel=onnxoptimizer.optimize(model)
model_simp, check = simplify(newmodel)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp,'./onnx/best-smi.onnx')
print('step 2 ok')


