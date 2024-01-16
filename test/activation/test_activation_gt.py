import torch
from torch.nn.functional import relu, sigmoid

device = 'cpu'
with open('X.txt', 'r') as f1:
    with open('Dy.txt', 'r') as f2:
        data_strs = f1.readline().strip().split()
        data_list = [float(data) for data in data_strs]
        data = torch.tensor(data_list, 
                                dtype=torch.float32, device=device, requires_grad=True)
        dy_strs = f2.readline().strip().split()
        dy_list = [float(data) for data in dy_strs]
        dy = torch.tensor(dy_list,
                                dtype=torch.float32, device=device, requires_grad=False)
        
with open("relu_Y.txt", "r") as f1:
    with open("relu_Dx.txt", "r") as f2:
        relu_y_strs = f1.readline().strip().split()
        relu_y_list = [float(y) for y in relu_y_strs]
        relu_y = torch.tensor(relu_y_list, dtype=torch.float32, device=device, requires_grad=False)
        
        relu_dx_strs = f2.readline().strip().split()
        relu_dx_list = [float(dx) for dx in relu_dx_strs]
        relu_dx = torch.tensor(relu_dx_list, dtype=torch.float32, device=device, requires_grad=False)
        
with open("sig_Y.txt", "r") as f1:
    with open("sig_Dx.txt", "r") as f2:
        sig_y_strs = f1.readline().strip().split()
        sig_y_list = [float(y) for y in sig_y_strs]
        sig_y = torch.tensor(sig_y_list, dtype=torch.float32, device=device, requires_grad=False)
        
        sig_dx_strs = f2.readline().strip().split()
        sig_dx_list = [float(dx) for dx in sig_dx_strs]
        sig_dx = torch.tensor(sig_dx_list, dtype=torch.float32, device=device, requires_grad=False)

def f(x):
    print((x > 1e-5).to(int))

out1 = relu(data)
out1.backward(dy)
print("gt_y - relu_pred_y:")
f(out1 - relu_y)
print("gt_dx - relu_pred_dx")
f(data.grad - relu_dx)

data.grad.zero_()
out2 = sigmoid(data)
out2.backward(dy)
print("gt_y - sig_pred_y:")
f(out2 - sig_y)
print("gt_dx - sig_pred_dx")
f(data.grad - sig_dx)
