import torch
# from torch.nn.functional import 

def read_data(f, dtype, device, requires_grad):
    shape_strs = f.readline().strip().split()
    shape = [int(data) for data in shape_strs]
    x_strs = f.readline().strip().split()
    x_list = [dtype(data) for data in x_strs]
    x = torch.tensor(x_list, dtype=dtype, device=device, requires_grad=requires_grad).reshape(shape)
    x = x.detach().clone().requires_grad_(True)
    return x 

def read_grad(f, shape, dtype, device):
    grad_strs = f.readline().strip().split()
    grad_list = [dtype(data) for data in grad_strs]
    grad = torch.tensor(grad_list, dtype=dtype, device=device, requires_grad=False).reshape(shape)
    return grad

def check_equal(x):
    return (abs(x) > 1e-5).to(int)

device = 'cpu'

with open("X.txt") as f1:
    with open("W.txt") as f2:
        with open("Y.txt") as f3:
            x = read_data(f1, float, device, True)
            w = read_data(f2, float, device, True)
            y = read_data(f3, float, device, True)

with open("Dx.txt", "r") as f1:
    with open("Dw.txt", "r") as f2:
        with open("Dy.txt") as f3:
            dx = read_grad(f1, x.size(), float, device)
            dw = read_grad(f2, w.size(), float, device)
            dy = read_grad(f3, y.size(), float, device)