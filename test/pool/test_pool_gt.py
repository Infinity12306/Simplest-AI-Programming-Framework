import torch
from torch.nn.functional import max_pool2d

torch.set_printoptions(precision=4, threshold=float('inf'), sci_mode=False)

def read_data(f, dtype, device="cpu", requires_grad=True):
    shape_strs = f.readline().strip().split()
    shape = [int(data) for data in shape_strs]
    x_strs = f.readline().strip().split()
    x_list = [dtype(data) for data in x_strs]
    x = torch.tensor(x_list, dtype=dtype, device=device, requires_grad=False).reshape(shape)
    x = x.detach().clone().requires_grad_(requires_grad)
    return x 

def check_equal(gt, pred, name):
    print(f"pred_{name}:", pred, f"gt_{name}:", gt, sep="\n")
    print(f"gt_{name} - pred_{name}:")
    return (abs(gt-pred))
    # return ((abs(gt-pred) / abs(gt)) > 1e-2).to(int)

device = 'cpu'

with open("X.txt") as f1:
    with open("Y.txt") as f2:
        x = read_data(f1, float, device, True)
        y = read_data(f2, float, device, True)

with open("Dx.txt", "r") as f1:
    with open("Dy.txt") as f2:
        dx = read_data(f1, float, device, False)
        dy = read_data(f2, float, device, False)
        
gt_y = max_pool2d(x, kernel_size=2)
gt_y.backward(dy)

print(check_equal(x.grad, dx, "dx"))