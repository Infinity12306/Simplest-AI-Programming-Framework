import torch
# from torch.nn.functional import 

torch.set_printoptions(precision=4, sci_mode=False)

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
    return ((abs(gt-pred)))
    # return ((abs(gt-pred) / abs(gt)) > 1e-2).to(int)

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
            dx = read_data(f1, float, device, False)
            dw = read_data(f2, float, device, False)
            dy = read_data(f3, float, device, False)
            
print(check_equal(gt_y, y, "y"))
print(check_equal(x.grad, dx, "dx"))
print(check_equal(w.grad, dw, "dw"))