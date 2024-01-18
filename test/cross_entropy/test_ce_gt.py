import torch
from torch.nn.functional import cross_entropy

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
    with open("labels.txt") as f2:
        with open("loss.txt") as f3:
            x = read_data(f1, float, device, True)
            labels = read_data(f2, int, device, False)
            loss = torch.tensor(float((f3.readline().strip())), dtype=float, device=device)

with open("Dx.txt", "r") as f1:
    dx = read_data(f1, float, device, False)

loss_gt = cross_entropy(x, labels)
loss_gt.backward()
            
print(check_equal(loss_gt, loss, "loss"))
print(check_equal(x.grad, dx, "dx"))