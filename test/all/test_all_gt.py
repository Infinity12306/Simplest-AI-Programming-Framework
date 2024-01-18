import torch
from torch.nn.functional import relu, linear, conv2d, max_pool2d, cross_entropy

torch.set_printoptions(precision=4, sci_mode=False)

def read_data(f, dtype, device="cpu", requires_grad=True):
    shape_strs = f.readline().strip().split()
    shape = [int(data) for data in shape_strs]
    x_strs = f.readline().strip().split()
    x_list = [dtype(data) for data in x_strs]
    x = torch.tensor(x_list, dtype=dtype, device=device, requires_grad=False).reshape(shape)
    x = x.detach().clone().requires_grad_(requires_grad)
    return x

def check_equal(gt, pred, name, full_show=False):
    if full_show:
        print(f"pred_{name}:", pred, f"gt_{name}:", gt, sep="\n")
    print(f"gt_{name} - pred_{name}:")
    return ((abs(gt-pred) / abs(gt)) > 1e-3).to(int)
    # return ((abs(gt-pred)))

device = 'cpu'

with open("x.txt") as f1:
    with open("labels.txt") as f2:
        with open("loss.txt") as f3:
            x = read_data(f1, float, device, True)
            labels = read_data(f2, int, device, False)
            loss = torch.tensor(float((f3.readline().strip())), dtype=float, device=device)

with open("fc1_w.txt") as f1:
    with open("fc2_w.txt") as f2:
        with open("conv1_w.txt") as f3:
            with open("conv2_w.txt") as f4:
                fc1_w = read_data(f1, float, device, True)
                fc2_w = read_data(f2, float, device, True)
                conv1_w = read_data(f3, float, device, True)
                conv2_w = read_data(f4, float, device, True)

with open("fc1_dw.txt", "r") as f1:
    with open("fc2_dw.txt") as f2:
        with open("conv1_dw.txt") as f3:
            with open("conv2_dw.txt") as f4:
                with open("dx.txt") as f5:
                    fc1_dw = read_data(f1, float, device, False)
                    fc2_dw = read_data(f2, float, device, False)
                    conv1_dw = read_data(f3, float, device, False)
                    conv2_dw = read_data(f4, float, device, False)
                    dx = read_data(f5, float, device, False)

y1 = conv2d(x, conv1_w, padding=1)
y2 = relu(y1)
y3 = max_pool2d(y2, kernel_size=2)
y4 = conv2d(y3, conv2_w, padding=1)
y5 = relu(y4)
y6 = max_pool2d(y5, kernel_size=2)

y7 = y6.view(y6.size(0), -1)

y8 = linear(y7, fc1_w.t())
y9 = relu(y8)
y10 = linear(y9, fc2_w.t())

loss_gt = cross_entropy(y10, labels)
loss_gt.backward()

print(check_equal(loss_gt, loss, "loss", True))
print(check_equal(x.grad, dx, "dx"))
print(check_equal(fc1_w.grad, fc1_dw, "fc1_dw"))
print(check_equal(fc2_w.grad, fc2_dw, "fc2_dw"))
print(check_equal(conv1_w.grad, conv1_dw, "conv1_dw"))
print(check_equal(conv2_w.grad, conv2_dw, "conv2_dw"))