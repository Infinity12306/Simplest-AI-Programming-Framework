import torch
from torch.nn.functional import relu, sigmoid

device = 'cpu'
with open('X.txt', 'r') as f1:
    with open('Dy. txt', 'r') as f2:
        data = torch.tensor(f1.readline().strip().split(), 
                                dtype=torch.float32, device=device, requires_grad=True)
        dy = torch.tensor(f2.readline().strip().split(), 
                                dtype=torch.float32, device=device, requires_grad=False)

        out1 = relu(data)
        out1.grad = dy
        out1.backward()
        print("relu_y:")
        print(out1)
        print("relu_dx")
        print(data.grad)

        data.grad = 0
        out2 = sigmoid(data)
        out2.grad = dy
        out2.backward()
        print("sig_y:")
        print(out2)
        print("sig_dx")
        print(data.grad)
