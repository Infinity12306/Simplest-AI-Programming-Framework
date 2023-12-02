import numpy as np
import torch

np.set_printoptions(suppress=True, precision=6)
W = []
with open("W.txt", "r") as f_w:
    lines = f_w.readlines()
    rnum = 0
    for idx in range(len(lines)):
        if lines[idx] == "\n":
            break
        rnum += 1
    raw_mat = lines[:rnum]
    W = [line.strip().split() for line in raw_mat]
        
X = []
with open("X.txt", "r") as f_x:
    lines = f_x.readlines()
    for idx in range(len(lines))[::-1]:
        if lines[idx] == "\n":
            lines.pop(idx)
        else:
            break
    rnum = 0
    for idx in range(len(lines)):
        if lines[idx] == "\n":
            break
        rnum += 1
    bnum = (len(lines) + 1) // (rnum + 1)
    for b in range(bnum):
        cur_idx = b*(rnum+1)
        raw_mat = lines[cur_idx : (cur_idx+rnum)]
        mat = [line.strip().split() for line in raw_mat]
        X.append(mat)
        
Dy = []
with open("Dy.txt", "r") as f_dy:
    lines = f_dy.readlines()
    for idx in range(len(lines))[::-1]:
        if lines[idx] == "\n":
            lines.pop(idx)
        else:
            break
    rnum = 0
    for idx in range(len(lines)):
        if lines[idx] == "\n":
            break
        rnum += 1
    bnum = (len(lines) + 1) // (rnum + 1)
    for b in range(bnum):
        cur_idx = b*(rnum+1)
        raw_mat = lines[cur_idx : (cur_idx+rnum)]
        mat = [line.strip().split() for line in raw_mat]
        Dy.append(mat)
        
W_np = np.array(W, dtype=np.float32)
W_tensor = torch.from_numpy(W_np)
X_np = np.array(X, dtype=np.float32)
X_tensor = torch.from_numpy(X_np)
Dy_np = np.array(Dy, dtype=np.float32)
Dy_tensor = torch.from_numpy(Dy_np)

Y_np = torch.matmul(X_tensor, W_tensor).numpy()
print("Ground Truth of Y:\n", Y_np)

Dw_np = torch.matmul(X_tensor.permute(0, 2, 1), Dy_tensor).numpy()
Dw_np = np.mean(Dw_np, axis=0)
Dx_np = torch.matmul(Dy_tensor, W_tensor.T).numpy()
print("Ground Truth of Dw:\n", Dw_np)
print("Ground Truth of Dx:\n", Dx_np)
