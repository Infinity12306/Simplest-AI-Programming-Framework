import numpy as np
import argparse

def read_data(f, dtype):
    shape_strs = f.readline().strip().split()
    shape = np.array([int(data) for data in shape_strs])
    
    data_strs = f.readline().strip().split()
    data = np.array([dtype(data) for data in data_strs]).reshape(shape)
    return data

def check_equal(x, y):
    print("pred:\n", x, "\n")
    print("gt:\n", y, "\n")
    return (abs(x - y) > 1e-5).astype(int)

parser = argparse.ArgumentParser()
parser.add_argument("-xt", type=int)
parser.add_argument("-wt", type=int)
args = parser.parse_args()

with open("X.txt") as fx:
    with open("W.txt") as fw:
        with open("Y.txt") as fy:
            x = read_data(fx, float)
            w = read_data(fw, float)
            print("x:", x, "\n", "w:", w, sep="\n")
            y = read_data(fy, float)
            str_x = "x"
            str_w = "w"
            if args.xt:
                x = x.T
                str_x = "x.T"
            if args.wt:
                w = w.T
                str_w = "w.T"
            gt_y = x @ w
            print(str_x + " @ " + str_w, check_equal(y, gt_y), sep="\n")