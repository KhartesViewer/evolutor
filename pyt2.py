import time
import torch
import numpy as np

class Timer():

    def __init__(self, active=True):
        self.t0 = time.time()
        self.active = active

    def __call__(self, msg=""):
        self.time(msg)


    def time(self, msg=""):
        t = time.time()
        if self.active:
            print("%.3f %s"%(t-self.t0, msg))
        self.t0 = t


print("accelerator", torch.accelerator.current_accelerator(), torch.accelerator.is_available())

cpu = torch.device('cpu')
gpu = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# gpu = "cpu"

print("cpu", cpu)
print("gpu", gpu)

# print(torch.xpu.get_device_properties(0))

sz = 8000

t = Timer()

grnd = torch.rand(sz, sz, device=gpu)
print(grnd.shape, grnd.dtype, grnd.device)
t("create grnd")

crnd = torch.rand(sz, sz, device=cpu)
print(crnd.shape, crnd.dtype, crnd.device)
t("create crnd")

# same precision as default torch float precision
nrnd = np.random.rand(sz, sz).astype(np.float32)
print(nrnd.shape, nrnd.dtype)
t("create nrnd")

# Note that MPS does not support torch.float64 type
gnrnd = torch.from_numpy(nrnd).to(gpu)
print(gnrnd.shape, gnrnd.dtype)
t("create gnrnd")

grr = grnd @ grnd
print(grr.device)
t("grr")

ngrr = grr.cpu().numpy()
t("ngrr")

crr = crnd @ crnd
print(crr.device)
t("crr")

nrr = nrnd @ nrnd
t("nrr")

exit()

# rnd[0,1,1] = 2.5
# print(rnd.shape)
# print(rnd)


rnd *= rnd
print(rnd.shape)
print(rnd)

# exit()

# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch/73164332

blurrer = torchvision.transforms.GaussianBlur(3)
brnd = blurrer.forward(rnd)
print(brnd.shape)
print(brnd)


