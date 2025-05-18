import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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

'''
# print(torch.xpu.get_device_properties(0))

sz = 10000

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
'''

# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch/73164332

def gaussian_kernel_1d(sigma, num_sigmas = 3.):
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

def gaussian_filter_2d(img, sigma):
    kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    print("kernel", kernel_1d.shape, kernel_1d.dtype, kernel_1d.device)
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    img = img.unsqueeze(0).unsqueeze_(0)  # Make copy, make 4D for ``conv2d()``
    print("img", img.shape, img.dtype, img.device)
    # Convolve along columns and rows
    # img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    # img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding="same")
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding="same")
    return img.squeeze_(0).squeeze_(0)  # Make 2D again

sz = 500
nrnd = np.random.rand(sz, sz).astype(np.float32)
im = Image.fromarray((nrnd*255).astype(np.uint8))
# im.show()

grnd = torch.from_numpy(nrnd).to(gpu)

bgrnd = gaussian_filter_2d(grnd, 2.)

bim = Image.fromarray((bgrnd.cpu().numpy()*255).astype(np.uint8))
bim.show()
