import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stto import ST
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

# python wind2d.py C:\Vesuvius\Projects\evol1\circle.tif --no_cache

# python wind2d.py "C:\Vesuvius\scroll 1 masked xx000\02000.tif" --cache_dir C:\Vesuvius\Projects\evol1 --umbilicus 3960,2280 --decimation 8 --colormap tab20 --interpolation nearest --maxrad 1800


def loadTIFF(fname):
    try:
        # image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED).astype(np.float64)
        pimage = Image.open(str(fname))
        image = np.array(pimage).astype(np.float32)
        image /= 65535.
        print("image", image.shape, image.dtype, image.min(), image.max())
    except Exception as e:
        print("Error while loading",fname,e)
        return
    return image

t = Timer()
fname = r"/Users/dev/Desktop/Progs/Vesuvius/Projects/evol1/circle.tif"
# fname = r"C:\Vesuvius\Projects\evol1\circle.tif"
# fname = r"C:\Vesuvius\scroll 1 masked xx000\02000.tif"
nim = loadTIFF(fname)
t.time("loaded")

gim = torch.from_numpy(nim).to(gpu)
t.time("converted")

st = ST(gim)
t.time("ST created")
st.computeTensors()
t.time("tensors computed")

exit()

# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch/73164332

def gaussian_kernel_1d(sigma, num_sigmas = 3.):
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_().to(gpu)
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
im.show()

grnd = torch.from_numpy(nrnd).to(gpu)

bgrnd = gaussian_filter_2d(grnd, 5.)

bim = Image.fromarray((bgrnd.cpu().numpy()*255).astype(np.uint8))
bim.show()
