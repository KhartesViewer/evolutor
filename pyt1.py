import torch
import torchvision

# gpu = torch.device("mps")
# cpu = torch.device("cpu")
# cuda = torch.device("cuda")
# print(gpu, cuda)
# print(torch.backends)

'''
cpu = torch.device('cpu')
gpu = torch.device('cuda') if torch.cuda.is_available() else cpu
gpu = torch.device('mps') if torch.backends.mps.is_available() else cpu
gpu = torch.device('xpu') if torch.xps.is_available() else cpu
'''

print("accelerator", torch.accelerator.current_accelerator(), torch.accelerator.is_available())

cpu = torch.device('cpu')
'''
if torch.cuda.is_available():
    gpu = torch.device('cuda')
elif torch.backends.mps.is_available():
    gpu = torch.device('mps')
elif torch.xpu.is_available:
    gpu = torch.device('xpu')
else:
    gpu = cpu
'''
'''
if torch.accelerator.is_available():
    gpu = torch.accelerator.current_accelerator()
else:
    gpu = cpu
'''

print("hello")
gpu = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# gpu = "cpu"

print("cpu", cpu)
print("gpu", gpu)

# print(torch.xpu.get_device_properties(0))

# rnd = torch.rand(2,3,3, device=gpu)
rnd = torch.zeros(2,3,3, device=gpu)
rnd[0,1,1] = 2.5
print(rnd.shape)
print(rnd)


rnd *= rnd
print(rnd.shape)
print(rnd)

# exit()

# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch/73164332

blurrer = torchvision.transforms.GaussianBlur(3)
brnd = blurrer.forward(rnd)
print(brnd.shape)
print(brnd)


