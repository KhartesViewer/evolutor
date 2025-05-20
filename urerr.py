import torch
import torch.nn.functional as F


print("accelerator", torch.accelerator.current_accelerator(), "is available:", torch.accelerator.is_available())

cpu = torch.device('cpu')
gpu = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

print("cpu device:", cpu)
print("gpu device:", gpu)

if str(gpu) == "xpu":
    print(torch.xpu.get_device_properties(0))

arr = torch.rand(1,2,5,5, device=gpu)
print("arr", arr.shape, arr.dtype, arr.device)

pts = -1+2*torch.rand(1,3,3,2, device=gpu)
print("pts", pts.shape, pts.dtype, pts.device)

out = F.grid_sample(arr, pts, align_corners=False)

print("out", out.shape, out.dtype, out.device)

