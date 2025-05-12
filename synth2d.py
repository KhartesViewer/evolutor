from pathlib import Path
import numpy as np
import cv2

outdir = Path(r"C:\Vesuvius\Projects\evol1")

ofile = outdir / "circle.tif"

h = 1000
w = 1000
omega = 1.
ecc = .5
# Rotated 90 degrees:
# omega = .5
# ecc = 2.
minrad = 80
maxrad = 400

cx = w/2 + 49
cy = h/2 - 37
print("output file", ofile)
print("umbilicus",cx,cy)

xys = np.mgrid[:h, :w].astype(np.float32)
xys[0] -= cy
xys[1] -= cx

# Prevent numerical problems due to
# excessive symmetry
xys[0] += .0001
xys[1] += .0002

rs = np.sqrt((xys*xys).sum(axis=0))

exys = xys
exys[1] *= ecc
ers = np.sqrt((exys*exys).sum(axis=0))

fimg = np.sin(omega*ers)
fimg += .5*np.sin(2*omega*ers)
fimg /= 1.5
fimg[rs < 10] = 0.
fimg[rs > maxrad] = 0.

fimg += .0001*np.random.rand(*fimg.shape)

iimg = (32767*(fimg+1.)).astype(np.uint16)
cv2.imwrite(str(ofile), iimg)

