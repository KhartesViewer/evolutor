import sys
import argparse
import os
import pathlib
import math
import csv
from scipy import sparse

from st import ST

from PyQt5.QtWidgets import (
        QApplication,
        QGridLayout,
        QLabel,
        QMainWindow,
        QStatusBar,
        QVBoxLayout,
        QWidget,
        )
from PyQt5.QtCore import (
        QPoint,
        QSize,
        Qt,
        )
from PyQt5.QtGui import (
        QCursor,
        QImage,
        QPixmap,
        )

import cv2
import numpy as np
import cmap
# from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import solve_ivp
import nrrd

class MainWindow(QMainWindow):

    def __init__(self, app, parsed_args):
        super(MainWindow, self).__init__()
        self.app = app
        self.setMinimumSize(QSize(750,600))
        self.already_shown = False
        self.st = None
        grid = QGridLayout()
        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.viewer = ImageViewer(self)
        # self.setCentralWidget(self.viewer)
        grid.addWidget(self.viewer, 0, 0)
        self.viewer.setDefaults()

        no_cache = parsed_args.no_cache
        tifname = pathlib.Path(parsed_args.input_tif)
        nrrd_dir = parsed_args.cache_dir
        umbstr = parsed_args.umbilicus
        decimation = parsed_args.decimation
        self.viewer.decimation = decimation

        if nrrd_dir is None:
            nrrd_dir = tifname.parent
        else:
            nrrd_dir = pathlib.Path(nrrd_dir)
        nrrdname = nrrd_dir / (tifname.with_suffix(".nrrd")).name

        # python wind2d.py C:\Vesuvius\Projects\evol1\circle.tif --no_cache

        # python wind2d.py "C:\Vesuvius\scroll 1 masked xx000\02000.tif" --cache_dir C:\Vesuvius\Projects\evol1 --umbilicus 3960,2280

        # upath = pathlib.Path(r"C:\Vesuvius\Projects\evol1\umbilicus-scroll1a_zyx.txt")
        # uzyxs = self.readUmbilicus(upath)
        # print("umbs", len(uzyxs))
        # print(uzyxs[0], uzyxs[-1])        
        print("loading tif", tifname)
        # loadTIFF also sets default umbilicus location
        self.viewer.loadTIFF(tifname)
        self.st = ST(self.viewer.image)
        if umbstr is not None:
            words = umbstr.split(',')
            if len(words) != 2:
                print("Could not parse --umbilicus argument")
            else:
                self.viewer.umb = np.array((float(words[0]),float(words[1])))

        if no_cache:
            print("computing structural tensors")
            self.st.computeEigens()
        else:
            print("computing/loading structural tensors")
            self.st.loadOrCreateEigens(nrrdname)

    @staticmethod
    def readUmbilicus(fpath):
        uf = open(fpath, "r")
        ur = csv.reader(uf)
        zyxs = []
        for row in ur:
            zyxs.append([int(s) for s in row])
        zyxs.sort()
        return zyxs

    @staticmethod
    def getUmbilicusXY(zyxs):
        pass

    def setStatusText(self, txt):
        self.status_bar.showMessage(txt)

    def showEvent(self, e):
        # print("show event")
        if self.already_shown:
            return
        self.viewer.setDefaults()
        self.viewer.drawAll()
        self.already_shown = True

    def resizeEvent(self, e):
        # print("resize event")
        self.viewer.drawAll()

    def keyPressEvent(self, e):
        self.viewer.keyPressEvent(e)

class ImageViewer(QLabel):

    def __init__(self, main_window):
        super(ImageViewer, self).__init__()
        self.setMouseTracking(True)
        self.main_window = main_window
        self.image = None
        self.zoom = 1.
        self.center = (0,0)
        self.bar0 = (0,0)
        self.mouse_start_point = QPoint()
        self.center_start_point = None
        self.is_panning = False
        self.dip_bars_visible = True
        self.umb = None
        self.overlay_data = None
        self.sparse_u_cross_grad = None
        self.decimation = 1

    def mousePressEvent(self, e):
        if self.image is None:
            return
        if e.button() | Qt.LeftButton:
            # modifiers = QApplication.keyboardModifiers()
            wpos = e.localPos()
            wxy = (wpos.x(), wpos.y())
            ixy = self.wxyToIxy(wxy)

            self.mouse_start_point = wpos
            self.center_start_point = self.center
            # print("ixys", ixy)
            self.is_panning = True

    def mouseMoveEvent(self, e):
        if self.image is None:
            return
        wpos = e.localPos()
        wxy = (wpos.x(), wpos.y())
        ixy = self.wxyToIxy(wxy)
        self.setStatusTextFromMousePosition()
        if self.is_panning:
            # print(wpos, self.mouse_start_point)
            delta = wpos - self.mouse_start_point
            dx,dy = delta.x(), delta.y()
            z = self.zoom
            # cx, cy = self.center
            six,siy = self.center_start_point
            self.center = (six-dx/z, siy-dy/z)
            self.drawAll()

    def mouseReleaseEvent(self, e):
        if e.button() | Qt.LeftButton:
            self.mouse_start_point = QPoint()
            self.center_start_point = None
            self.is_panning = False

    def leaveEvent(self, e):
        if self.image is None:
            return
        self.main_window.setStatusText("")

    def wheelEvent(self, e):
        if self.image is None:
            return
        self.setStatusTextFromMousePosition()
        d = e.angleDelta().y()
        z = self.zoom
        z *= 1.001**d
        self.setZoom(z)
        self.drawAll()

    def keyPressEvent(self, e):
        '''
        if e.key() == Qt.Key_1:
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("pt 1 at",ixy)
            self.pt1 = ixy
            self.drawAll()
        elif e.key() == Qt.Key_2:
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("p2 1 at",ixy)
            self.pt2 = ixy
            self.drawAll()
            st = self.getST()
            if st is None:
                return
            for nudge in (0.,):
                y = None
                if self.pt1 is not None and self.pt2 is not None:
                    if False and nudge > 0:
                        y = st.interp2dLsqr(self.pt1, self.pt2, nudge)
                    else:
                        y = st.interp2dWHP(self.pt1, self.pt2)
                if y is not None:
                    print("ti2d", y.shape)
                    # pts = st.sparse_result(y, 0, 5)
                    pts = y
                    if pts is not None:
                        print("pts", pts.shape)
                        self.rays.append(pts)

            self.drawAll()
        elif e.key() == Qt.Key_T:
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("t at",ixy)
            st = self.getST()
            if st is None:
                return
            xnorm = st.vector_u_interpolator([ixy[::-1]])[0]
            for nudge in (0.,):
                for sign in (1,-1):
                    y = st.call_ivp(ixy+1.*nudge*xnorm, sign, tmax=1000)
                    if y is not None:
                        pts = y
                        if pts is not None:
                            self.rays.append(pts)

            if len(self.rays) > 0:
                self.drawAll()
        elif e.key() == Qt.Key_C:
            if len(self.rays) == 0:
                return
            self.rays = []
            self.drawAll()
        '''
        if e.key() == Qt.Key_W:
            # print("w at",ixy)
            # self.iterateWinding()
            self.solveWindingOneStep()
            self.drawAll()
        elif e.key() == Qt.Key_U:
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("umb at",ixy)
            self.umb = np.array(ixy)
            self.drawAll()

        elif e.key() == Qt.Key_V:
            self.dip_bars_visible = not self.dip_bars_visible
            self.drawAll()

    def getST(self):
        return self.main_window.st

    def mouseXy(self):
        pt = self.mapFromGlobal(QCursor.pos())
        return (pt.x(), pt.y())

    def setStatusTextFromMousePosition(self):
        wxy = self.mouseXy()
        ixy = self.wxyToIxy(wxy)
        self.setStatusText(ixy)

    def setStatusText(self, ixy):
        if self.image is None:
            return
        labels = ["X", "Y"]
        stxt = ""
        for i in (0,1):
            f = ixy[i]
            dtxt = "%.2f"%f
            if f < 0 or f > self.image.shape[1-i]-1:
                dtxt = "("+dtxt+")"
            stxt += "%s "%dtxt
        self.main_window.setStatusText(stxt)

    def ixyToWxy(self, ixy):
        ix,iy = ixy
        cx,cy = self.center
        z = self.zoom
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        wx = int(z*(ix-cx)) + wcx
        wy = int(z*(iy-cy)) + wcy
        return (wx,wy)

    def ixysToWxys(self, ixys):
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        c = self.center
        z = self.zoom
        dxys = ixys.copy()
        dxys -= c
        dxys *= z
        dxys = dxys.astype(np.int32)
        dxys[...,0] += wcx
        dxys[...,1] += wcy
        return dxys

    def wxyToIxy(self, wxy):
        wx,wy = wxy
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        dx,dy = wx-wcx, wy-wcy
        cx,cy = self.center
        z = self.zoom
        ix = cx + dx/z
        iy = cy + dy/z
        return (ix, iy)

    def wxysToIxys(self, wxys):
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2

        # dxys = wx-wcx, wy-wcy
        dxys = wxys.copy()
        dxys[...,0] -= wcx
        dxys[...,1] -= wcy
        cx,cy = self.center
        z = self.zoom
        ixys = np.zeros(wxys.shape)
        ixys[...,0] = cx + dxys[...,0]/z
        ixys[...,1] = cy + dxys[...,1]/z
        return ixys

    def setDefaults(self):
        if self.image is None:
            return
        ww = self.width()
        wh = self.height()
        # print("ww,wh",ww,wh)
        iw = self.image.shape[1]
        ih = self.image.shape[0]
        self.center = (iw//2, ih//2)
        zw = ww/iw
        zh = wh/ih
        zoom = min(zw, zh)
        self.setZoom(zoom)
        # self.bar0 = self.center
        print("center",self.center[0],self.center[1],"zoom",self.zoom)

    def setZoom(self, zoom):
        # TODO: set min, max zoom
        prev = self.zoom
        self.zoom = zoom
        if prev != 0:
            bw,bh = self.bar0
            cw,ch = self.center
            # print(self.bar0, self.center)
            bw -= cw
            bh -= ch
            bw /= zoom/prev
            bh /= zoom/prev
            self.bar0 = (bw+cw, bh+ch)

    # class function
    def rectIntersection(ra, rb):
        (ax1, ay1, ax2, ay2) = ra
        (bx1, by1, bx2, by2) = rb
        # print(ra, rb)
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        if (x1<x2) and (y1<y2):
            r = (x1, y1, x2, y2)
            # print(r)
            return r

    def loadTIFF(self, fname):
        try:
            image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED).astype(np.float64)
            image /= 65535.
        except Exception as e:
            print("Error while loading",fname,e)
            return
        self.image = image
        self.image_mtime = fname.stat().st_mtime
        self.setDefaults()
        self.umb = np.array((image.shape[1]/2, image.shape[0]/2))
        self.drawAll()

    @staticmethod
    def sparseUCrossGrad(weighted_u_vector, smoothing, umb):
        # img = self.image
        # st = self.main_window.st
        # uvecf = st.vector_u
        uvecf = weighted_u_vector
        # full number of rows, columns of image;
        # it is assumed that the image and st.vector_u
        # are the same size.
        # nrf, ncf = img.shape
        nrf, ncf = uvecf.shape[:2]
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, uvecf.shape[:2])
        # n1d is like n1df but shrunk by 1 in row and column directions
        n1d = n1df[:nr, :nc]
        # n1dfr is full in the row direction, but shrunk by 1 in column dir
        n1dfr = n1df[:, :nc]
        # n1dfc is full in the column direction, but shrunk by 1 in row dir
        n1dfc = n1df[:nr, :]

        umbpt = n1df[int(umb[1]), int(umb[0])]
        print("umbpt", umbpt)
        # flat array (size nrf-1 times ncf-1) where each element
        # contains a position in the original nrf by ncf array. 
        n1d_flat = n1d.flatten()
        n1dfr_flat = n1dfr.flatten()
        n1dfc_flat = n1dfc.flatten()

        # diag3 = np.stack((n1d_flat, n1d_flat, np.zeros(n1d_flat.shape)), axis=1).astype(np.float32)
        # float32 is not precise enough for carrying indices of large
        # flat matrices!
        diag3 = np.stack((n1d_flat, n1d_flat, np.zeros(n1d_flat.shape)), axis=1)
        diag3fr = np.stack((n1dfr_flat, n1dfr_flat, np.zeros(n1dfr_flat.shape)), axis=1)
        diag3fc = np.stack((n1dfc_flat, n1dfc_flat, np.zeros(n1dfc_flat.shape)), axis=1)
        print("diag3", diag3.shape, diag3.dtype)
        # print(diag3[0:10], diag3[997:1010])

        # clean up memory
        n1df_flat = None
        n1dfr_flat = None
        n1dfc_flat = None
        n1df = None
        n1d_flat = None
        n1dfr_flat = None
        n1dfc_flat = None
        n1d = None

        dx0g = diag3fr.copy()
        dx0g[:,2] = -1.

        dx1g = diag3fr.copy()
        dx1g[:,1] += 1
        dx1g[:,2] = 1.

        ddxg = np.concatenate((dx0g, dx1g), axis=0)
        # print("ddx", ddx.shape, ddx.dtype)

        # clean up memory
        dx0g = None
        dx1g = None

        dy0g = diag3fc.copy()
        dy0g[:,2] = -1.

        dy1g = diag3fc.copy()
        dy1g[:,1] += ncf
        dy1g[:,2] = 1.

        ddyg = np.concatenate((dy0g, dy1g), axis=0)

        # clean up memory
        dy0g = None
        dy1g = None

        ddxg[:,0] *= 2
        ddyg[:,0] *= 2
        ddyg[:,0] += 1

        grad = np.concatenate((ddxg, ddyg), axis=0)
        print("grad", grad.shape, grad.min(axis=0), grad.max(axis=0), grad.dtype)
        sparse_grad = sparse.coo_array((grad[:,2], (grad[:,0], grad[:,1])), shape=(2*nrf*ncf, nrf*ncf))


        uvec_flat = uvecf[:nr, :nc].reshape(-1, 2)
        print("uvec_flat", uvec_flat.shape)

        dx0 = diag3.copy()
        dx0[:,2] = -uvec_flat[:,1]
        # dx0[:,2] = 1

        dx1 = diag3.copy()
        dx1[:,1] += 1
        dx1[:,2] = uvec_flat[:,1]
        # dx1[:,2] = -1

        ddx = np.concatenate((dx0, dx1), axis=0)
        print("ddx", ddx.shape, ddx.dtype)

        # clean up memory
        dx0 = None
        dx1 = None

        '''
        dx0[:,2] = -1
        dx1[:,2] = 1
        ddxg = np.concatenate((dx0, dx1), axis=0)
        '''

        '''
        # clean up memory
        dx0 = None
        dx1 = None
        '''

        dy0 = diag3.copy()
        dy0[:,2] = uvec_flat[:,0]
        # dy0[:,2] = -1

        dy1 = diag3.copy()
        dy1[:,1] += ncf
        dy1[:,2] = -uvec_flat[:,0]
        # dy1[:,2] = 1

        ddy = np.concatenate((dy0, dy1), axis=0)
        print("ddy", ddy.shape, ddy.dtype)

        '''
        dy0[:,2] = -1
        dy1[:,2] = 1
        ddyg = np.concatenate((dy0, dy1), axis=0)
        '''

        # clean up memory
        dy0 = None
        dy1 = None

        '''
        ddxg[:,0] *= 2
        ddyg[:,0] *= 2
        ddyg[:,0] += 1

        grad = np.concatenate((ddxg, ddyg), axis=0)
        print("grad", grad.shape, grad.min(axis=0), grad.max(axis=0), grad.dtype)
        sparse_grad = sparse.coo_array((grad[:,2], (grad[:,0], grad[:,1])), shape=(2*nrf*ncf, nrf*ncf))
        '''

        # sparse_grad seems to carry views of the grad array, so
        # subsequent changes to grad will propagate back into
        # sparse_grad.  We don't want this, so make a copy to work on
        grad = grad.copy()
        grad[:,0] += ncf*nrf
        print("ddx,ddy,grad", ddx.max(axis=0), ddy.max(axis=0), grad.min(axis=0), grad.max(axis=0))
        grad[:,2] *= smoothing

        umbzero = np.array([[3*ncf*nrf, umbpt, 1.]])

        uxg = np.concatenate((ddx, ddy, grad, umbzero), axis=0)
        print("uxg", uxg.shape, uxg.dtype, uxg[:,0].max(), uxg[:,1].max())
        ddx = None
        ddy = None
        grad = None
        sparse_uxg = sparse.coo_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(3*nrf*ncf+1, nrf*ncf))
        uxg = None
        return sparse_grad.tocsc(), sparse_uxg.tocsc()

    def solveWindingLinear(self, basew, smoothing):
        '''
        dwdx = np.diff(basew, axis=1)[:-1,:]
        dwdy = np.diff(basew, axis=0)[:,:-1]
        print(dwdx.shape, dwdy.shape)
        dxylen = np.sqrt(dwdx*dwdx+dwdy*dwdy)
        print("dxylen", dxylen.min(), dxylen.max())
        dxylen[dxylen==0] = .001
        dwdx /= dxylen
        dwdy /= dxylen
        dwd = np.stack((dwdx,dwdy), axis=2)
        print(dwd.shape)
        # return .5*(dwdy+1.0)
        st = self.main_window.st
        stnorm = st.vector_u[:-1,:-1]
        coh = st.coherence[:-1,:-1]
        cross = np.cross(stnorm, dwd)
        # return .5*(coh*cross+1)
        # out = .5*(cross+1)
        out = .5*(coh*cross+1)
        #print(out.max(), out.min(), out[1,1])
        return out
        '''

        st = self.main_window.st
        decimation = self.decimation
        print("decimation", decimation)
        # u = st.vector_u[:-1,:-1]
        # c = st.coherence[:-1,:-1]
        # rows, cols = np.mgrid[:basew.shape[0]-1, :basew.shape[1]-1]

        if self.sparse_u_cross_grad is None:
            vecu = st.vector_u
            coh = st.coherence[:,:,np.newaxis]
            wvecu = coh*vecu
            if decimation > 1:
                wvecu = wvecu[::decimation, ::decimation, :]
                basew = basew.copy()[::decimation, ::decimation]
            self.sparse_grad, self.sparse_u_cross_grad = self.sparseUCrossGrad(wvecu, smoothing, np.array(self.umb)/decimation)
        print("is sparse", 
              sparse.issparse(self.sparse_grad),
              sparse.issparse(self.sparse_u_cross_grad))
        

        A = self.sparse_u_cross_grad
        print("A", A.shape, A.dtype)
        # nn = 120
        # a = A[nn*250+100:nn*250+102]
        # print("a", a.shape, a.dtype, a[a!=0], np.argwhere(a))

        b = -self.sparse_u_cross_grad @ basew.flatten()
        b[basew.size:] = 0.

        At = A.transpose()
        AtA = At @ A
        print("AtA", AtA.shape, sparse.issparse(AtA))
        # ata = AtA[110*250+100:110*250+104]
        # print("ata", ata.shape, ata.dtype, ata[ata!=0], np.argwhere(ata))
        ssum = AtA.sum(axis=0)
        print("ssum", np.argwhere(np.abs(ssum)>1.e-10))
        # print(ssum[:10])
        asum = np.abs(AtA).sum(axis=0)
        print("asum", np.argwhere(asum==0))
        Atb = At @ b
        print("Atb", Atb.shape, sparse.issparse(Atb))

        lu = sparse.linalg.splu(AtA.tocsc())
        x = lu.solve(Atb)
        print("x", x.shape, x.dtype, x.min(), x.max())
        out = x.reshape(basew.shape)
        out += basew
        print("out", out.shape, out.min(), out.max())
        maxrad = np.sqrt((self.umb*self.umb).sum())
        print("maxrad", maxrad)
        maxrad = 1500
        out = out/maxrad
        if decimation > 1:
            out = cv2.resize(out, (vecu.shape[1], vecu.shape[0]), interpolation=cv2.INTER_AREA)
        return out

        '''
        uxgbf = (self.sparse_u_cross_grad @ basew.flatten())[:basew.size]
        print("uxgbf", uxgbf.shape, sparse.issparse(uxgbf), uxgbf.min(), uxgbf.max())
        uxgb = uxgbf.reshape(basew.shape)
        # print(uxgb[450,495:500])
        gbf = self.sparse_grad @ basew.flatten()
        gb = gbf.reshape((basew.shape[0], basew.shape[1], 2))
        gblen = np.sqrt(gb[:,:,0]*gb[:,:,0]+gb[:,:,1]*gb[:,:,1])
        print("gblen", gblen.min(), gblen.max())
        gblen[gblen==0] = .001
        # coh = st.coherence
        uxgbn = uxgb/gblen
        print("uxgbn", uxgbn.min(), uxgbn.max())
        # out = .5*(coh*uxgbn+1)
        out = .5*(uxgbn+1)
        if decimation > 1:
            # zslc = cv2.resize(data[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
            out = cv2.resize(out, (vecu.shape[1], vecu.shape[0]), interpolation=cv2.INTER_AREA)
        return out
        '''

    def solveWindingOneStep(self):
        im = self.image
        if im is None:
            return
        umb = self.umb
        # startw = np.zeros(im.shape, np.float64)
        # basew = np.zeros(im.shape, np.float64)
        # ogrid is an open grid, but it turns out to work
        # in the code below because of broadcasting.
        # Better to be explicit by using mgrid, however.
        # iys, ixs = np.ogrid[:im.shape[0], :im.shape[1]]
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
        rad = np.sqrt(radsq)
        print("rad", rad.shape)

        smoothing = .01
        # smoothing = 0.
        nextw = self.solveWindingLinear(rad, smoothing)
        self.overlay_data = nextw

    colormaps = {
            "gray": "matlab:gray",
            "viridis": "bids:viridis",
            "bwr": "matplotlib:bwr",
            "cool": "matlab:cool",
            "bmr_3c": "chrisluts:bmr_3c",
            "rainbow": "gnuplot:rainbow",
            }

    # input: 2D float array, range 0.0 to 1.0
    # output: RGB array, uint8, with colors determined by the
    # colormap and alpha, zoomed in based on the current
    # window size, center, and zoom factor
    def dataToZoomedRGB(self, data, alpha=1., colormap="gray", interpolation="linear"):
        if colormap in self.colormaps:
            colormap = self.colormaps[colormap]
        cm = cmap.Colormap(colormap, interpolation=interpolation)

        iw = data.shape[1]
        ih = data.shape[0]
        z = self.zoom
        # zoomed image width, height:
        ziw = max(int(z*iw), 1)
        zih = max(int(z*ih), 1)
        # viewing window width, height:
        ww = self.width()
        wh = self.height()
        # print("di ww,wh",ww,wh)
        # viewing window half width
        whw = ww//2
        whh = wh//2
        cx,cy = self.center

        # Pasting zoomed data slice into viewing-area array, taking
        # panning into account.
        # Need to calculate the interesection
        # of the two rectangles: 1) the panned and zoomed slice, and 2) the
        # viewing window, before pasting
        ax1 = int(whw-z*cx)
        ay1 = int(whh-z*cy)
        ax2 = ax1+ziw
        ay2 = ay1+zih
        bx1 = 0
        by1 = 0
        bx2 = ww
        by2 = wh
        ri = ImageViewer.rectIntersection((ax1,ay1,ax2,ay2), (bx1,by1,bx2,by2))
        outrgb = np.zeros((wh,ww,3), dtype=np.uint8)
        if ri is not None:
            (x1,y1,x2,y2) = ri
            # zoomed data slice
            x1s = int((x1-ax1)/z)
            y1s = int((y1-ay1)/z)
            x2s = int((x2-ax1)/z)
            y2s = int((y2-ay1)/z)
            # print(sw,sh,ww,wh)
            # print(x1,y1,x2,y2)
            # print(x1s,y1s,x2s,y2s)
            zslc = cv2.resize(data[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
            cslc = cm(zslc)
            # print("cslc", colormap, cslc[0,0], zslc[0,0])
            # print(zslc.min(), zslc.max())
            # print(cslc.shape, cslc.dtype, cslc.min(), cslc.max())
            outrgb[y1:y2, x1:x2, :] = (255*cslc[:,:,:3]*alpha).astype(np.uint8)
        return outrgb

    def drawAll(self):
        if self.image is None:
            return
        main_alpha = .4
        total_alpha = .8

        # print(self.image.shape, self.image.min(), self.image.max())
        outrgb = self.dataToZoomedRGB(self.image, alpha=main_alpha)
        st = self.main_window.st
        other_data = None
        if st is not None:
            other_data = st.coherence
            # print(other_data.shape, other_data.min(), other_data.max())
        other_data = self.overlay_data
        if other_data is not None:
            colormap = "viridis"
            colormap = "bmr_3c"
            colormap = "bwr"
            colormap = "colorbrewer:Spectral_11"
            colormap = "colorbrewer:Set3_12"
            # watch out for the sqrt!
            # outrgb += self.dataToZoomedRGB(np.sqrt(other_data), alpha=total_alpha-main_alpha, colormap=colormap, interpolation="nearest")
            outrgb += self.dataToZoomedRGB(other_data, alpha=total_alpha-main_alpha, colormap=colormap, interpolation="linear")

        ww = self.width()
        wh = self.height()

        if st is not None and self.dip_bars_visible:
            dh = 15
            w0i,h0i = self.wxyToIxy((0,0))
            w0i -= self.bar0[0]
            h0i -= self.bar0[1]
            dhi = 2*dh/self.zoom
            w0i = int(math.floor(w0i/dhi))*dhi
            h0i = int(math.floor(h0i/dhi))*dhi
            w0i += self.bar0[0]
            h0i += self.bar0[1]
            w0,h0 = self.ixyToWxy((w0i,h0i))
            dpw = np.mgrid[h0:wh:2*dh, w0:ww:2*dh].transpose(1,2,0)
            # switch from y,x to x,y coordinates
            dpw = dpw[:,:,::-1]
            # print ("dpw", dpw.shape, dpw.dtype, dpw[0,5])
            dpi = self.wxysToIxys(dpw)
            # interpolators expect y,x ordering
            dpir = dpi[:,:,::-1]
            # print ("dpi", dpi.shape, dpi.dtype, dpi[0,5])
            uvs = st.vector_u_interpolator(dpir)
            vvs = st.vector_v_interpolator(dpir)
            # print("vvs", vvs.shape, vvs.dtype, vvs[0,5])
            # coherence = st.coherence_interpolator(dpir)
            coherence = st.linearity_interpolator(dpir)
            # testing
            # coherence[:] = .5
            # print("coherence", coherence.shape, coherence.dtype, coherence[0,5])
            linelen = 25.

            lvecs = linelen*vvs*coherence[:,:,np.newaxis]
            x0 = dpw
            x1 = dpw+lvecs

            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            # cv2.polylines(outrgb, lines, False, (255,255,0), 1)

            lvecs = linelen*uvs*coherence[:,:,np.newaxis]

            x1 = dpw+lvecs
            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            # cv2.polylines(outrgb, lines, False, (0,255,0), 1)

            xm = dpw-.5*lvecs
            xp = dpw+.5*lvecs
            lines = np.concatenate((xm,xp), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            cv2.polylines(outrgb, lines, False, (0,255,0), 1)

            points = dpw.reshape(-1,1,1,2).astype(np.int32)
            cv2.polylines(outrgb, points, True, (0,255,255), 3)

        if self.umb is not None:
            wumb = self.ixyToWxy(self.umb)
            cv2.circle(outrgb, wumb, 3, (255,0,255), -1)

        '''
        for i,ray in enumerate(self.rays):
            points = self.ixysToWxys(ray)
            points = points.reshape(-1,1,1,2)
            colors = (
                    (255,255,255), (255,0,255), 
                    (255,255,224), (255,0,224), 
                    (255,255,192), (255,0,192), 
                    (255,255,160), (255,0,160), 
                    (255,255,128), (255,0,128), 
                    )
            colors = (
                    (255,0,255), 
                    (255,64,192), 
                    (255,128,128), 
                    (255,192,64), 
                    (255,255,0), 
                    )
            color = colors[i%len(colors)]

            cv2.polylines(outrgb, points, True, color, 4)
            cv2.circle(outrgb, points[0,0,0], 3, (255,0,255), -1)
        if self.pt1 is not None:
            wpt1 = self.ixyToWxy(self.pt1)
            cv2.circle(outrgb, wpt1, 3, (255,0,255), -1)
        if self.pt2 is not None:
            wpt2 = self.ixyToWxy(self.pt2)
            cv2.circle(outrgb, wpt2, 3, (255,0,255), -1)
        '''

        bytesperline = 3*outrgb.shape[1]
        # print(outrgb.shape, outrgb.dtype)
        qimg = QImage(outrgb, outrgb.shape[1], outrgb.shape[0],
                      bytesperline, QImage.Format_RGB888)
        # print("created qimg")
        pixmap = QPixmap.fromImage(qimg)
        # print("created pixmap")
        self.setPixmap(pixmap)
        # print("set pixmap")

class Tinter():

    def __init__(self, app, parsed_args):
        window = MainWindow(app, parsed_args)
        self.app = app
        self.window = window
        window.show()

# From https://stackoverflow.com/questions/11713006/elegant-command-line-argument-parsing-for-pyqt

def process_cl_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Test determining winding numbers using structural tensors")
    parser.add_argument("input_tif",
                        help="input tiff slice")
    parser.add_argument("--cache_dir",
                        default=None,
                        help="directory where the cache of the structural tensor data is or will be stored; if not given, directory of input tiff slice is used")
    parser.add_argument("--no_cache",
                        action="store_true",
                        help="Don't use cached structural tensors")
    parser.add_argument("--umbilicus",
                        default=None,
                        help="umbilicus location in x,y, for example 3960,2280")
    parser.add_argument("--decimation",
                        type=int,
                        default=1,
                        help="decimation factor (default is no decimation)")

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args


if __name__ == '__main__':
    parsed_args, unparsed_args = process_cl_args()
    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)

    tinter = Tinter(app, parsed_args)
    sys.exit(app.exec())
