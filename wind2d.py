import sys
import argparse
import os
import pathlib
import math
import csv
from scipy import sparse
import nrrd

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
        grid.addWidget(self.viewer, 0, 0)
        self.viewer.setDefaults()

        no_cache = parsed_args.no_cache
        self.viewer.no_cache = no_cache
        tifname = pathlib.Path(parsed_args.input_tif)
        cache_dir = parsed_args.cache_dir
        umbstr = parsed_args.umbilicus
        window_width = parsed_args.window
        decimation = parsed_args.decimation
        self.viewer.decimation = decimation
        self.viewer.overlay_colormap = parsed_args.colormap
        self.viewer.overlay_interpolation = parsed_args.interpolation
        maxrad = parsed_args.maxrad

        if cache_dir is None:
            cache_dir = tifname.parent
        else:
            cache_dir = pathlib.Path(cache_dir)
        cache_file_base = cache_dir / tifname.stem
        self.viewer.cache_dir = cache_dir
        self.viewer.cache_file_base = cache_file_base
        # nrrdname = cache_dir / (tifname.with_suffix(".nrrd")).name

        # python wind2d.py C:\Vesuvius\Projects\evol1\circle.tif --no_cache

        # python wind2d.py "C:\Vesuvius\scroll 1 masked xx000\02000.tif" --cache_dir C:\Vesuvius\Projects\evol1 --umbilicus 3960,2280 --decimation 8 --colormap tab20 --interpolation nearest --maxrad 1800
        # 0: 3960,2290
        # 90: 5608,3960
        # 180: 4136,5608

        #  python wind2d.py "C:\Vesuvius\scroll 1 full xx000\03000f.tif" --cache_dir C:\Vesuvius\Projects\evol1 --umbilicus 3670,2200 --decimation 8 --colormap tab20 --interpolation nearest --maxrad 1400

        # upath = pathlib.Path(r"C:\Vesuvius\Projects\evol1\umbilicus-scroll1a_zyx.txt")
        # uzyxs = self.readUmbilicus(upath)
        # print("umbs", len(uzyxs))
        # print(uzyxs[0], uzyxs[-1])        
        print("loading tif", tifname)
        # loadTIFF also sets default umbilicus location
        self.viewer.loadTIFF(tifname)
        if umbstr is not None:
            words = umbstr.split(',')
            if len(words) != 2:
                print("Could not parse --umbilicus argument")
            else:
                self.viewer.umb = np.array((float(words[0]),float(words[1])))
        umb = self.viewer.umb
        self.viewer.umb_maxrad = np.sqrt((umb*umb).sum())
        if maxrad is None:
            maxrad = self.viewer.umb_maxrad
        self.viewer.overlay_maxrad = maxrad

        self.viewer.window_width = window_width
        if window_width is not None:
            hw = window_width // 2
            ux = int(umb[0])
            uy = int(umb[1])
            self.viewer.image = self.viewer.image[uy-hw:uy+hw, ux-hw:ux+hw]
            umb[0] = hw
            umb[1] = hw

        self.st = ST(self.viewer.image)

        part = "_e.nrrd"
        if decimation is not None and decimation > 1:
            part = "_d%d%s"%(decimation, part)
        if window_width is not None:
            part = "_w%d%s"%(window_width, part)
        nrrdname = cache_file_base.with_name(cache_file_base.name + part)
        if no_cache:
            print("computing structural tensors")
            self.st.computeEigens()
        else:
            print("computing/loading structural tensors")
            self.st.loadOrCreateEigens(nrrdname)

        mask = self.viewer.createMask()

        self.viewer.setOverlayDefaults()
        self.viewer.saveCurrentOverlay()
        self.viewer.overlay_name = "coherence"
        self.viewer.overlay_data = self.st.coherence.copy()
        self.viewer.overlay_data *= mask
        self.viewer.overlay_colormap = "viridis"
        self.viewer.overlay_interpolation = "linear"
        self.viewer.overlay_maxrad = 1.0
        self.viewer.saveCurrentOverlay()
        self.viewer.getNextOverlay()

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

class Overlay():
    def __init__(self, name, data, maxrad, colormap="viridis", interpolation="linear"):
        self.name = name
        self.data = data
        self.colormap = colormap
        self.interpolation = interpolation
        self.maxrad = maxrad

    @staticmethod
    def createFromOverlay(overlay):
        no = Overlay(overlay.name, overlay.data, overlay.maxrad, overlay.colormap, overlay.interpolation)
        return no

    @staticmethod
    def findIndexByName(overlays, name):
        for i,item in enumerate(overlays):
            if item.name == name:
                return i
        return -1

    @staticmethod
    def findItemByName(overlays, name):
        index = Overlay.findIndexByName(overlays, name)
        if index < 0:
            return None
        return overlays[index]

    @staticmethod
    def findNextItem(overlays, cur_name):
        index = Overlay.findIndexByName(overlays, cur_name)
        if index < 0:
            return overlays[0]
        return overlays[(index+1)%len(overlays)]

    @staticmethod
    def findPrevItem(overlays, cur_name):
        index = Overlay.findIndexByName(overlays, cur_name)
        if index < 0:
            return overlays[0]
        return overlays[(index+len(overlays)-1)%len(overlays)]



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
        self.overlays = []
        self.overlay_data = None
        self.overlay_name = ""
        # self.sparse_u_cross_grad = None
        self.decimation = 1
        self.overlay_colormap = "viridis"
        self.overlay_interpolation = "linear"
        self.overlay_maxrad = None
        self.overlay_defaults = None

    def setOverlayDefaults(self):
        self.overlay_defaults = Overlay("", None, self.overlay_maxrad, self.overlay_colormap, self.overlay_interpolation)

    def makeOverlayCurrent(self, overlay):
        self.saveCurrentOverlay()
        self.overlay_data = overlay.data
        self.overlay_name = overlay.name
        self.overlay_colormap = overlay.colormap
        self.overlay_interpolation = overlay.interpolation
        self.overlay_maxrad = overlay.maxrad

    def setOverlayByName(self, name):
        o = Overlay.findItemByName(self.overlays, name)
        if o is None:
            return
        self.makeOverlayCurrent(o)


    def saveCurrentOverlay(self):
        name = self.overlay_name
        no = Overlay(name, self.overlay_data, self.overlay_maxrad, self.overlay_colormap, self.overlay_interpolation)
        index = Overlay.findIndexByName(self.overlays, name)
        if index < 0:
            self.overlays.append(no)
        else:
            self.overlays[index] = no

    def getNextOverlay(self):
        name = self.overlay_name

        no = Overlay.findNextItem(self.overlays, name)
        self.makeOverlayCurrent(no)

    def getPrevOverlay(self):
        name = self.overlay_name

        no = Overlay.findPrevItem(self.overlays, name)
        self.makeOverlayCurrent(no)

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

    colormaps = {
            "gray": "matlab:gray",
            "viridis": "bids:viridis",
            "bwr": "matplotlib:bwr",
            "cool": "matlab:cool",
            "bmr_3c": "chrisluts:bmr_3c",
            "rainbow": "gnuplot:rainbow",
            "spec11": "colorbrewer:Spectral_11",
            "set12": "colorbrewer:Set3_12",
            "tab20": "seaborn:tab20",
            "hsv": "matlab:hsv",
            }

    @staticmethod
    def nextColormapName(cur):
        cms = ImageViewer.colormaps
        keys = list(cms.keys())
        index = keys.index(cur)
        index = (index+1) % len(keys)
        return keys[index]

    @staticmethod
    def prevColormapName(cur):
        cms = ImageViewer.colormaps
        keys = list(cms.keys())
        index = keys.index(cur)
        index = (index+len(keys)-1) % len(keys)
        return keys[index]

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
            self.solveWindingOneStep()
            self.drawAll()
        elif e.key() == Qt.Key_C:
            if e.modifiers() & Qt.ShiftModifier:
                self.overlay_colormap = self.prevColormapName(self.overlay_colormap)
            else:
                self.overlay_colormap = self.nextColormapName(self.overlay_colormap)
            self.drawAll()
        elif e.key() == Qt.Key_O:
            if e.modifiers() & Qt.ShiftModifier:
                self.getPrevOverlay()
            else:
                self.getNextOverlay()
            self.drawAll()
        elif e.key() == Qt.Key_Up:
            self.getPrevOverlay()
            self.drawAll()
        elif e.key() == Qt.Key_Down:
            self.getNextOverlay()
            self.drawAll()
        elif e.key() == Qt.Key_I:
            if self.overlay_interpolation == "linear":
                self.overlay_interpolation = "nearest"
            else:
                self.overlay_interpolation = "linear"
            self.drawAll()
        elif e.key() == Qt.Key_R:
            if e.modifiers() & Qt.ControlModifier:
                delta = 10
            elif e.modifiers() & Qt.AltModifier:
                delta = 1
            else:
                delta = 100
            if e.modifiers() & Qt.ShiftModifier:
                # print("Cap R")
                self.overlay_maxrad += delta
            elif self.overlay_maxrad > delta:
                self.overlay_maxrad -= delta
                #print("r")
            print("maxrad", self.overlay_maxrad)
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
        elif e.key() == Qt.Key_Q:
            print("Exiting")
            exit()
        elif e.key() == Qt.Key_E:
            self.createEThetaArray()

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
        inside = True
        for i in (0,1):
            f = ixy[i]
            dtxt = "%.2f"%f
            if f < 0 or f > self.image.shape[1-i]-1:
                dtxt = "("+dtxt+")"
                inside = False
            stxt += "%s "%dtxt
        aixy = np.array(ixy)
        aumb = np.array(self.umb)
        da = aixy-aumb
        rad = np.sqrt((da*da).sum())
        stxt += "r=%.2f "%rad
        if inside:
            iix,iiy = int(round(ixy[0])), int(round(ixy[1]))
            imi = self.image[iiy, iix]
            stxt += "%.2f "%imi
            if self.overlay_data is not None:
                imi = self.overlay_data[iiy, iix]
                stxt += "%s=%.2f "%(self.overlay_name, imi)
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
        print("center",self.center[0],self.center[1],"zoom",self.zoom)

    def setZoom(self, zoom):
        # TODO: set min, max zoom
        prev = self.zoom
        self.zoom = zoom
        if prev != 0:
            bw,bh = self.bar0
            cw,ch = self.center
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

    # set is_cross True if op is cross product, False if
    # op is dot product
    @staticmethod
    def sparseVecOpGrad(vec2d, is_cross):
        # full number of rows, columns of image;
        # it is assumed that the image and vec2d
        # are the same size, except each vec2d element
        # has 2 components.
        nrf, ncf = vec2d.shape[:2]
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, vec2d.shape[:2])
        # No immediate effect, since n1df is a view of n1df_flat
        n1df_flat = None
        # n1d is like n1df but shrunk by 1 in row and column directions
        n1d = n1df[:nr, :nc]
        # No immediate effect, since n1d is a view of n1df
        n1df = None
        # flat array (size nrf-1 times ncf-1) where each element
        # contains a position in the original nrf by ncf array. 
        n1d_flat = n1d.flatten()
        # No immediate effect, since n1d_flat is a view of n1d
        n1d = None
        # diag3 is the diagonal matrix of n1d_flat, in 3-column sparse format.
        # float32 is not precise enough for carrying indices of large
        # flat matrices, so use default (float64)
        diag3 = np.stack((n1d_flat, n1d_flat, np.zeros(n1d_flat.shape)), axis=1)
        print("diag3", diag3.shape, diag3.dtype)
        # clean up memory
        n1d_flat = None

        vec2d_flat = vec2d[:nr, :nc].reshape(-1, 2)
        print("vec2d_flat", vec2d_flat.shape)

        dx0 = diag3.copy()

        dx1 = diag3.copy()
        dx1[:,1] += 1
        if is_cross:
            dx0[:,2] = vec2d_flat[:,1]
            dx1[:,2] = -vec2d_flat[:,1]
        else:
            dx0[:,2] = -vec2d_flat[:,0]
            dx1[:,2] = vec2d_flat[:,0]

        ddx = np.concatenate((dx0, dx1), axis=0)
        print("ddx", ddx.shape, ddx.dtype)

        # clean up memory
        dx0 = None
        dx1 = None

        dy0 = diag3.copy()

        dy1 = diag3.copy()
        dy1[:,1] += ncf
        if is_cross:
            dy0[:,2] = -vec2d_flat[:,0]
            dy1[:,2] = vec2d_flat[:,0]
            pass
        else:
            dy0[:,2] = -vec2d_flat[:,1]
            dy1[:,2] = vec2d_flat[:,1]

        ddy = np.concatenate((dy0, dy1), axis=0)
        print("ddy", ddy.shape, ddy.dtype)

        # clean up memory
        dy0 = None
        dy1 = None

        print("ddx,ddy", ddx.max(axis=0), ddy.max(axis=0))

        uxg = np.concatenate((ddx, ddy), axis=0)
        print("uxg", uxg.shape, uxg.dtype, uxg[:,0].max(), uxg[:,1].max())
        ddx = None
        ddy = None
        sparse_uxg = sparse.coo_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(nrf*ncf, nrf*ncf))
        # sparse_uxg = sparse.csc_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(nrf*ncf, nrf*ncf))
        print("sparse_uxg", sparse_uxg.shape, sparse_uxg.dtype)
        return sparse_uxg


    @staticmethod
    def sparseDiagonal(shape):
        nrf, ncf = shape
        ix = np.arange(nrf*ncf)
        ones = np.full((nrf*ncf), 1.)
        # diag3 = np.stack((ix, ix, ones()), axis=1)
        sparse_diag = sparse.coo_array((ones, (ix, ix)), shape=(nrf*ncf, nrf*ncf))
        return sparse_diag

    @staticmethod
    def sparseGrad(shape, multiplier=None, interleave=True):
        # full number of rows, columns of image
        nrf, ncf = shape
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, shape)
        n1df_flat = None

        # n1dfr is full in the row direction, but shrunk by 1 in column dir
        n1dfr = n1df[:, :nc]
        # n1dfc is full in the column direction, but shrunk by 1 in row dir
        n1dfc = n1df[:nr, :]
        n1df = None
        n1dfr_flat = n1dfr.flatten()
        n1dfr = None
        n1dfc_flat = n1dfc.flatten()
        n1dfc = None

        mfr = None
        mfc = None
        if multiplier is not None:
            mfr = multiplier.flatten()[n1dfr_flat]
            mfc = multiplier.flatten()[n1dfc_flat]

        # float32 is not precise enough for carrying indices of large
        # flat matrices, so use default (float64)
        diag3fr = np.stack((n1dfr_flat, n1dfr_flat, np.zeros(n1dfr_flat.shape)), axis=1)
        n1dfr_flat = None
        diag3fc = np.stack((n1dfc_flat, n1dfc_flat, np.zeros(n1dfc_flat.shape)), axis=1)
        n1dfc_flat = None

        dx0g = diag3fr.copy()
        if mfr is not None:
            dx0g[:,2] = -mfr
        else:
            dx0g[:,2] = -1.

        dx1g = diag3fr.copy()
        dx1g[:,1] += 1
        if mfr is not None:
            dx1g[:,2] = mfr
        else:
            dx1g[:,2] = 1.

        ddxg = np.concatenate((dx0g, dx1g), axis=0)
        # print("ddx", ddx.shape, ddx.dtype)

        # clean up memory
        diag3fr = None
        dx0g = None
        dx1g = None

        dy0g = diag3fc.copy()
        if mfc is not None:
            dy0g[:,2] = -mfc
        else:
            dy0g[:,2] = -1.

        dy1g = diag3fc.copy()
        dy1g[:,1] += ncf
        if mfc is not None:
            dy1g[:,2] = mfc
        else:
            dy1g[:,2] = 1.

        ddyg = np.concatenate((dy0g, dy1g), axis=0)

        # clean up memory
        diag3fc = None
        dy0g = None
        dy1g = None

        if interleave:
            ddxg[:,0] *= 2
            ddyg[:,0] *= 2
            ddyg[:,0] += 1

            grad = np.concatenate((ddxg, ddyg), axis=0)
            print("grad", grad.shape, grad.min(axis=0), grad.max(axis=0), grad.dtype)
            sparse_grad = sparse.coo_array((grad[:,2], (grad[:,0], grad[:,1])), shape=(2*nrf*ncf, nrf*ncf))
            return sparse_grad
        else:
            sparse_grad_x = sparse.coo_array((ddxg[:,2], (ddxg[:,0], ddxg[:,1])), shape=(nrf*ncf, nrf*ncf))
            sparse_grad_y = sparse.coo_array((ddyg[:,2], (ddyg[:,0], ddyg[:,1])), shape=(nrf*ncf, nrf*ncf))
            return sparse_grad_x, sparse_grad_y

    @staticmethod
    def sparseUmbilical(shape, umb):
        nrf, ncf = shape
        nr = nrf-1
        nc = ncf-1
        n1df_flat = np.arange(nrf*ncf)
        # nrf x ncf array where each element is a number
        # representing the element's position in the array
        n1df = np.reshape(n1df_flat, shape)
        umbpt = n1df[int(umb[1]), int(umb[0])]
        umbzero = np.array([[0, umbpt, 1.]])
        sparse_umb = sparse.coo_array((umbzero[:,2], (umbzero[:,0], umbzero[:,1])), shape=(nrf*ncf, nrf*ncf))
        return sparse_umb

    def solveRadius0(self, basew, smoothing_weight):
        st = self.main_window.st
        decimation = self.decimation
        print("decimation", decimation)

        vecu = st.vector_u
        coh = st.coherence[:,:,np.newaxis]
        wvecu = coh*vecu
        if decimation > 1:
            wvecu = wvecu[::decimation, ::decimation, :]
            basew = basew.copy()[::decimation, ::decimation]
        shape = wvecu.shape[:2]
        sparse_uxg = ImageViewer.sparseVecOpGrad(wvecu, is_cross=True)
        sparse_grad = ImageViewer.sparseGrad(shape)
        sparse_umb = ImageViewer.sparseUmbilical(shape, np.array(self.umb)/decimation)
        sparse_u_cross_grad = sparse.vstack((sparse_uxg, smoothing_weight*sparse_grad, sparse_umb))

        A = sparse_u_cross_grad
        print("A", A.shape, A.dtype)

        b = -sparse_u_cross_grad @ basew.flatten()
        b[basew.size:] = 0.
        x = self.solveAxEqb(A, b)
        out = x.reshape(basew.shape)
        out += basew
        print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (vecu.shape[1], vecu.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out


    def solveRadius1(self, rad0, smoothing_weight, cross_weight):
        st = self.main_window.st
        decimation = self.decimation
        print("decimation", decimation)

        icw = 1.-cross_weight

        uvec = st.vector_u
        # coh = st.coherence[:,:,np.newaxis]
        coh = st.coherence

        # TODO: for testing
        mask = self.createMask()
        coh = coh.copy()*mask

        coh = coh[:,:,np.newaxis]

        wuvec = coh*uvec
        if decimation > 1:
            wuvec = wuvec[::decimation, ::decimation, :]
            coh = coh.copy()[::decimation, ::decimation, :]
            rad0 = rad0.copy()[::decimation, ::decimation] / decimation
        shape = wuvec.shape[:2]
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=False)
        sparse_grad = ImageViewer.sparseGrad(shape)
        sgx, sgy = ImageViewer.sparseGrad(shape, interleave=False)
        hxx = sgx.transpose() @ sgx
        hyy = sgy.transpose() @ sgy
        hxy = sgx @ sgy
        print("sgx", sgx.shape, "hxx", hxx.shape, "hxy", hxy.shape)

        '''
        sgcsr = sparse_grad.tocsr()
        sg0 = sgcsr[0::2, 0]
        sg1 = sgcsr[1::2, 0]
        print("sg0", sg0.shape)
        print("sg1", sg1.shape)
        sg2 = scipy.sparse.vstack((sg0, sg1))
        print("sg2", sg2.shape)
        '''


        # sparse_hess = sparse_grad @ sparse_grad.transpose()
        # sparse_hess = sparse_hess.reshape(-1,sparse_grad.shape[1])

        # print("grad", sparse_grad.shape, "hess", sparse_hess.shape)
        sparse_umb = ImageViewer.sparseUmbilical(shape, np.array(self.umb)/decimation)
        # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*sparse_grad, sparse_umb))
        # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, smoothing_weight*hxy, sparse_umb))
        sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, sparse_umb))
        ## sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*sparse_hess, sparse_umb))

        A = sparse_all
        print("A", A.shape, A.dtype)

        # b[rad0.size:] = 0.
        b = np.zeros((A.shape[0]), dtype=np.float64)
        # NOTE multiplication by decimation factor
        b[:rad0.size] = 1.*coh.flatten()*decimation*icw
        x = self.solveAxEqb(A, b)
        out = x.reshape(rad0.shape)
        print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out

    def solveThetaNew(self, rad, uvec, coh, dot_weight, smoothing_weight, theta_weight):
        st = self.main_window.st
        decimation = self.decimation
        print("decimation", decimation)

        theta = self.createThetaArray()
        oldshape = rad.shape
        # uvec = st.vector_u
        # coh = st.coherence.copy()
        # uvec, coh = self.synthesizeUVecArray(rad)
        # TODO: for testing only!
        # mask = self.createMask()
        # coh *= mask
        coh = coh[:,:,np.newaxis]
        weight = coh.copy()
        # weight = coh*coh*coh
        # weight[:,:] = 1.
        wuvec = weight*uvec
        rwuvec = rad[:,:,np.newaxis]*wuvec
        if decimation > 1:
            # rwuvec = rwuvec[::decimation, ::decimation, :]
            wuvec = wuvec[::decimation, ::decimation, :]
            theta = theta[::decimation, ::decimation]
            # coh = coh.copy()[::decimation, ::decimation, :]
            weight = weight[::decimation, ::decimation, :]
            # Note that rad is divided by decimation
            rad = rad.copy()[::decimation, ::decimation] / decimation
            # recompute rwuvec to account for change in rad
            rwuvec = rad[:,:,np.newaxis]*wuvec
        shape = theta.shape
        sparse_grad = ImageViewer.sparseGrad(shape, rad)
        # sparse_grad = rad[:,:,np.newaxis]*ImageViewer.sparseGrad(shape)
        # sparse_grad = np.vstack((rad,rad)).flatten()*ImageViewer.sparseGrad(shape)
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=False)
        sparse_theta = ImageViewer.sparseDiagonal(shape)
        sparse_all = sparse.vstack((sparse_u_cross_g, dot_weight*sparse_u_dot_g, smoothing_weight*sparse_grad, theta_weight*sparse_theta))
        print("sparse_all", sparse_all.shape)

        umb = np.array(self.umb)
        decimated_umb = umb/decimation
        iumb = decimated_umb.astype(np.int32)
        # bc: branch cut
        bc_rad = rad[iumb[1], :iumb[0]]
        bc_rwuvec = rwuvec[iumb[1], :iumb[0]]
        bc_dot = 2*np.pi*bc_rwuvec[:,1]
        # bc_grad = np.full((bc_rwuvec.shape[0]), 2*np.pi)
        bc_grad = 2*np.pi*bc_rad
        bc_cross = 2*np.pi*bc_rwuvec[:,0]
        bc_f0 = shape[1]*iumb[1]
        bc_f1 = bc_f0 + iumb[0]

        b_dot = np.zeros((sparse_u_dot_g.shape[0]), dtype=np.float64)
        b_dot[bc_f0:bc_f1] += bc_dot.flatten()
        # b_cross = np.full((sparse_u_cross_g.shape[0]), 1.)
        b_cross = weight.flatten()
        b_cross[bc_f0:bc_f1] += bc_cross.flatten()
        b_grad = np.zeros((sparse_grad.shape[0]), dtype=np.float64)
        # b_grad[2*bc_f0+1:2*bc_f1+1:2] += bc_dot.flatten()
        b_grad[2*bc_f0+1:2*bc_f1+1:2] += bc_grad.flatten()
        # b_grad[2*bc_f0:2*bc_f1:2] += bc_dot.flatten()
        # b_theta = np.zeros((sparse_theta.shape[0]), dtype=np.float64)
        b_theta = theta.flatten()
        b_all = np.concatenate((b_cross, dot_weight*b_dot, smoothing_weight*b_grad, theta_weight*b_theta))
        print("b_all", b_all.shape)

        x = self.solveAxEqb(sparse_all, b_all)
        out = x.reshape(shape)
        print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            outl = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
            outn = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_NEAREST)
        # return outl,outn
        return outl

    def computeGrad(self, arr):
        decimation = self.decimation
        oldshape = arr.shape
        if decimation > 1:
            arr = arr.copy()[::decimation, ::decimation]
        shape = arr.shape
        sparse_grad = ImageViewer.sparseGrad(shape)

        # NOTE division by decimation
        grad_flat = (sparse_grad @ arr.flatten()) / decimation
        grad = grad_flat.reshape(shape[0], shape[1], 2)
        gradx = grad[:,:,0]
        grady = grad[:,:,1]
        if decimation > 1:
            gradx = cv2.resize(gradx, (oldshape[1], oldshape[0]), interpolation=cv2.INTER_LINEAR)
            grady = cv2.resize(grady, (oldshape[1], oldshape[0]), interpolation=cv2.INTER_LINEAR)
        return gradx, grady

    def solveThetaOld(self, rad, smoothing_weight, cross_weight):

        st = self.main_window.st
        decimation = self.decimation
        print("decimation", decimation)

        theta = self.createThetaArray()
        oldshape = rad.shape
        uvec = st.vector_u
        coh = st.coherence[:,:,np.newaxis]
        icw = 1.-cross_weight
        wuvec = coh*uvec
        if decimation > 1:
            wuvec = wuvec[::decimation, ::decimation, :]
            theta = theta[::decimation, ::decimation]
            # Note that rad is divided by decimation
            rad = rad.copy()[::decimation, ::decimation] / decimation
        shape = theta.shape
        sparse_grad = ImageViewer.sparseGrad(shape)
        # sparse_u_cross_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=True)
        # sparse_u_dot_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=False)
        rwuvec = rad[:,:,np.newaxis]*wuvec
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=False)
        umb = np.array(self.umb)
        umb_east = umb + (100,0)
        decimated_umb_east = umb_east/decimation
        decimated_umb = umb/decimation
        sparse_umb = ImageViewer.sparseUmbilical(shape, decimated_umb_east)
        sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*sparse_grad, sparse_umb))
        print("sparse_all", sparse_all.shape)
        iumb = decimated_umb.astype(np.int32)
        # bc: branch cut
        bc_rad = rad[iumb[1], :iumb[0]]
        bc_wuvec = wuvec[iumb[1], :iumb[0]]
        bc_dot = 2*np.pi*bc_rad*bc_wuvec[:,1]
        bc_grad = np.full((bc_rad.shape[0]), 2*np.pi)
        # bc_grad = 2*np.pi*bc_rad
        bc_cross = -2*np.pi*bc_rad*bc_wuvec[:,0]
        bc_f0 = rad.shape[1]*iumb[1]
        bc_f1 = bc_f0 + iumb[0]

        b_dot = np.zeros((sparse_u_dot_g.shape[0]), dtype=np.float64)
        b_dot[bc_f0:bc_f1] += bc_dot.flatten()
        b_cross = np.full((sparse_u_cross_g.shape[0]), 1.)
        b_cross[bc_f0:bc_f1] += bc_cross.flatten()
        b_grad = np.zeros((sparse_grad.shape[0]), dtype=np.float64)
        # b_grad[2*bc_f0+1:2*bc_f1+1:2] += bc_dot.flatten()
        b_grad[2*bc_f0+1:2*bc_f1+1:2] += bc_grad.flatten()
        # b_grad[2*bc_f0:2*bc_f1:2] += bc_dot.flatten()
        b_umb = np.zeros((sparse_umb.shape[0]), dtype=np.float64)
        b_all = np.concatenate((icw*b_dot, cross_weight*b_cross, smoothing_weight*b_grad, b_umb))
        print("b_all", b_all.shape)

        x = self.solveAxEqb(sparse_all, b_all)
        out = x.reshape(rad.shape)
        print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out

        '''
        dtheta_flat = sparse_grad @ theta.flatten()
        dtheta = dtheta_flat.reshape(shape[0], shape[1], 2)
        dtheta *=  rad[:,:,np.newaxis]
        dthetay = dtheta[:,:,1]
        dthetax = dtheta[:,:,0]
        # out = np.sqrt((dtheta*dtheta).sum(axis=2))
        out = dthetax
        '''
        '''
        # dtheta_flat = rad.flatten()*(sparse_u_cross_g @ theta.flatten())
        # dtheta_flat[bc_f0:bc_f1] -= bc_cross.flatten()
        dtheta_flat = rad.flatten()*(sparse_u_dot_g @ theta.flatten())
        dtheta_flat[bc_f0:bc_f1] -= bc_dot.flatten()
        dtheta = dtheta_flat.reshape(shape[0], shape[1])
        # print("bcr", bc_cross[120])
        # print("dtheta", dtheta[iumb[1], 120], dtheta[iumb[1]+1, 120], dtheta[iumb[1]-1, 120])
        # ixy = iumb + (-50, +30)
        # ixyo = iumb + (+50, +30)
        # print("dtheta")
        # self.printArray(dtheta, ixy)
        # self.printArray(dtheta, ixyo)
        # print("rad")
        # self.printArray(rad, ixy)
        # self.printArray(rad, ixyo)
        # print("theta")
        # self.printArray(theta, ixy)
        # self.printArray(theta, ixyo)
        # print("wuvec")
        # self.printArray(wuvec, ixy)
        # self.printArray(wuvec, ixyo)
        # out = dtheta + 1.0
        out = dtheta
        '''

        '''
        if decimation > 1:
            out = cv2.resize(out, (oldshape[1], oldshape[0]), interpolation=cv2.INTER_LINEAR)
        print("out", out.min(), out.max())
        return out
        '''


        '''
        icw = 1.-cross_weight

        uvec = st.vector_u
        coh = st.coherence[:,:,np.newaxis]
        wuvec = coh*uvec
        if decimation > 1:
            wuvec = wuvec[::decimation, ::decimation, :]
            theta = theta[::decimation, ::decimation]
            coh = coh.copy()[::decimation, ::decimation, :]
            # Note that rad is divided by decimation
            rad = rad.copy()[::decimation, ::decimation] / decimation
        shape = wuvec.shape[:2]
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(wuvec, is_cross=False)
        sparse_grad = ImageViewer.sparseGrad(shape)
        sparse_umb = ImageViewer.sparseUmbilical(shape, np.array(self.umb)/decimation)
        sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*sparse_grad, sparse_umb))

        b = np.zeros((A.shape[0]), dtype=np.float64)
        b[:rad.size] = 1.*coh.flatten()*decimation*icw
        '''
        self.saveCurrentOverlay()

    def printArray(self, arr, ixy):
        la = arr[ixy[1]:ixy[1]+2, ixy[0]:ixy[0]+2]
        print(la)
        print(la[0,1]-la[0,0])
        print(la[1,0]-la[0,0])

    @staticmethod
    def solveAxEqb(A, b):
        At = A.transpose()
        AtA = At @ A
        print("AtA", AtA.shape, sparse.issparse(AtA))
        # print("ata", ata.shape, ata.dtype, ata[ata!=0], np.argwhere(ata))
        '''
        ssum = AtA.sum(axis=0)
        ssumb = np.abs(ssum)>2.e-10
        # ssum = np.argwhere(np.abs(ssum)>1.e-10)
        print("ssum", np.argwhere(ssumb))
        print(ssum[ssumb])
        '''
        asum = np.abs(AtA).sum(axis=0)
        print("asum", np.argwhere(asum==0))
        Atb = At @ b
        print("Atb", Atb.shape, sparse.issparse(Atb))

        lu = sparse.linalg.splu(AtA.tocsc())
        print("lu created")
        x = lu.solve(Atb)
        print("x", x.shape, x.dtype, x.min(), x.max())
        return x

    def alignUVVec(self, rad0):
        st = self.main_window.st
        uvec = st.vector_u
        shape = uvec.shape[:2]
        sparse_grad = self.sparseGrad(shape)
        delr_flat = sparse_grad @ rad0.flatten()
        delr = delr_flat.reshape(uvec.shape)
        # print("delr", delr[iy,ix])
        dot = (uvec*delr).sum(axis=2)
        # print("dot", dot[iy,ix])
        print("dots", (dot<0).sum())
        print("not dots", (dot>=0).sum())
        st.vector_u[dot<0] *= -1
        st.vector_v[dot<0] *= -1

        # Replace vector interpolator by simple interpolator
        st.vector_u_interpolator = ST.createInterpolator(st.vector_u)
        st.vector_v_interpolator = ST.createInterpolator(st.vector_v)

    def loadRadius0(self, fname):
        try:
            data, data_header = nrrd.read(str(fname), index_order='C')
        except Exception as e:
            print("Error while loading", fname, e)
            return None
        print("rad0 loaded")
        return data

    def saveRadius0(self, fname, rad0):
        header = {"encoding": "raw",}
        nrrd.write(str(fname), rad0, index_order='C')

    def loadArray(self, part):
        fname = self.cache_file_base.with_name(self.cache_file_base.name + part + ".nrrd")
        try:
            data, data_header = nrrd.read(str(fname), index_order='C')
        except Exception as e:
            print("Error while loading", fname, e)
            return None
        print("arr loaded", part)
        return data

    def saveArray(self, part, arr):
        fname = self.cache_file_base.with_name(self.cache_file_base.name + part + ".nrrd")
        header = {"encoding": "raw",}
        nrrd.write(str(fname), arr, index_order='C')

    def loadOrCreateArray(self, part, fn):
        if self.decimation is not None and self.decimation > 1:
            part = "_d%d%s"%(self.decimation, part)
        if self.window_width is not None:
            part = "_w%d%s"%(self.window_width, part)
        if self.no_cache:
            print("calculating arr", part)
            arr = fn()
            return arr

        print("loading arr", part)
        arr = self.loadArray(part)
        if arr is None:
            print("calculating arr", part)
            arr = fn()
            print("saving arr", part)
            self.saveArray(part, arr)
        return arr

    def createRadiusArray(self):
        umb = self.umb
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
        rad = np.sqrt(radsq)
        print("rad", rad.shape)
        return rad

    def createThetaArray(self):
        umb = self.umb
        umb[1] += .5
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        # radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
        theta = np.arctan2(iys-umb[1], ixs-umb[0])
        print("theta", theta.shape, theta.min(), theta.max())
        return theta

    def synthesizeUVecArray(self, rad):
        gradx, grady = self.computeGrad(rad)
        uvec = np.stack((gradx, grady), axis=2)
        print("suvec", uvec.shape)
        # luvec = np.sqrt((uvec*uvec).sum(axis=2))[:,:,np.newaxis]
        luvec = np.sqrt((uvec*uvec).sum(axis=2))
        lnz = luvec != 0
        print("ll", uvec.shape, luvec.shape, lnz.shape)
        uvec[lnz] /= luvec[lnz][:,np.newaxis]
        coh = np.full(rad.shape, 1.)
        coh[:,-1] = 0
        coh[-1,:] = 0
        # coh[rad < 100] = 0
        return uvec, coh

    def createERadiusArray(self, ecc):
        umb = self.umb
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])/(ecc*ecc)
        rad = np.sqrt(radsq)
        print("rad", rad.shape)
        return rad

    def createEThetaArray(self, ecc):
        cth = np.linspace(-np.pi, np.pi, num=129)
        ex = np.cos(cth)
        ey = ecc*np.sin(cth)
        exy = np.vstack((ex,ey)).T
        # print("exy", exy.shape)
        dex = np.diff(exy, axis=0)
        lex = np.sqrt((dex*dex).sum(axis=1))
        # print("lex", lex.shape)
        lsum = np.cumsum(lex)
        lsum = np.concatenate(([0.], lsum))
        # print("lsum", lsum.shape, lsum[0], lsum[-1])
        tlen = lsum[-1]
        # print("tlen", tlen)
        print("r factor", tlen/(2*np.pi))
        lsum /= tlen
        lsum = (2*lsum-1.)*np.pi
        lsum[0] -= .000001
        lsum[-1] += .000001
        # print("lsum", lsum[0], lsum[-1])
        eth = lsum

        umb = self.umb
        umb[1] += .5
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        # print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        # radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
        ctheta = np.arctan2((iys-umb[1])/ecc, ixs-umb[0])
        # etheta = np.interp(ctheta, eth, cth)
        etheta = np.interp(ctheta, cth, eth)
        return etheta

    def createMask(self):
        theta = self.createThetaArray()
        radius = self.createRadiusArray()
        # br = np.logical_and(radius > 80, radius < 200)
        # br = np.logical_and(radius > 180, radius < 200)
        # br = np.logical_and(radius > 80, radius < 100)
        # br = np.logical_and(radius > 80, radius < 120)
        # br = np.logical_and(radius > 180, radius < 220)
        br = np.logical_and(radius > 180, radius < 250)
        # bth = np.logical_and(theta > 1.4, theta < 1.8)
        # bth = np.logical_and(theta > 1.2, theta < 2.0)
        # bth = np.logical_and(theta > .6, theta < 1.)
        # bth = np.logical_and(theta > -.15, theta < .25)
        bth = np.logical_and(theta > -.35, theta < .45)
        # bth = np.logical_and(theta > .2, theta < .6)
        # b = np.logical_and(radius > 180, radius < 220)
        b = np.logical_and(br, bth)
        # mask = np.zeros(self.image.shape, dtype=np.float32)
        mask = np.full(self.image.shape, 1.)
        # mask[b] = .8
        # mask[b] = .9
        # mask[b] = .5
        mask[b] = 0
        return mask

    def solveWindingOneStep(self):
        im = self.image
        if im is None:
            return
        rad = self.createRadiusArray()

        # smoothing_weight = .01
        smoothing_weight = .1
        rad0 = self.loadOrCreateArray(
                "_r0", lambda: self.solveRadius0(rad, smoothing_weight))
        self.alignUVVec(rad0)
        self.overlay_data = rad0
        self.overlay_name = "rad0"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

        # TODO: for testing
        # self.overlay_data = rad0
        # return

        # smoothing_weight = .01
        # hess
        smoothing_weight = .1
        smoothing_weight = .2
        # smoothing_weight = .0001
        # grad
        # smoothing_weight = .01
        cross_weight = 0.95
        # cross_weight = 0.99
        # cross_weight = 0.75
        # cross_weight = 0.95
        # smoothing_weight = .05
        # cross_weight = 0.75
        rad1 = self.loadOrCreateArray(
                "_r1", lambda: self.solveRadius1(rad0, smoothing_weight, cross_weight))

        # TODO: For testing!
        # rad1 *= 2.5
        # rad1 *= 1.4
        # rad1 *= 1.2
        # TODO: For testing!
        # ecc = .5
        # rad1 = self.createERadiusArray(ecc)
        # # r factor
        # rad1 *= .7709

        self.overlay_data = rad1
        self.overlay_name = "rad1"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

        gradx, grady = self.computeGrad(rad1)
        # grad = np.sqrt(gradx*gradx+grady*grady)
        st = self.main_window.st
        uvec = st.vector_u
        gdu = gradx*uvec[:,:,0] + grady*uvec[:,:,1]
        self.overlay_name = "rad1 gdu"
        self.overlay_data = gdu
        self.overlay_maxrad = 2.
        self.overlay_colormap = "hsv"
        self.overlay_interpolation = "linear"
        self.saveCurrentOverlay()

        self.setOverlayByName("rad1")
        # return

        # suvec, scoh = self.synthesizeUVecArray(rad1)

        # TODO: for testing
        # self.overlay_data = rad1
        # return

        # theta0 = self.createThetaArray()
        cross_weight = 0.2
        # cross_weight = 0.5
        smoothing_weight = .01
        # theta0 = self.solveThetaOld(rad, smoothing_weight, cross_weight)
        '''
        # dot_weight = 4
        # dot_weight = 2
        # dot_weight = .5
        # dot_weight = 1
        dot_weight = .1
        dot_weight = .001
        dot_weight = .01
        dot_weight = .1
        dot_weight = .3
        dot_weight = .6
        theta_weight = .001
        dot_weight = .01
        smoothing_weight = .001
        theta_weight = .001

        # sort of symmetric:
        dot_weight = .6
        theta_weight = .001
        smoothing_weight = .001

        dot_weight = .01
        smoothing_weight = .5

        # theta_weight = .1
        # theta_weight = .0
        # smoothing_weight = .01
        # smoothing_weight = 0.0001
        # smoothing_weight = .001
        '''

        # dot_weight = .1
        # smoothing_weight = .001
        dot_weight = .001
        smoothing_weight = .1
        # smoothing_weight = .5
        # smoothing_weight = .6
        smoothing_weight = .4
        # smoothing_weight = .3
        theta_weight = .0001
        # TODO: testing
        # slice 2000
        # rad1 *= 1.8
        # slice 3000
        # rad1 *= 2.8


        # rad1 *= 1.3
        # rad1 *= 1.5
        # ecc = .5
        # rad1 = self.createERadiusArray(ecc)
        # r factor
        # rad1 *= .7709

        # theta0 = self.solveThetaNew(rad1, dot_weight, smoothing_weight, theta_weight)
        st = self.main_window.st
        uvec = st.vector_u
        coh = st.coherence
        # save a copy of the original coh
        uvec, coh = self.synthesizeUVecArray(rad1)
        # coh[:,:] = 1.
        ''''''
        # theta0,theta0nearest = self.loadOrCreateArray(
        theta0 = self.loadOrCreateArray(
                  "_th0", lambda: self.solveThetaNew(rad1, uvec, coh, dot_weight, smoothing_weight, theta_weight))
        ''''''
        # theta0 *= 10

        '''
        ecc = .5
        theta0 = self.createEThetaArray(ecc)
        rad1 = self.createERadiusArray(ecc)
        # r factor
        rad1 *= .7709
        o = Overlay.findItemByName(self.overlays, "rad1")
        o.data = rad1
        '''


        # self.overlay_data = theta0nearest
        self.overlay_data = theta0
        self.overlay_name = "theta0"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = 3.
        self.saveCurrentOverlay()
        # rad1 *= 500.
        # self.overlay_data = rad1
        gradx, grady = self.computeGrad(theta0)
        gradx *= rad1
        grady *= rad1
        grad = np.sqrt(gradx*gradx+grady*grady)

        self.overlay_data = gradx
        self.overlay_name = "rad1 grad theta0 x"
        # self.overlay_colormap = "tab20"
        # self.overlay_interpolation = "nearest"
        self.overlay_colormap = "hsv"
        self.overlay_interpolation = "linear"
        self.overlay_maxrad = 2.
        self.saveCurrentOverlay()
        self.overlay_data = grady
        self.overlay_name = "rad1 grad theta0 y"
        self.saveCurrentOverlay()
        self.overlay_data = grad
        self.overlay_name = "rad1 grad theta0"
        self.overlay_colormap = "hsv"
        self.overlay_interpolation = "linear"
        self.saveCurrentOverlay()


        # st = self.main_window.st
        # uvec = st.vector_u
        gxu = -gradx*uvec[:,:,1] + grady*uvec[:,:,0]
        self.overlay_name = "th0 gxu"
        self.overlay_data = gxu
        self.overlay_colormap = "hsv"
        self.overlay_interpolation = "linear"
        self.saveCurrentOverlay()

        # st = self.main_window.st
        # uvec = st.vector_u
        gdu = gradx*uvec[:,:,0] + grady*uvec[:,:,1]
        self.overlay_name = "th0 gdu"
        self.overlay_data = gdu
        self.overlay_colormap = "hsv"
        self.overlay_interpolation = "linear"
        self.saveCurrentOverlay()

        stcoh = st.coherence
        cargs = np.argsort(stcoh.flatten())
        min_coh = stcoh.flatten()[cargs[len(cargs)//4]]
        rargs = np.argsort(rad1.flatten())
        max_rad1 = rad1.flatten()[rargs[len(rargs)//4]]

        crb = np.logical_and(coh > min_coh, rad1 < max_rad1)
        mgxu = np.median(gxu[crb])

        # uargs = args[:len(args)//4]
        # cgxus = gxu.flatten()[uargs]
        # mgxu = np.median(cgxus)
        print("mgxu", mgxu)

        rad1 /= mgxu
        # smoothing_weight = .6

        theta1 = self.loadOrCreateArray(
                  "_th1", lambda: self.solveThetaNew(rad1, uvec, coh, dot_weight, smoothing_weight, theta_weight))

        self.overlay_data = theta1
        self.overlay_name = "theta1"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = 3.
        self.saveCurrentOverlay()

        gradx, grady = self.computeGrad(theta1)
        gradx *= rad1
        grady *= rad1
        gxu = -gradx*uvec[:,:,1] + grady*uvec[:,:,0]
        self.overlay_name = "th1 gxu"
        self.overlay_data = gxu
        self.overlay_colormap = "hsv"
        self.overlay_maxrad = 2.
        self.overlay_interpolation = "linear"
        self.saveCurrentOverlay()

        # cgxus = gxu.flatten()[uargs]
        # mgxu = np.median(cgxus)
        mgxu = np.median(gxu[crb])
        print("mgxu", mgxu)

        self.setOverlayByName("theta1")

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
            # zslc = cv2.resize(data[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
            zslc = cv2.resize(data[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            # cslc = cm(np.remainder(zslc, 1.))
            cslc = cm(np.remainder(zslc, 1.))
            outrgb[y1:y2, x1:x2, :] = (255*cslc[:,:,:3]*alpha).astype(np.uint8)
        return outrgb

    def drawAll(self):
        if self.image is None:
            return
        self.setStatusTextFromMousePosition()
        main_alpha = .4
        total_alpha = .8

        # print(self.image.shape, self.image.min(), self.image.max())
        outrgb = self.dataToZoomedRGB(self.image, alpha=main_alpha)
        st = self.main_window.st
        other_data = None
        if self.overlay_maxrad is None:
            other_data = self.overlay_data
        elif self.overlay_data is not None:
            other_data = self.overlay_data / self.overlay_maxrad
            # print("maxrad", self.overlay_maxrad)
        if other_data is not None:
            outrgb += self.dataToZoomedRGB(other_data, alpha=total_alpha-main_alpha, colormap=self.overlay_colormap, interpolation=self.overlay_interpolation)

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

            # x1 = dpw+lvecs
            x1 = dpw+.6*lvecs
            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            cv2.polylines(outrgb, lines, False, (0,255,0), 1)

            xm = dpw-.5*lvecs
            xp = dpw+.5*lvecs
            lines = np.concatenate((xm,xp), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            # cv2.polylines(outrgb, lines, False, (0,255,0), 1)

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
    parser.add_argument("--window",
                        type=int,
                        default=None,
                        help="size of window centered around umbilicus")
    parser.add_argument("--colormap",
                        default="viridis",
                        help="colormap")
    parser.add_argument("--interpolation",
                        default="linear",
                        help="interpolation, either linear or nearest")
    parser.add_argument("--maxrad",
                        type=float,
                        default=None,
                        help="max expected radius, in pixels (if not given, will be calculated from umbilicus position)")
    parser.add_argument("--decimation",
                        type=int,
                        default=1,
                        help="decimation factor (default is no decimation)")

    # I decided not to use parse_known_args because
    # I prefer to get an error message if an argument
    # is unrecognized
    # parsed_args, unparsed_args = parser.parse_known_args()
    # return parsed_args, unparsed_args
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    # parsed_args, unparsed_args = process_cl_args()
    parsed_args = process_cl_args()
    # qt_args = sys.argv[:1] + unparsed_args
    qt_args = sys.argv[:1] 
    app = QApplication(qt_args)

    tinter = Tinter(app, parsed_args)
    sys.exit(app.exec())
