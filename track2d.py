import sys
import argparse
import os
# sys.path.append(os.path.join(sys.path[0], '..'))
import pathlib
import math
import csv

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


        '''
        # outdir = pathlib.Path(r"C:\Vesuvius\Projects\tnorm")
        outdir = pathlib.Path(r"C:\Vesuvius\Projects\evol1")
        # tifname = r"C:\Vesuvius\scroll 1 2000-2030\02015.tif"
        tifname = r"C:\Vesuvius\scroll 1 masked 1820-1830\01822.tif"
        # nrrdname = outdir / '02015.nrrd'
        nrrdname = outdir / '01822.nrrd'
        '''
        no_cache = parsed_args.no_cache
        tifname = pathlib.Path(parsed_args.input_tif)
        nrrd_dir = parsed_args.cache_dir
        if nrrd_dir is None:
            nrrd_dir = tifname.parent
        else:
            nrrd_dir = pathlib.Path(nrrd_dir)
        nrrdname = nrrd_dir / (tifname.with_suffix(".nrrd")).name
        # print("tifname", tifname)
        # print("nrrdname", nrrdname)
        # exit()

        # python track2d.py C:\Vesuvius\Projects\evol1\circle.tif --no_cache
        # python track2d.py "C:\Vesuvius\scroll 1 masked xx000\02000.tif" --cache_dir C:\Vesuvius\Projects\evol1

        '''
        use_synth = False
        if not use_synth:
            ''''''
            outdir = pathlib.Path(r"C:\Vesuvius\Projects\evol1")
            ntif = 2000
            stif = "%05d"%ntif
            tifname = r"C:\Vesuvius\scroll 1 masked xx000\%s.tif"%stif
            nrrdname = outdir / ('%s.nrrd'%stif)
            ''''''
        else:

            ''''''
            outdir = pathlib.Path(r"C:\Vesuvius\Projects\evol1")
            tifname = outdir / "circle.tif"
            nrrdname = outdir / "circle.nrrd"
            ''''''
        '''


        '''
        pdir = pathlib.Path(r"C:\Vesuvius\Projects\fourth_easy.khprj")
        tifname = pdir / "st_debug.tif"
        nrrdname = pdir / "st_debug.nrrd"
        '''
        # upath = pathlib.Path(r"C:\Vesuvius\Projects\evol1\umbilicus-scroll1a_zyx.txt")
        # uzyxs = self.readUmbilicus(upath)
        # print("umbs", len(uzyxs))
        # print(uzyxs[0], uzyxs[-1])        
        print("loading tif", tifname)
        self.viewer.loadTIFF(tifname)
        self.st = ST(self.viewer.image)
        '''
        compute_eigens = False
        if compute_eigens:
            print("calculating st")
            self.st.computeEigens()
            print("saving st")
            self.st.saveEigens(nrrdname)
        else:
            print("loading eigens")
            self.st.loadEigens(nrrdname)
        '''
        '''
        if use_synth:
            print("computing eigens (test)")
            self.st.computeEigens()
        else:
            print("computing/loading st")
            self.st.loadOrCreateEigens(nrrdname)
        '''
        if no_cache:
            print("computing structural tensors")
            self.st.computeEigens()
        else:
            print("computing/loading structural tensors")
            self.st.loadOrCreateEigens(nrrdname)
        # print("computing eigens (test)")
        # self.st.computeEigens()
        # for testing
        # self.st.computeEigens()
        # self.testInterp2d()

        

    @staticmethod
    def readUmbilicusOld(fpath):
        zyxs = []
        for line in open(fpath, "r"):
            z,y,x = [int(s) for s in line.split()]
            zyxs.append((z,y,x))
        zyxs.sort()
        return zyxs

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

    def testInterp2d(self):
        # pt1 = (4235, 4495)
        # pt2 = (4621, 4378)
        # pt2 = (4245, 4490)
        # pt1 = (300,300)
        # pt2 = (310,305)
        pt1 = (220,300)
        pt2 = (270,300)
        y = self.st.interp2dWHP(pt1, pt2, .5)
        print("ti2d", y.shape)


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
        self.rays = []
        self.pt1 = None
        self.pt2 = None

    def mousePressEvent(self, e):
        if self.image is None:
            return
        if e.button() | Qt.LeftButton:
            modifiers = QApplication.keyboardModifiers()
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
        if e.key() == Qt.Key_1:
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("pt 1 at",ixy)
            self.pt1 = ixy
            # for testing
            # self.pt1 = (416.0, 217.785766015625)
            # self.pt1 = (416.+1/100, 217.78)
            # self.pt1 = (415.0, 217.78)
            # self.pt1 = (416.00001, 217.78)
            self.drawAll()
        elif e.key() == Qt.Key_2:
            # pt1 = (4235, 4495)
            # pt2 = (4621, 4378)
            # pt2 = (4245, 4500)
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("p2 1 at",ixy)
            self.pt2 = ixy
            # for testing
            # self.pt2 = (751.0, 279.28)
            # self.pt2 = (751.+1/1000, 279.2841796875)
            # self.pt2 = (751.0001, 279.28)
            # self.pt2 = (752.0, 279.28)
            self.drawAll()
            st = self.getST()
            if st is None:
                return
            # for nudge in (-1., -.5, 0, .5, 1.):
            # for nudge in (0., .05, .1):
            # for nudge in (0., -10, 10):
            for nudge in (0.,):
            # for nudge in (1, -1):
            # for nudge in (-1,):
            # for nudge in (0., .1, .2, .4, .8):
            # for nudge in (0., .05, .1, .15, .2):
            # for nudge in (0., .025, .05, .075, .1):
            # for nudge in (-10, -1, 1, 10):
            # for nudge in (.95, .98, .99, 1.):
                y = None
                if self.pt1 is not None and self.pt2 is not None:
                    if False and nudge > 0:
                        y = st.interp2dLsqr(self.pt1, self.pt2, nudge)
                    else:
                        # y = st.interp2dWHP(self.pt1, self.pt2, nudge)
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
            # self.rays = []
            # for nudge in (0, 2., 5.):
            # for nudge in (5.,):
            xnorm = st.vector_u_interpolator([ixy[::-1]])[0]
            # print("xnorm", xnorm)
            # for nudge in (-5., 0., 5.):
            # for nudge in range(-3,4):
            # for nudge in range(0,5):
            for nudge in (0.,):
                for sign in (1,-1):
                    '''
                    status,y,t,nfev = st.call_ivp(ixy, sign, nudge)
                    print(sign, "status", status, nfev)
                    # status 0 means success
                    if status == 0:
                        self.rays.append(y)
                    '''
                    # y = st.sparse_result(ixy, 0, 5, sign, nudge)
                    # y = st.sparse_result(ixy, 0, 5, sign, nudge)
                    # y = st.call_ivp(ixy, sign, nudge)
                    y = st.call_ivp(ixy+1.*nudge*xnorm, sign, tmax=1000)
                    # y = st.call_ivp(ixy, sign, grad_nudge=nudge*.1)
                    if y is not None:
                        # pts = st.sparse_result(y, 0, 5)
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
        # clipping suggested by @TizzyTom
        # clip = 36000./65535.
        # image[image>clip] = clip
        self.image = image
        self.setDefaults()
        # print("loadTIFF")
        self.drawAll()

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
        # print("colormap", colormap)
        # print(data.dtype, data.min(), data.max())
        # cm = cmap.Colormap(colormap, interpolation="linear")
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
            # print(cslc.shape, cslc.dtype, cslc.min(), cslc.max())
            # outrgb[y1:y2, x1:x2, :] = (zslc*alpha*255)[:,:,np.newaxis].astype(np.uint8)
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
        if other_data is not None:
            colormap = "colorbrewer:Set3_12"
            colormap = "colorbrewer:Spectral_11"
            colormap = "viridis"
            outrgb += self.dataToZoomedRGB(np.sqrt(other_data), alpha=total_alpha-main_alpha, colormap=colormap, interpolation="nearest")

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


            ''' TESTING

            # print("lvecs", lvecs.shape, lvecs.dtype, lvecs[5,5])
            # x0 = dpw-lvecs
            x0 = dpw
            x1 = dpw.astype(np.float64)
            uls = st.lambda_u_interpolator(dpir)
            print("uls", uls.shape, uls.dtype, uls[5,5])
            print("x1", x1.shape, x1.dtype, x1[5,5])
            linelen = 10000.
            lvecs = linelen*uls
            x1[:,:,0] += lvecs
            vls = st.lambda_v_interpolator(dpir)
            lvecs = linelen*vls
            x1[:,:,1] += lvecs

            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            cv2.polylines(outrgb, lines, False, (0,255,0), 1)

            '''

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

        for i,ray in enumerate(self.rays):
            # print("ray", self.ray)
            points = self.ixysToWxys(ray)
            # print("points", points)
            points = points.reshape(-1,1,1,2)
            # colors = ((255,255,255), (255,0,255), (0,255,0))
            # colors = ((255,0,255), (0,255,0), (255,255,255), (0,255,0), (255,0,255), )
            # colors = ((255,0,255), (255,0,255), (255,255,255), (0,255,0), (0,255,0), )
            # colors = ((255,0,255), (255,0,0), (0,255,0), (0,255,255), )
            # colors = ((0,255,0),)
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
            # color = colors[(i//2)%len(colors)]
            # TODO: correct version
            color = colors[i%len(colors)]
            # TODO: for testing
            # color = colors[1+(i*2)%len(colors)]

            # cv2.polylines(outrgb, points, True, color, 2)
            # cv2.polylines(outrgb, points, True, color, 2)
            cv2.polylines(outrgb, points, True, color, 4)
            cv2.circle(outrgb, points[0,0,0], 3, (255,0,255), -1)
        if self.pt1 is not None:
            wpt1 = self.ixyToWxy(self.pt1)
            cv2.circle(outrgb, wpt1, 3, (255,0,255), -1)
        if self.pt2 is not None:
            wpt2 = self.ixyToWxy(self.pt2)
            cv2.circle(outrgb, wpt2, 3, (255,0,255), -1)

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
            description="Test one-point and two-point tracking using structural tensors")
    parser.add_argument("input_tif",
                        help="input tiff slice")
    parser.add_argument("--cache_dir",
                        default=None,
                        help="directory where the cache of the structural tensor data is or will be stored; if not given, directory of input tiff slice is used")
    parser.add_argument("--no_cache",
                        action="store_true",
                        help="Don't use cached structural tensors")

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args


if __name__ == '__main__':
    parsed_args, unparsed_args = process_cl_args()
    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)

    tinter = Tinter(app, parsed_args)
    sys.exit(app.exec())
