import sys
import pathlib
import math

import cv2
import numpy as np
import scipy
# from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator
import nrrd

'''
Structural Tensor code based on:

Structure-oriented smoothing and semblance
D. Hale - Center for Wave Phenomena, 2009
https://scholar.google.com/scholar?cluster=15774580112029706695

Other relevant papers:

https://en.wikipedia.org/wiki/Structure_tensor

Horizon volumes with interpreted constraints
X Wu, D Hale - Geophysics, 2015
https://scholar.google.com/scholar?cluster=8389346155142410449

Estimators for orientation and anisotropy in digitized images
LJ Van Vliet, PW Verbeek - ASCI, 1995
https://scholar.google.com/scholar?cluster=8104629214698074825

Fast structural interpretation with structure-oriented filtering
Gijs C. Fehmers and Christian F. W. H Hocker - Geophysics, 2003
https://sci-hub.se/10.1190/1.1598121

The repository
https://github.com/Skielex/structure-tensor
gives a python/CUDA implementation of structure tensors and provides
some interesting references
'''

class ST(object):

    # assumes image is a floating-point numpy array
    def __init__(self, image):
        self.image = image
        self.lambda_u = None
        self.lambda_v = None
        self.vector_u = None
        self.vector_v = None
        self.isotropy = None
        self.linearity = None
        self.coherence = None
        # self.vector_u_interpolator_ = None
        # self.vector_v_interpolator_ = None
        self.lambda_u_interpolator_ = None
        self.lambda_v_interpolator_ = None
        self.vector_u_interpolator_ = None
        self.vector_v_interpolator_ = None
        self.grad_interpolator_ = None
        self.isotropy_interpolator_ = None
        self.linearity_interpolator_ = None
        self.coherence_interpolator_ = None

    def saveImage(self, fname):
        timage = (self.image*65535).astype(np.uint16)
        cv2.imwrite(str(fname), timage)

    def computeEigens(self):
        tif = self.image
        sigma0 = 1.  # value used by Hale
        sigma0 = 2.
        ksize = int(math.floor(.5+6*sigma0+1))
        hksize = ksize//2
        kernel = cv2.getGaussianKernel(ksize, sigma0)
        dkernel = kernel.copy()
        for i in range(ksize):
            x = i - hksize
            factor = x/(sigma0*sigma0)
            dkernel[i] *= factor
        # this gives a slightly wrong answer when tif is constant
        # (should give 0, but gives -1.5e-9)
        # gx = cv2.sepFilter2D(tif, -1, dkernel, kernel)
        # this is equivalent, and gives zero when it should
        gx = cv2.sepFilter2D(tif.transpose(), -1, kernel, dkernel).transpose()
        gy = cv2.sepFilter2D(tif, -1, kernel, dkernel)
        grad = np.concatenate((gx, gy)).reshape(2,gx.shape[0],gx.shape[1]).transpose(1,2,0)

        # gaussian blur
        # OpenCV function is several times faster than
        # numpy equivalent
        sigma1 = 8. # value used by Hale
        # sigma1 = 16.
        # gx2 = gaussian_filter(gx*gx, sigma1)
        gx2 = cv2.GaussianBlur(gx*gx, (0, 0), sigma1)
        # gy2 = gaussian_filter(gy*gy, sigma1)
        gy2 = cv2.GaussianBlur(gy*gy, (0, 0), sigma1)
        # gxy = gaussian_filter(gx*gy, sigma1)
        gxy = cv2.GaussianBlur(gx*gy, (0, 0), sigma1)

        gar = np.array(((gx2,gxy),(gxy,gy2)))
        gar = gar.transpose(2,3,0,1)

        # Explicitly calculate eigenvalue of 2x2 matrix instead of
        # using the numpy linalg eigh function; 
        # the explicit method is about 10 times faster.
        # See https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
        # for a derivation
        ad = gx2+gy2
        sq = np.sqrt((gx2-gy2)**2+4*gxy**2)
        lu = .5*(ad+sq)
        lv = .5*(ad-sq)
        # lv should never be < 0, but numerical issues
        # apparently sometimes cause it to happen
        lv[lv<0]=0
        
        # End of explicit calculation of eigenvalues
        
        # both eigenvalues are non-negative, and lu >= lv.
        # lu is zero if gx2, gy2, and gxy are all zero

        # if lu is 0., set lu and lv to 1.; this will give
        # the correct values for isotropy (1.0) and linearity (0.0)
        # for this case

        lu0 = (lu==0)
        # lu0 = (lu<.002)
        # print("lu==0", lu0.sum())
        lu[lu0] = 1.
        lv[lu0] = 1.

        isotropy = lv/lu
        linearity = (lu-lv)/lu
        coherence = ((lu-lv)/(lu+lv))**2

        # explicitly calculate normalized eigenvectors

        # eigenvector u
        # eigenvector u is perpendicular to the layering.
        # The y component of eigenvector u is always >= 0.
        # This can lead to trouble when the layering is
        # nearly parallel to the y axis, because then eigenvector u
        # is nearly parallel to the x axis.  Small changes in the
        # slope can lead to a change in sign of u's x component,
        # in order to keep the y component sign >= 0.  The sign
        # change in x will cause trouble if the u eigenvector is
        # linearly interpolated, because when the x component changes
        # sign, linear interpolation will send the x component to zero.
        # Yikes!
        vu = np.concatenate((gxy, lu-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vu[lu0,:] = 0.
        vulen = np.sqrt((vu*vu).sum(axis=2))
        vulen[vulen==0] = 1
        vu /= vulen[:,:,np.newaxis]
        # print("vu", vu.shape, vu.dtype)
        
        # eigenvector v
        # eigenvector v is parallel to the layering,
        # and perpendicular to eigenvector u
        '''
        # vv = np.concatenate((gxy, lv-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vv = np.concatenate((lv-gy2, gxy)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vv[lu0,:] = 0.
        vvlen = np.sqrt((vv*vv).sum(axis=2))
        vvlen[vvlen==0] = 1
        vv /= vvlen[:,:,np.newaxis]
        # At this point, the y component of eigenvector v is always <= 0.
        # But it would be better (more consistent) for the cross
        # product of eigenvectors u and v to always have a consistent
        # sign.  This can be ensured by making sure the x component
        # of eigenvector v is always >= 0.
        vv[vv[:,:,0]<0] *= -1
        '''
        # Compute vv as the perpendicular to vu instead of doing
        # a separate numerical calculation for vv as an eigenvector.
        # This provides
        # more stability in the case of precision problems
        vv = vu[:,:,::-1].copy()
        vv[:,:,0] *= -1

        '''
        vuvv = np.abs((vu*vv).sum(axis=2))
        print("vuvv", vuvv.max())
        bb = vuvv > .1
        # bb = np.logical_and(bb, gxy > .0000001)
        print("bads", bb.sum())
        # print("test", (gxy > .0000001).sum())
        # In theory, vuvv (the dot product of u and v eigenvectors)
        # should always be zero.  Sometimes, because of numerical
        # instability caused by extremely low values of gxy,
        # the dot product can be around 1.  
        print("vuvv gxy", vuvv[np.abs(gxy) > 1.e-18].max())
        if bb.sum() > 0:
            print(gx2[bb][0])
            print(gxy[bb][0])
            print(gy2[bb][0])
            print(lu[bb][0])
            print(lv[bb][0])
        '''

        self.lambda_u = lu
        self.lambda_v = lv
        self.vector_u = vu
        self.vector_v = vv
        self.grad = grad
        self.isotropy = isotropy
        self.linearity = linearity
        self.coherence = coherence
        '''
        print(lu[499:501,469:471])
        print(lv[499:501,469:471])
        print(vu[499:501,469:471])
        print(vv[499:501,469:471])
        print(gxy[499:501,469:471])
        print(gx2[499:501,469:471])
        print(gy2[499:501,469:471])
        '''

        '''
        self.lambda_u_interpolator = ST.createInterpolator(self.lambda_u)
        self.lambda_v_interpolator = ST.createInterpolator(self.lambda_v)
        self.vector_u_interpolator = ST.createVectorInterpolator(self.vector_u)
        self.vector_v_interpolator = ST.createVectorInterpolator(self.vector_v)
        self.grad_interpolator = ST.createInterpolator(self.grad)
        self.isotropy_interpolator = ST.createInterpolator(self.isotropy)
        self.linearity_interpolator = ST.createInterpolator(self.linearity)
        self.coherence_interpolator = ST.createInterpolator(self.coherence)
        '''

    @property
    def lambda_u_interpolator(self):
        if self.lambda_u_interpolator_ is None:
            self.lambda_u_interpolator_ = ST.createInterpolator(self.lambda_u)
        return self.lambda_u_interpolator_

    @property
    def lambda_v_interpolator(self):
        if self.lambda_v_interpolator_ is None:
            self.lambda_v_interpolator_ = ST.createInterpolator(self.lambda_v)
        return self.lambda_v_interpolator_

    @property
    def vector_u_interpolator(self):
        if self.vector_u_interpolator_ is None:
            print("create vui")
            self.vector_u_interpolator_ = ST.createVectorInterpolator(self.vector_u)
        return self.vector_u_interpolator_

    @vector_u_interpolator.setter
    def vector_u_interpolator(self, nv):
        # if nv is not None:
        #     raise ValueError("Value must be None")
        print("replace vui")
        self.vector_u_interpolator_ = nv

    @property
    def vector_v_interpolator(self):
        if self.vector_v_interpolator_ is None:
            print("create vvi")
            self.vector_v_interpolator_ = ST.createVectorInterpolator(self.vector_v)
        return self.vector_v_interpolator_

    @vector_v_interpolator.setter
    def vector_v_interpolator(self, nv):
        # if nv is not None:
        #     raise ValueError("Value must be None")
        print("replace vvi")
        self.vector_v_interpolator_ = nv

    @property
    def grad_interpolator(self):
        if self.grad_interpolator_ is None:
            self.grad_interpolator_ = ST.createInterpolator(self.grad)
        return self.grad_interpolator_

    @property
    def isotropy_interpolator(self):
        if self.isotropy_interpolator_ is None:
            self.isotropy_interpolator_ = ST.createInterpolator(self.isotropy)
        return self.isotropy_interpolator_

    @property
    def linearity_interpolator(self):
        if self.linearity_interpolator_ is None:
            self.linearity_interpolator_ = ST.createInterpolator(self.linearity)
        return self.linearity_interpolator_

    @property
    def coherence_interpolator(self):
        if self.coherence_interpolator_ is None:
            self.coherence_interpolator_ = ST.createInterpolator(self.coherence)
        return self.coherence_interpolator_

    def create_vel_func(self, xy, sign, inertia, grad_nudge):
        x0 = np.array((xy))
        # vv0 = self.vector_v_interpolator((x0[::-1]))[0]
        # print(x0, vv0)
        def vf(t, y):
            # print("y", y.shape, y.dtype)
            # print(t,y)
            # vv stores vector at x,y in vv[y,x], but the
            # vector itself is returned in x,y order
            # self.t2xy[t] = y
            if len(self.ivp_ts) == 0 or self.ivp_ts[-1] < t:
                self.ivp_ts.append(t)
                self.ivp_xys.append(y)
            yr = y[::-1]
            vv = self.vector_v_interpolator((yr))[0]
            grad = self.grad_interpolator((yr))[0]
            cohs = self.coherence_interpolator((yr))[0]
            # print(t,y,vv,grad)
            # print("vv", vv.shape, vv.dtype)
            # print(vv)
            # vvx = vv[0]
            # yx = y[1]
            # if vvx * (yx-x0) < 0:
            acting_x0 = x0
            '''
            ts = self.t2xy.keys()
            if len(ts) > 10:
                ts = sorted(ts)
                acting_x0 = self.t2xy[ts[-10]]
                # print(acting_x0)
            '''
            if len(self.ivp_xys) > 10:
                # acting_x0 = self.ivp_xys[-3]
                lxys = np.array(self.ivp_xys[-10:-5])
                acting_x0 = lxys.sum(axis=0)/len(lxys)

            if t==0:
                if vv[0]*sign < 0:
                    vv *= -1
            elif (vv*(y-acting_x0)).sum() < 0: # check vv direction
                vv *= -1                # relative to starting point
            # print(vv)
            # vv += 5*grad
            d = y-acting_x0
            ln = np.sqrt((d*d).sum())
            if ln == 0:
                ln = .01
            d /= ln
            vv *= cohs
            vv += inertia*d
            # vv += nudge*grad
            vv += grad_nudge*grad
            return vv
        return vf

    # the idea of using Runge-Kutta (which solve_ivp uses)
    # was suggested by @TizzyTom
    # Uses Runge-Kutta to extrapolate (to the left or
    # right, depending on sign) from the given point.
    def call_ivp(self, xy, sign, inertia=0.2, grad_nudge=0.2, tmax=500):
        vel_func = self.create_vel_func(xy, sign, inertia=inertia, grad_nudge=grad_nudge)
        # tmax = 400
        # tmax = 500
        # tsteps = np.linspace(0,tmax,tmax//10)
        # sol = solve_ivp(fun=vel_func, t_span=[0,tmax], t_eval=tsteps, atol=1.e-8, rtol=1.e-5, y0=xy)
        # sol = solve_ivp(fun=vel_func, t_span=[0,tmax], t_eval=tsteps, atol=1.e-8, y0=xy)
        # for testing
        # sol = solve_ivp(fun=vel_func, t_span=[0,tmax], rtol=1.e-5, atol=1.e-8, y0=xy)
        # self.t2xy = {}
        self.ivp_ts = []
        self.ivp_xys = []
        sol = solve_ivp(fun=vel_func, t_span=[0,tmax], max_step=2., y0=xy)
        # self.t2xy = {}
        # print("solution")
        # print(sol)
        # return (sol.status, sol.y, sol.t, sol.nfev)
        if sol.status != 0:
            print("ivp status", sol.status)
            return None
        if sol.y is None:
            print("ivp y is None")
            return None
        return sol.y.transpose()

    # This is used by the Wu-Hale version of interp2d
    def solve2d(self, xs, ys, constraints):
        # return ys.copy()

        # vv stores the vector at x,y in vv[y,x], but the
        # vector itself is returned in x,y order
        # yxs = np.stack((ys, xs), axis=1)
        # vvecs = self.vector_v_interpolator(yxs)
        # cohs = self.coherence_interpolator(yxs)

        # for each segment in xs,ys, find the midpoint
        mxs = .5*(xs[:-1]+xs[1:])
        # length (in x direction) of each segment
        lxs = xs[1:]-xs[:-1]
        mys = .5*(ys[:-1]+ys[1:])
        myxs = np.stack((mys, mxs), axis=1)
        # print("xs")
        # print(xs)
        # print("myxs")
        # print(myxs)
        # vv stores the vector at x,y in vv[y,x], but the
        # vector itself is returned in x,y order
        vvecs = self.vector_v_interpolator(myxs)
        # print("vvecs")
        # print(vvecs)
        cohs = self.coherence_interpolator(myxs)
        # cohs = self.linearity_interpolator(myxs)
        # cohs = np.full((myxs.shape[0]), 1.0, dtype=np.float64)
        grads = self.grad_interpolator(myxs)
        # make sure vvecs[:,0] is always > 0
        vvecs[vvecs[:,0] < 0] *= 1
        # nudge:
        # vvecs[:,1] += 5.*grads[:,1]
        vvzero = vvecs[:,0] == 0
        vslope = np.zeros((vvecs.shape[0]), dtype=np.float64)
        vslope[~vvzero] = vvecs[~vvzero,1]/vvecs[~vvzero,0]
        # print("vslope")
        # print(vslope)

        # compute pim1 = slope = vvecs[:,1]/vvecs[:,0]
        # compute dfidx = dy/dx from xs and ys
        # wim1 = cohs
        # print("cohs")
        # print(cohs)

        # xys = np.stack((xs, ys), axis=1)
        
        # Following the notation in Wu and Hale (2015):
        f = ys
        # print("f", f.shape)
        # print(f)
        # W = np.diag(cohs)
        G = np.diag(-1./lxs)
        # print(G.shape)
        G = np.append(G, np.zeros((G.shape[0]),dtype=np.float64)[:,np.newaxis], axis=1)
        # print(G.shape)
        G[:,1:] += np.diag(1./lxs)
        # print("G", G.shape)
        # print(G)
        # G = np.concatenate((G,G,G), axis=0)
        G = np.concatenate((G,G), axis=0)
        # print("G", G.shape)
        # print(G)

        # print("cohs", cohs.shape)
        rweight = .1
        rws = np.full(cohs.shape, rweight, dtype=np.float64)
        # print("rws", rws.shape)
        rvs = np.full(vslope.shape, 0., dtype=np.float64)
        # print("rvs", rvs.shape)
        # gweight = 0.1
        # gws = np.full(cohs.shape, gweight, dtype=np.float64)
        # # print("gws", gws.shape)
        # gvs = grads[:,1]
        # # print("gvs", gvs.shape)

        # W = np.diag(np.concatenate((cohs,rws,gws)))
        W = np.diag(np.concatenate((cohs,rws)))
        # print("W", W.shape)
        # v = np.concatenate((vslope, rvs, gvs))
        v = np.concatenate((vslope, rvs))
        # print("v", v.shape)

        wgtw = (W@G).T @ W
        A = wgtw@G
        # print("A", A.shape)
        # print(A)
        # print(A.sum(axis=1))
        b = wgtw@v
        # print("b", b.shape)
        # print(b)
        cons = np.array(constraints, dtype=np.float64)
        cidxs = cons[:,0].astype(np.int64)
        cys = cons[:,1]

        Z = np.identity(A.shape[0])
        # Z = Z[:,1:]
        Z = np.delete(Z, cidxs, axis=1)
        # print("Z", Z.shape)
        # print(np.linalg.inv(Z.T@A@Z))
        f0 = np.zeros(f.shape[0], dtype=np.float64)
        f0[cidxs] = cys
        # print("f0", f0.shape)
        # print(f0)

        try:
            p = np.linalg.inv(Z.T@A@Z)@Z.T@(b-A@f0)
        except:
            print("Singular matrix!")
            return ys.copy()

        newf = f0 + Z@p

        # print(np.linalg.inv(A))
        # print(scipy.linalg.inv(A))

        # newf = np.linalg.inv(A)@b
        # newf = np.linalg.inv(Z.T@A@Z)@b
        # print("newf")
        # print(newf)
        newys = newf + ys[0] - newf[0]
        return newys
        # return ys.copy()

    # Given 2 2D points in xy, create transform (rotation plus
    # translation, no rescaling) to transform from line-based
    # (line from xy1 to xy2) coordinates to xy coordinates
    @staticmethod
    def createXform(xy1, xy2):
        d12 = np.array(xy2)-np.array(xy1)
        dst = np.sqrt((d12*d12).sum())
        if dst == 0:
            dst = .1
        n12 = d12/dst
        nx = n12[0]
        ny = n12[1]
        # transform st to xy
        pxform = [
                [nx, -ny, xy1[0]],
                [ny,  nx, xy1[1]],
                [ 0,   0,     1 ]
                ]
        st2xy = np.array(pxform)
        return st2xy

    # Apply 3x3 xform (2x2 similarity transform, plus translation)
    # to Nx2 array (i.e. "a" is a list of 2D points).
    # Output is also Nx2.
    @staticmethod
    def applyXform(a, xform):
        # Add a 3rd column, all 1's to a
        ap1 = np.insert(a, 2, 1., axis=1)
        # print(xform.shape, a.shape, ap1.shape)
        out = ((xform @ ap1.T)[:2,:]).T
        return out

    # This is used by the parameterized Wu-Hale version of interp2d.
    # Note that this differs from Wu-Hale in that it compares the slope of
    # the v vector (the vector pointing in the direction of linear
    # structures) to the slope of the solved-for line, rather
    # than comparing normals.
    def solve2dp(self, ss, ts, st2xy, constraints, smoothing=.075):

        # for each segment in ss,ys, find the midpoint
        mss = .5*(ss[:-1]+ss[1:])
        # length (in x direction) of each segment
        lss = ss[1:]-ss[:-1]
        mts = .5*(ts[:-1]+ts[1:])
        '''
        msts = np.stack((mss, mts), axis=1)
        msts = np.insert(msts, 2, 1., axis=1)

        mxys = ((st2xy @ msts.T)[:2,:]).T
        '''
        msts = np.stack((mss, mts), axis=1)
        mxys = self.applyXform(msts, st2xy)

        # print("mxys")
        # print(mxys)

        # vv stores the vector at x,y in vv[y,x], but the
        # vector itself is returned in x,y order
        myxs = mxys[:, ::-1]
        xyvvecs = self.vector_v_interpolator(myxs)
        # create xy2st_rot from st2xy
        xy2st_rot = np.linalg.inv(st2xy[:2,:2])
        vvecs = (xy2st_rot @ xyvvecs.T).T
        cohs = self.coherence_interpolator(myxs)
        xygrads = self.grad_interpolator(myxs)
        grads = (xy2st_rot @ xygrads.T).T

        # make sure vvecs dot (xy1 to xy2) is always > 0
        # (but this is unnecessary, since slope is
        # vvecs[:,1] / vvecs[:,0], whose sign is the
        # same either way)
        # vvecs[vvecs[:,0] < 0] *= -1
        # vvecs[vvecs[:,1] < 0] *= -1

        ''''''
        vvzero = vvecs[:,0] == 0
        # print(vvzero.sum())
        # print(vvecs)
        vslope = np.zeros((vvecs.shape[0]), dtype=np.float64)
        vslope[~vvzero] = vvecs[~vvzero,1]/vvecs[~vvzero,0]
        # multiply cohs by inverse of slope (outside of some boundaries)
        # so that steep slopes are weighted less
        slope_break = 2.
        steep = np.abs(vslope) > slope_break
        factor = slope_break / vslope[steep]
        factor = factor * factor
        cohs[steep] *= factor

        ''''''
        '''
        max_slope = 2.
        vslope[vslope>max_slope] = max_slope
        vslope[vslope<-max_slope] = -max_slope
        '''
        '''
        vslope = np.arctan2(vvecs[:,1], vvecs[:,0])
        vslope[vslope<-.5*np.pi] += np.pi
        vslope[vslope>.5*np.pi] -= np.pi
        '''

        # Following the notation in Wu and Hale (2015):
        f = ts
        G = np.diag(-1./lss)
        G = np.append(G, np.zeros((G.shape[0]),dtype=np.float64)[:,np.newaxis], axis=1)
        G[:,1:] += np.diag(1./lss)
        G = np.concatenate((G,G), axis=0)

        rweight = smoothing
        rws = np.full(cohs.shape, rweight, dtype=np.float64)
        rvs = np.full(vslope.shape, 0., dtype=np.float64)

        W = np.diag(np.concatenate((cohs,rws)))
        v = np.concatenate((vslope, rvs))

        wgtw = (W@G).T @ W
        A = wgtw@G
        b = wgtw@v
        cons = np.array(constraints, dtype=np.float64)
        cidxs = cons[:,0].astype(np.int64)
        cys = cons[:,1]

        Z = np.identity(A.shape[0])
        Z = np.delete(Z, cidxs, axis=1)
        f0 = np.zeros(f.shape[0], dtype=np.float64)
        f0[cidxs] = cys

        try:
            p = np.linalg.inv(Z.T@A@Z)@Z.T@(b-A@f0)
        except:
            print("Singular matrix!")
            return ts.copy()

        newf = f0 + Z@p

        # print(np.linalg.inv(A))
        # print(scipy.linalg.inv(A))

        # TODO: what does this do?
        newts = newf + ts[0] - newf[0]

        return newts
        # return ys.copy()

    # Wu-Hale version, using the Wu-Hale approach to
    # solving (minimizing) the objective function,
    # on a parameterized line
    def interp2dWHP(self, xy1, xy2, smoothing=.075):
        print("interp2dWHP", xy1, xy2)
        epsilon = .01
        d12 = np.array(xy2)-np.array(xy1)
        dst = np.sqrt((d12*d12).sum())
        ns = int(dst+1)
        ss = np.linspace(0., ns, ns, dtype=np.float64)
        ts = np.zeros(ns, dtype=np.float64)
        constraints = ((0., 0.), (ns-1, 0.))
        st2xy = self.createXform(xy1, xy2)

        prev_dts = -1.
        min_dts = -1.
        min_ts = None
        for _ in range(20):
            # call solver: given ss, ts, constraints, return new ts based on
            # solving a linear equation
            prev_ts = ts.copy()
            ts = self.solve2dp(ss, ts, st2xy, constraints, smoothing)
            # print("new ts", ts)
            # keep calling solver until solution stabilizes
            # stop condition: average absolute update is less than epsilon
            dts = np.abs(ts-prev_ts)
            avg_dts = dts.sum()/ns
            print("avg_dts", avg_dts)
            if min_dts < 0. or avg_dts < min_dts:
                min_dts = avg_dts
                min_ts = ts.copy()
            if avg_dts < epsilon:
                break
            # if prev_dys > 0 and prev_dys < avg_dys:
            #     ys = prev_ys
            #     break
            prev_dts = avg_dts
        # xform ss, ts to xys
        stst = np.stack((ss, min_ts), axis=1)
        # print("stst", stst.shape)
        xys = self.applyXform(stst, st2xy)
        print("xys", xys.shape)
        # print(xys)
        return xys


    # Wu-Hale version, using the Wu-Hale approach to
    # solving (minimizing) the objective function.
    def interp2dWH(self, xy1, xy2):
        print("interp2dWH", xy1, xy2)
        # order of xy1, xy2 shouldn't matter, except when creating xs
        # create xs (ordered list of x values; y is to be solved for)
        epsilon = .01
        oxy1, oxy2 = xy1, xy2
        if xy1[0] > xy2[0]:
            oxy1,oxy2 = oxy2,oxy1
        x1 = oxy1[0]
        y1 = oxy1[1]
        x2 = oxy2[0]
        y2 = oxy2[1]
        nx = int(x2-x1)+1
        xs = np.linspace(x1,x2,nx, dtype=np.float64)
        # create list of constraints (each is (index, y))
        constraints = ((0, y1), (nx-1, y2))
        # create initial ys (line interpolating xy1 and xy2)
        ys = np.linspace(y1,y2,nx, dtype=np.float64)
        prev_dys = -1.
        min_dys = -1.
        min_ys = None
        for _ in range(20):
            # call solver: given xs, ys, constraints, return new ys based on
            # solving a linear equation
            prev_ys = ys.copy()
            ys = self.solve2d(xs, ys, constraints)
            # print("new ys", ys)
            # keep calling solver until solution stabilizes
            # stop condition: average absolute update is less than epsilon
            dys = np.abs(ys-prev_ys)
            avg_dys = dys.sum()/nx
            print("avg_dys", avg_dys)
            if min_dys < 0. or avg_dys < min_dys:
                min_dys = avg_dys
                min_ys = ys.copy()
            if avg_dys < epsilon:
                break
            # if prev_dys > 0 and prev_dys < avg_dys:
            #     ys = prev_ys
            #     break
            prev_dys = avg_dys
        # xys = np.stack((xs, ys), axis=1)
        xys = np.stack((xs, min_ys), axis=1)
        print("xys", xys.shape)
        # print(xys)
        return xys

    # Interpolate between two points, using the same objective function
    # as the Wu-Hale interpolator, but a different solver:
    # least_squares from scipy.optimize
    def interp2dLsqr(self, xy1, xy2, nudge=0.):
        print("interp2dLsqr", xy1, xy2)
        oxy1, oxy2 = xy1, xy2
        if xy1[0] > xy2[0]:
            oxy1,oxy2 = oxy2,oxy1
        ynudge = 0.
        x1 = oxy1[0]
        y1 = oxy1[1] + ynudge
        x2 = oxy2[0]
        y2 = oxy2[1] + ynudge
        nx = int(x2-x1)+1
        xs = np.linspace(x1,x2,nx, dtype=np.float64)
        # create list of constraints (each is (index, y))
        constraints = ((0, y1), (nx-1, y2))
        cons = np.array(constraints, dtype=np.float64)
        cidxs = cons[:,0].astype(np.int64)
        ncidxs=cidxs.shape[0]
        cys = cons[:,1]
        # create initial ys (line interpolating xy1 and xy2)
        y0s = np.linspace(y1,y2,nx, dtype=np.float64)

        # for each segment in xs,ys, find the midpoint
        mxs = .5*(xs[:-1]+xs[1:])
        # length (in x direction) of each segment
        lxs = xs[1:]-xs[:-1]
        # nx = xs.shape[0]
        ndx = nx-1
        # fs = np.zeros((2*ndx+cidxs.shape[0],))
        # print("fs", fs.shape)
        vslope = np.zeros((ndx), dtype=np.float64)
        # rweight = nudge
        # rweight = .001
        # nudge = 1.
        self.global_cohs = None
        # self.global_dcohs_dy = None
        self.global_ddangle_dy = None
        # use_angle = True
        # coh_mult = abs(nudge)
        coh_thresh = .99
        # If use_angle is true, instead of using the Wu-Hale
        # objective function (error in slope) use error-in-angle
        use_angle = nudge < 0
        rweight = .1

        # objective function used by least_squares
        def fun(iys):
            # print("iys", iys)
            oys = iys[cidxs]
            ys = iys.copy()
            ys[cidxs] = cys
            # print("ys", ys.shape)
            # print(ys)
            mys = .5*(ys[:-1]+ys[1:])
            myxs = np.stack((mys, mxs), axis=1)
            vvecs = self.vector_v_interpolator(myxs)
            # grads = self.grad_interpolator(myxs)
            # vvecs[:,1] += nudge*grads[:,1]
            cohs = self.coherence_interpolator(myxs)
            # cohs = self.linearity_interpolator(myxs)
            # myxsd = myxs.copy()
            # dyc = .01
            # myxsd[:,0] += dyc
            # cohsd = self.linearity_interpolator(myxsd)
            # TODO: Testing only!!
            # cohs = np.linspace(.8,.8+(ndx+1)*.01,ndx, dtype=np.float64)
            # self.global_dcohs_dy = (cohsd-cohs)/dyc
            # print("cohs", cohs)
            # make sure vvecs[:,0] is always > 0
            vvecs[vvecs[:,0] < 0] *= -1
            fs = np.zeros((2*ndx+ncidxs,))
            vys = vvecs[:,1]
            # if coh_mult < 19.:
            #     cohs = cohs*(1.-(abs(vys)+.0001)**coh_mult)
            cohs[np.abs(vys)>coh_thresh] = 0.
            self.global_cohs = cohs

            if use_angle:
                vangle = np.arctan2(vvecs[:,1],vvecs[:,0])
                yangle = np.arctan2(ys[1:]-ys[:-1], lxs)
                dya = .01
                yanglep = np.arctan2(dya+ys[1:]-ys[:-1], lxs)
                ddangle = (yanglep-yangle)/dya
                # ddangle[ddangle<-np.pi/2] += np.pi
                # ddangle[ddangle>np.pi/2] -= np.pi
                self.global_ddangle_dy = ddangle

                dangle = vangle-yangle
                dangle[dangle<-np.pi/2] += np.pi
                dangle[dangle>np.pi/2] -= np.pi
                fs[:ndx] = dangle*cohs

                yslope = (ys[1:]-ys[:-1])/lxs
                fs[ndx:2*ndx] = rweight*yslope
                # fs[ndx:2*ndx] = rweight*yangle
            else:
                # nudge:
                # vvecs[:,1] += 5.*grads[:,1]
                vvzero = vvecs[:,0] == 0
                vslope[:] = 0.
                # vslope = np.zeros((vvecs.shape[0]), dtype=np.float64)
                vslope[~vvzero] = vvecs[~vvzero,1]/vvecs[~vvzero,0]
                # aslope = np.abs(vslope)**2
                # aslopegt1 = aslope > 1.
                # cohs[aslopegt1] /= aslope[aslopegt1]
                self.global_cohs = cohs

                yslope = (ys[1:]-ys[:-1])/lxs
                # fs = np.array((2*ndx+cidxs.shape[0],))
                # print("fs fun", fs.shape)
                fs = np.zeros((2*ndx+ncidxs,))
                fs[:ndx] = (vslope-yslope)*cohs
                fs[ndx:2*ndx] = rweight*yslope
            fs[2*ndx:] = cys-oys

            '''
            print("vangle", vangle.shape)
            print(vangle)
            print("vvecs", vvecs.shape)
            print(vvecs)
            print("yangle", yangle.shape)
            print(yangle)
            print("dy")
            print(ys[1:]-ys[:-1])
            print("lxs", lxs.shape)
            print(lxs)
            '''
            # print("fs", fs.shape)
            # print(fs)
            # print("fs", fs)
            # print()
            return fs

        # Used to specify the Jacobian (see below)
        def matvec(idys):
            # print("idys", idys.shape)
            # print("idys", idys)
            idys = idys.flatten()
            dys = idys.copy()
            odys = dys[cidxs]
            # print(dys)
            dys[cidxs] = 0.
            # print(cidxs, odys)
            dfs = np.zeros((2*ndx+ncidxs,))
            # print("dfs", dfs.shape)

            if use_angle:
                dfs[:ndx] = self.global_ddangle_dy*dys[:-1]
                dfs[:ndx] += -self.global_ddangle_dy*dys[1:]
                dfs[:ndx] *= self.global_cohs
                # dfs[ndx:2*ndx] = -rweight*self.global_ddangle_dy*dys[:-1]
                # dfs[ndx:2*ndx] += rweight*self.global_ddangle_dy*dys[1:]
                dfs[ndx:2*ndx] = -rweight*dys[:-1]
                dfs[ndx:2*ndx] += rweight*dys[1:]
                dfs[ndx:2*ndx] /= lxs
            else:
                dfs[:ndx] = dys[:-1]
                dfs[:ndx] += -dys[1:]
                dfs[:ndx] *= self.global_cohs
                # dfs[:ndx] += .5*dys[:-1]*self.global_dcohs_dy
                # dfs[:ndx] += .5*dys[1:]*self.global_dcohs_dy
                dfs[:ndx] /= lxs
                dfs[ndx:2*ndx] = -rweight*dys[:-1]
                dfs[ndx:2*ndx] += rweight*dys[1:]
                dfs[ndx:2*ndx] /= lxs

            dfs[2*ndx:] = -odys
            # print("dfs", dfs)
            # print()
            return dfs

        # Used to specify the Jacobian (see below)
        def rmatvec(idfs):
            # print("idfs", idfs)
            dfs = idfs
            dys = np.zeros((nx,))
            cohs = self.global_cohs
            if use_angle:
                dys[:ndx] = self.global_ddangle_dy*dfs[:ndx]*cohs
                dys[1:nx] += -self.global_ddangle_dy*dfs[:ndx]*cohs
                # dys[1:nx] += rweight*self.global_ddangle_dy*dfs[ndx:2*ndx]
                # dys[:ndx] += -rweight*self.global_ddangle_dy*dfs[ndx:2*ndx]
                dys[:ndx] += -rweight*dfs[ndx:2*ndx]/lxs
                dys[1:nx] += rweight*dfs[ndx:2*ndx]/lxs
            else:
                dys[:ndx] = dfs[:ndx]*cohs/lxs
                dys[1:nx] += -dfs[:ndx]*cohs/lxs

                # dcohs_dy = self.global_dcohs_dy
                # dys[:ndx] += dcohs_dy
                # dys[1:nx] += dcohs_dy
                dys[:ndx] += -rweight*dfs[ndx:2*ndx]/lxs
                dys[1:nx] += rweight*dfs[ndx:2*ndx]/lxs


            # Notice: not +=
            dys[cidxs] = -dfs[2*ndx:]
            # print("dys", dys)
            # print()
            return dys


        '''
        def matmat(idys):
            dys = idys.copy()
            odys = dys[cidxs,:]
            # print(dys)
            dys[cidxs,:] = 0.
            # print(cidxs, odys)
            dfs = np.zeros((2*ndx+ncidxs,dys.shape[1]))
            dfs[:ndx,:] = dys[:-1,:]
            dfs[:ndx,:] += -dys[1:,:]
            dfs[:ndx,:] *= self.global_cohs
            dfs[ndx:2*ndx,:] = -rweight*dys[:-1,:]
            dfs[ndx:2*ndx,:] += rweight*dys[1:,:]
            dfs[2*ndx:,:] = -odys
            return dfs
        '''

        # Jacobian function used by least_squares.
        # If no Jacobian is specified, least_squares uses
        # explicit differencing, which is much slower
        def jac(ys):
            return LinearOperator((2*ndx+ncidxs, nx), matvec=matvec, rmatvec=rmatvec)

        '''
        f0 = fun(y0s)
        print(self.global_cohs)
        # print(self.global_dcohs_dy)
        # lo = jac(y0s)
        # for i in range(nx):
        for i in range(2):
            y1 = y0s.copy()
            y1[i] += .001
            f1 = fun(y1)
            print(f1-f0)
            # print(matvec(y1-y0s))
            print(matvec(y1-y0s))
            print()
        '''

        '''
        maxx = 3
        # maxx = 100

        f0 = fun(y0s)

        for i in range(min(maxx,nx)):
            y1 = y0s.copy()
            y1[i] += .001
            f1 = fun(y1)
            print(f1-f0)
        print()

        for i in range(min(maxx,nx)):
            y1 = np.zeros((y0s.shape))
            y1[i] += .001
            print(matvec(y1))
        print()

        for i in range(min(maxx,2*ndx+ncidxs)):
            f1 = np.zeros((2*ndx+ncidxs))
            f1[i] += .001
            print(rmatvec(f1))
        print()
        '''

        r = least_squares(fun, y0s, jac=jac)
        # r = least_squares(fun, y0s)

        # if self.global_cohs is not None:
        #     print("global_cohs")
        #     print(self.global_cohs)
        print("r", r.status, r.nfev, r.njev, r.cost)
        # print("r.grad")
        # print(r.grad)
        # print("r.x")
        # print(r.x)
        # return r.x
        xys = np.stack((xs, r.x), axis=1)
        print("xys", xys.shape)
        # print(xys)
        return xys

    # ix is rounding pixel position, ix0 is shift before rounding
    # output is transposed relative to y from call_ivp
    # def evenly_spaced_result(self, xy, ix0, ix, sign, nudge=0):
    def evenly_spaced_result(self, y, ix0, ix):
        # status, y, t, nfev = self.call_ivp(xy, sign, nudge)
        # if status != 0:
        #     print("status", status)
        #     return None
        # y = y.transpose()
        # print("y 1", y.shape)
        if y is None:
            return None
        if len(y) < 2:
            print("too few points", len(y))
            return None
        dx = np.diff(y[:,0])
        sign = 1
        if dx[0] < 0:
            sign = -1
        bad = (sign*dx <= 0)
        ibad = np.argmax(bad)
        # can't happen now; sign is based on dx[0]
        if bad[0]:
            print("bad0")
            print(y)
            return None
        if ibad > 0:
            y = y[:ibad+1,:]
        # print("y 2", y.shape)

        xs = sign*y[:,0]
        ys = y[:,1]
        cs = CubicSpline(xs, ys)

        six0 = sign*ix0
        xmin = math.ceil((xs[0]+six0)/ix)*ix - six0
        xmax = math.floor((xs[-1]+six0)/ix)*ix - six0
        xrange = np.arange(xmin,xmax,ix)
        csys = cs(xrange)
        # esy = np.stack((sign*xrange,csys), axis=1)
        esy = np.stack((sign*xrange,csys), axis=1)
        return esy


    # def sparse_result(self, xy, ix0, ix, sign, nudge=0):
    def sparse_result(self, y, ix0, ix):
        # esy = self.evenly_spaced_result(xy, ix0, ix, sign, nudge)
        origx = y[:,0].copy()
        esy = self.evenly_spaced_result(y, ix0, ix)
        if esy is None:
            return None
        # print("esy", esy.shape)
        if len(esy) == 0:
            return None
        # xs = sign*esy[:,0]
        xs = esy[:,0].copy()
        if xs[0] > xs[-1]:
            xs *= -1
            origx *= -1
        ys = esy[:,1]
        # xmin = xs[0]
        # xmax = xs[-1]

        b = np.full(xs.shape, False)
        # ignore = np.full(xs.shape, False)
        # Always ignore (delete) points that are close (< 2*ix) to the
        # start and end points
        ignore_distance = 2*ix
        ignore = (np.abs(xs-origx[0])<ignore_distance) | (np.abs(xs-origx[-1])<ignore_distance)
        # b flags points that have already been checked
        b[0] = True
        b[-1] = True
        tol = 1.
        # print("b",b.shape,b.dtype)
        # print("xrange",xrange.shape,xrange.dtype)
        # Delete as many points as possible while still keeping
        # the curve within tolerance
        while b.sum() < len(xs):
            idxs = b.nonzero()
            itx = xs[idxs]
            ity = ys[idxs]
            tcs = CubicSpline(itx,ity)
            oty = tcs(xs)
            diff = np.abs(oty-ys)
            diff[ignore] = 0.
            # print("diff", diff)
            # print("diff",diff.shape)
            midx = diff.argmax()
            # print("midx",midx)
            if b[midx]:
                break
            d = diff[midx]
            if d <= tol:
                break
            b[midx] = True

        # print(y)
        # print(esy)
        # print(ignore.nonzero(), b.nonzero())
        b[ignore] = False
        idxs = b.nonzero()
        oy = esy[idxs]
        # print("oy", oy)

        return oy


    # class function
    def createInterpolator(ar):
        interp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), ar, method='linear', bounds_error=False, fill_value=0.)
        return interp


    # Interpolate vectors that may have an ambiguous sign
    # (such as ST eigenvectors).  When interpolating, make sure
    # that adjacent vectors are consistently aligned.
    def createVectorInterpolator(ar):

        arh = ar.copy()
        arh[arh[:,:,0]<0,:] *= -1
        arv = ar.copy()
        arv[arv[:,:,1]<0,:] *= -1
        # print(arh[499:501,469:471])
        # print(arv[499:501,469:471])
        hinterp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), arh, method='linear', bounds_error=False, fill_value=0.)
        vinterp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), arv, method='linear', bounds_error=False, fill_value=0.)
        def vectorInterpolator(pts):
            # print(pts.shape)
            rh = hinterp(pts)
            dh = (rh*rh).sum(axis=-1)
            rv = vinterp(pts)
            dv = (rv*rv).sum(axis=-1)
            ro = rh.copy()
            # print(ro.shape, dv.shape, (dv>dh).shape, ro[dv>dh].shape)
            b = (dv>dh)
            '''
            if b.sum() > 0:
                # print(b)
                # print(b0)
                print(rh[b][0])
                print(dh[b][0])
                print(rv[b][0])
                print(dv[b][0])
            '''
            # if b.sum() > 0:
            #     print(b.sum())
            # print(b.shape)
            # print(b)
            # print(ro[b])
            # b = b[:,:,np.newaxis]
            # print(b.shape)
            # print(ro[b].shape)
            ro[b] = rv[b]
            '''
            if b.sum() > 0:
                print(b.sum())
                c = np.logical_and(b, (np.abs(ro[:,:,0]) > np.abs(ro[:,:,1])))
                if c.sum() > 0:
                    print(pts[c])
                    print(rh[c])
                    print(dh[c])
                    print(rv[c])
                    print(dv[c])
                    print(ro[c])
                    print
            '''
            '''
            c = pts[:,:,0] > 499
            c = np.logical_and(c, pts[:,:,0] < 501)
            c = np.logical_and(c, pts[:,:,1] < 470)
            # c = np.logical_and(c, pts[pts[:,:,0] < 501])
            c = np.logical_and(c, (np.abs(ro[:,:,1] > .5*np.abs(ro[:,:,0]))))
            if c.sum() > 0:
                print(pts[c])
                print(ro[c])
                print(rh[c])
                print(rv[c])
            '''

            return ro

        return vectorInterpolator



    def saveEigens(self, fname):
        if self.lambda_u is None:
            print("saveEigens: eigenvalues not computed yet")
            return
        lu = self.lambda_u
        lv = self.lambda_v
        vu = self.vector_u
        vv = self.vector_v
        grad = self.grad
        # print(lu.shape, lu[np.newaxis,:,:].shape)
        # print(vu.shape)
        st_all = np.concatenate((lu[:,:,np.newaxis], lv[:,:,np.newaxis], vu, vv, grad), axis=2)
        # turn off the default gzip compression
        header = {"encoding": "raw",}
        nrrd.write(str(fname), st_all, header, index_order='C')

    def loadOrCreateEigens(self, fname):
        self.lambda_u = None
        print("loading eigens")
        self.loadEigens(fname)
        if self.lambda_u is None:
            print("calculating eigens")
            self.computeEigens()
            print("saving eigens")
            self.saveEigens(fname)

    def loadEigens(self, fname):
        try:
            data, data_header = nrrd.read(str(fname), index_order='C')
        except Exception as e:
            print("Error while loading",fname,e)
            return
        print("eigens loaded")

        self.lambda_u = data[:,:,0]
        self.lambda_v = data[:,:,1]
        self.vector_u = data[:,:,2:4]
        self.vector_v = data[:,:,4:6]
        self.grad = data[:,:,6:8]
        lu = self.lambda_u
        lv = self.lambda_v
        self.isotropy = lv/lu
        self.linearity = (lu-lv)/lu
        self.coherence = ((lu-lv)/(lu+lv))**2

        # print("lambda_u", self.lambda_u.shape, self.lambda_u.dtype)
        # print("vector_u", self.vector_u.shape, self.vector_u.dtype)

        '''
        print("creating interpolators")
        self.lambda_u_interpolator = ST.createInterpolator(self.lambda_u)
        self.lambda_v_interpolator = ST.createInterpolator(self.lambda_v)
        # TODO: vector_u can abruptly change sign in areas of
        # near-vertical layers, in whichy case linear interpolation
        # of vector_u is invalid.  To avoid this, vector_u (and vector_v) 
        # should be computed, instead of interpolated, at each interpolation
        # point, from interpolated lambda_u and lambda_v.  Or do
        # the lambdas need to be computed instead of interpolated
        # as well?
        # self.vector_u_interpolator = ST.createVectorInterpolator(self.vector_u)
        # self.vector_v_interpolator = ST.createVectorInterpolator(self.vector_v)
        # self.vector_u_interpolator_ = None
        # self.vector_v_interpolator_ = None
        self.grad_interpolator = ST.createInterpolator(self.grad)
        self.isotropy_interpolator = ST.createInterpolator(self.isotropy)
        self.linearity_interpolator = ST.createInterpolator(self.linearity)
        self.coherence_interpolator = ST.createInterpolator(self.coherence)
        print("interpolators created")
        '''

