import sys
import argparse
import os
import pathlib
import math
import datetime
import csv
from scipy import sparse
import nrrd
import skimage
import torch
import torch.nn.functional as F

from stto import ST, Timer

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
from skimage.transform import PiecewiseAffineTransform
import nrrd

'''
This script takes a 2D scroll slice as input,
and produces an undeformed version of the slice as output.
It also acts as an interactive viewer so that
you can analyze the steps that were used to
produce the final result.

Quick start

0) The goal here is to create a small synthetic tif
file, and then undeform it in wind2d.py.

1) Modify synth2d.py so that it creates a synthetic
tif file in the desired directory.

2) Run synth2d.py.  Note the umbilicus coordinates
that are printed out.

3) Run wind2d.py, using command-line options to specify
the location of the tif file created by synth2d,py, and
the coordinates of the umbilicus.  For example:
    python wind2d.py /mnt/vesuvius/circle.tif --umbiicus 549,463

4) Wait for the wind2d.py window to come up, then press the
'w' key to compute the undeformed version.  Note that with
the synthetic example, this should take only a few seconds.
With real data (and using the --window 4000 option), it can
take more than an hour.  You only need to press the 'w' key
once; rerunning it won't change anything.

Now you are ready to look at the data and its overlays in the window.

You can use your mouse to do the usual pan and zoom.

You can use the up and down arrows to view different
overlays on top of the original data, or to view the
undeformed data.  Note that the info bar in the bottom
gives the name of the current overlay.

On some overlays you can see cyan lines, each with a dot on one end.
These lines show values computed from the computed structure tensors,
and should point in a direction normal to the surfaces.

You can use the 'v' key to show/hide these lines.

One of the overlays is called coh (coherence); it shows
the local coherence of the data, calculated from the structure
tensors.  This overlay can be viewed even before you hit the
'w' key.

Red dots should be visible; these show the points that are used
in the warping (undeform) transform.  Use the 'Shift-v' key combo
to change the size of (or hide) these dots.

The 'c' key lets you cycle through different colormaps for
the current overlay.

The 'i' key toggles whether the colormap of the current overlay is
interpolated linearly or by nearest neighbor (on many of
the available colormaps, the difference is not visible).

The 'q' key will exit the program.

Here are some general concepts.

In order to compute the undeform transform, wind2d proceeds
through several steps.  After each step, the newly calculated 
result is saved in the cache directory.  When wind2d is rerun,
at each step, if it finds the corresponding file in the
cache, it will reload that file instead of recomputing it.

If you don't want any of the cache files to be reused, use
the --no_cache command line option.  Or if you want to
reuse only certain files, delete (or hides) the ones that
you don't want reused.

At each step, a number of overlays (which you just learned
how to view) are created.  The number of overlays created
depends on whether the --diagnostics flag appears on
the command line.

In summary, the processing steps are:

When the image is loaded, compute structure tensors
and associated values at each point of the image.
One of these values is u, the apparent normal.

Compute r, the current radius of each point (distance
from the umbilicus), and theta, the current angle
of each point around the umbilicus.

Using r and u, compute r0, the pre-deformation radius
(the distance that each point would have been from the
umbilicus before the scroll was crushed).

Where necessary, multipy the u vector by -1 to better
align u with the normals implied by grad r0.

Using the adjusted u, compute r1, a refined value for the
pre-deformation radius.

Replace u (the original surface normal) by n, a surface
normal computed from grad r1.

Using r1 and n, compute theta0, the pre-deformation angle
(the angle each point would have had around the umbilicus prior
to deformation).

Adjust r1 using a value derived from theta0.

Using the adjusted r1, and n, compute
theta1, a refined value for the pre-deformation angle.

Using r, theta, r1, and theta1, compute a mapping from the
current (deformed) coordinate system to the pre-deformation
(undeformed) coordinate system; apply this mapping to
undeform the image.


In detail, the processing steps are:

When the image is loaded:

The structure tensors and related 
values are computed (or loaded from the cache).  The most 
important value is called u in our notation; this is a two-component
vector that is defined at each point of the image.  In
areas of the image where sheets are clearly defined, the u
vector points in the direction of the sheet's normal.  

The u vector always has a length of 1.  However, 
u has a sign ambiguity.  The u vector at each 
location in the image might point towards the umbilicus, 
or away from it; this can vary from location to location.

Another value that is computed from the structure tensors
is called coherence.  It has a value between 0.0 and 1.0,
and represents how well-defined the sheets are.  If the
sheets are not clearly distinguishable, the coherence
will be close to 0, and the u vector cannot be relied
upon to provide a valid value.

See the comments in st.py for links to a number of
papers about structure tensors.

After 'w' is pressed:

At each point of the image, compute the current
(post-deformation) value of r, the distance (radius) 
from the umbilicus.  This is a very quick and easy 
computation, since it is pure geometry.  Likewise,
compute the current value of theta, the angle
of the point around the umbilicus, relative
to the positive x axis.  Again, this is pure
geometry.

A note about decimation.  Most of the following
operations are fairly compute-intensive.  One way to
speed them up is to limit the number of points that
are involved in the computation.  The --decimation
command-line flag sets the amount of decimation,
which is applied both horizontally and vertically.
So for instance, if the decimation factor is 8,
only every 8th point, in every 8th row, will be used.
This means that in the image, only 1 out of every
64 points will be used.  In practice, a decimation
factor of 8 seems to work well enough.

At each point (strictly speaking, at every
point remaining after decimation) compute the value r0, 
which is defined as the pre-deformation radius (distance to 
the umbilicus).  In other words, if we had access to the scroll before
it was deformed, and were able to measure the distance
from the umbilicus to the given sheet, r0 is the value
that we would measure.

The function that computes r0 takes two inputs: the
u vector at each point, and the current r (post-deformation
radius) at each point.  The values for r0 (recall that
there is a value at each point of the image) are computed by
applying the least square method to an over-determined set
of linear equations.

These equations need to take into account the sign ambiguity
in u.  They also use the current radius r for stability.
Let us define r0' by: r0 = r + r0' 
so r0' is the difference between current-day r and our
sought value, r0.
We want the gradient of r0 to be aligned with the structure
tensor normals; one way of expressing this
is u cross (grad r0) = 0
(that is, the gradient of r0, which gives the sheet
normals, should be aligned with the u vectors.)
Substituting for r0', we get:
u cross (grad r0') = -u cross (grad r0).
Then denote the u cross grad operator by matrix A,
the sought-for quantity r0' by vector x, and 
the right-hand side of the equation, which is constant,
by vector b.
Now we have the classic equation Ax=b, which can be
solved numerically by a least-squares approach.

There are a few enhancements that are used in practice.
For instance, the coherency is applied at each point
as a weighting factor, so that reliable u values are
given a higher weight.  Some smoothing equations
are used to ensure that r0' is smooth in
areas where it is otherwise under-determined.
And r0 at the umbilicus point is constrained to
be zero.

Once r0 is computed, it is possible to resolve the
sign ambiguity in the u vector.  The u vector is
supposed to represent the outward-pointing normal
of the sheet, but the value computed using the structure tensor
is ambiguous in sign, so the correct sign (direction) of the
normal must be determined using other information.
Before the scroll was crushed, "outward"
could be determined by finding which of the two directions
pointed away from the umbilicus.  But due to deformation,
the sheet may have since been twisted to such an extent that 
its outward normal now points towards the present-day umbilicus
point.  Thus we cannot use the umbilicus location to help us.

Fortunately, the r0 value lets us resolve the sign ambiguity.
The gradient of r0, grad r0, points in the direction of the
outward normal.  So the computation step here is: at each
point of the image, set the sign of the u vector such that 
the value of (grad r0) dot u is positive.

Now that the sign ambiguity in u has been resolved, we use
u to refine the value of the pre-deformation radius.
To compute this new radius value, r1, we use another
set of over-determined linear equations:

u dot (grad r1) = 1
u cross (grad r1) = 0  (this was also used to compute r0)

The first set of equations, with the dot product, should help
ensure that the change in the pre-deformation radius with
respect to the post-deformation distance averages out to be 1.
That is, if the crushing decreased the present-day radius in
some places, it should have increased it in other places.

(Footnote: This assumption is not strictly true; an 
improved version of the algorithm might use a function f(r1) 
instead of a constant value of 1, where f(r1)
averages out to about 1.  Of course, inserting f(r1) into
this equation would make it non-linear.)

These equations can be put into the form Ax=b, and solved
by the least-squares method to yield r1

As with the r0 computation, the formulas leave out a couple
of details, such as weighting by the coherence value, and a
smoothing term.  In the r1 case, the smoothing term does
not seek to push the local r1 gradient towards 0 (which would
be fighting against the dot product formula above); the smoothing
term instead tries to push the second derivative of r1, taken in the
x and y directions, towards zero.  As with the r0 computation,
r1 at the umbilicus is constrained to be zero.

Note, by the way, that r0 is not directly used in the computation
of r1.  However, r0 played an indirect role, in that it was
used to resolve the sign ambiguities of the u vector.

Once r1 is computed, an adjustment factor is calculated.
Averaged over the entire image, r1 / r should equal 1.  The
actual ratio is calculated (in the center part of the image),
and r1 is multiplied by whatever factor is needed to
bring the average r1 / r ratio to 1.

At this point it is possible to dispense with the u vector.
The problem with the u vector is that it is reliable only
in areas of high coherence.  Fortunately, now that r1 has
been calculated, we have a replacement.  The vector 
n = normal(grad r1) can be used in place of u.  (For historical
reasons, the vector that here called n is called th0uvec
in the code).

Theta0, the pre-deformation angle of each point around the
umbilicus, is calculated next.  Imagine that we have an
undeformed scroll, which contains a sheet of radius r.  At a 
certain point on the sheet, we can draw the normal vector n relative 
to the sheet, and the tangent vector t.  n and t are perpendicular
to each other.  Although all points on this sheet are at
the same radius, r, they have different theta values (angles).
At the point where we have defined normal n and tangent t, 
we know that (r1 grad theta) dot t equals 1; this is simple geometry.  
But t is inconvenient to use, so we'll replace 
the dot product with t by the
cross product with n.  So: (r1 grad theta) cross n = 1.
On the post-deformation image, this formula is still valid,
because the sheet conserves its length during the deformation.

So theta0 (th0 for short) at every point is the solution 
to the set of equations:
(r1 grad th0) cross n = 1

In practice, there are a few other equations to try and
keep the solution smooth.

But as before, the equations are linear and can be solved
by the least-squares method.

A complication occurs along the line (x < umbx, y = umby).
In the pre-deformation scroll, theta jumps in value from
pi (just above the line) to -pi (just below the line).
This line is called a "branch cut".
In the case of the post-deformation scroll, the th0 values
either side of the branch cut are probably not exactly pi and -pi, 
but they do increase in value by the same amount: 2 pi.

In order to take the branch cut discontinuity into account, the
grad calculations need to include, in the y direction, 
a term of 2 pi to compensate.

After theta0 is computed, the value (r1 grad th0) cross n
is computed at the points near the center of the image.  A
multiplicative factor is applied to r1 to bring 
average of this value to 1.

(Note for improvement: r1 should be adjusted in a more 
sophisticated way.  For instance, a new value, r2, could
be calculated, where r2 is a monotonically increasing
function of r1.  The function mapping r1 to r2 would be
chosen so as to bring the (r2 grad th0) cross n value
as close to 1 as possible.)

After r1 is adjusted, the function used to calculate theta0
is called again, this time with the adjusted r1.  The result
is denoted as theta1 (th1 for short).

We now have everything we need to map the image from 
its current appearance to its original undeformed state.
At each point of the image we know its pre-deformation radius,
r1, and its pre-deformation angle, th1.  The r1 and th1 values
are easily converted to pre-deformation x and y locations,
so each pixel of the current-day image can be traced back to
its original location.

In practice, the image mapping is done using a decimated set
of control points (the --warp_decimation flag controls this
value), and the warping is done using the piecewise affine transform
function provided by scikit-image.

(The points used in the warping are drawn as red dots on the
pre- and post-deformation images).

So that is the process.

There are a number of places where the process can go wrong;
here are a couple.

First of all, the scrolls were deformed in 3D, not 2D.  So a
single present-day 2D slice actually contains information from
what would have been a number of slices in the pre-deformation
scroll.  Conclusion: the undeform process needs to be applied
in 3D.

Second, the undeform warping is non-physical, in that it doesn't
take into account the fact that papyrus sheets deform in a different
way than the space between the sheets.  Sheets can bend, but they
do not change thickness.  On the other hand, when a scroll is
deformed, the space between the sheets can easily change thickness
(volume)

Consdier a region of the scroll where the papyrus sheets were squashed
together during deformation.  When this region is transformed back
to its "undeformed" state, the sheets should maintain their thickness;
only the air gaps between the sheets should expand.  
But using the algorithm presented here, the sheets and the space
between them will become thicker to the same extent, in a way that does 
not make physical sense.  Just as the sheets can become thicker
in a non-physical way, point-like noise in the original 
image can get stretched into linear noise as the result of 
the "undeform" operation.

But overall, the algorithm, despite its flaws, works better
than I had expected.

'''
# Adapted rom https://github.com/scikit-image/scikit-image/issues/6864
# Modified to use less memory
class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        print("starting __call__")
        coords = np.asarray(coords)
        print("coords", coords.shape)



        raw_affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))])
        print("created raw_affines", raw_affines.shape)

        # process only "step" coords at a time
        # in order not to use too much memory
        step = 10000000
        result = np.zeros((coords.shape[0], 3), dtype=coords.dtype)
        for start in range(0, coords.shape[0], step):
            end = min(start+step, coords.shape[0])
            lcoords = coords[start:end]
            simplex = self._tesselation.find_simplex(lcoords)
            pts = np.c_[lcoords, np.ones((end-start, 1))]
            result[start:end] = np.einsum("ij,ikj->ik", pts, raw_affines[simplex])
            result[start:end][simplex == -1, :] = -1
            # print(" ",start)


        print("leaving __call__")
        return result

    def old__call__(self, coords):
        print("starting __call__")
        coords = np.asarray(coords)
        print("coords", coords.shape)

        simplex = self._tesselation.find_simplex(coords)
        print("found simplexes", simplex.shape)

        '''
        affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))]
        )[simplex]
        '''
        raw_affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))])
        # affines = raw_affines[simplex]

        # print("created affines", len(self.affines), raw_affines.shape, affines.shape)
        print("created raw_affines", len(self.affines), raw_affines.shape)

        pts = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ij,ikj->ik", pts, raw_affines[simplex])
        print("did einsum", pts.shape, result.shape, result.flags['C_CONTIGUOUS'])
        result[simplex == -1, :] = -1

        return result

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
        self.viewer.diagnostics = parsed_args.diagnostics
        window_width = parsed_args.window
        decimation = parsed_args.decimation
        self.viewer.decimation = decimation
        self.viewer.warp_decimation = parsed_args.warp_decimation
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

        '''
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
        '''
        t = Timer()
        self.st.computeTensors()
        t.time("computed tensors")

        # mask = self.viewer.createMask()

        self.viewer.setOverlayDefaults()
        self.viewer.saveCurrentOverlay()

        self.viewer.overlay_name = "coherence"
        # self.viewer.overlay_data = self.st.coherence.cpu().numpy().copy()
        self.viewer.overlay_data = self.st.coherence
        # self.viewer.overlay_data *= mask
        self.viewer.overlay_colormap = "viridis"
        self.viewer.overlay_interpolation = "linear"
        self.viewer.overlay_maxrad = 1.0
        self.viewer.saveCurrentOverlay()

        '''
        self.viewer.overlay_name = "scaled"
        # self.viewer.overlay_data = self.st.coherence.copy()
        # self.viewer.overlay_data *= mask
        im = self.viewer.image
        scale = 2.
        rescaled = cv2.resize(im, (int(scale*im.shape[1]), int(scale*im.shape[0])), interpolation=cv2.INTER_LINEAR)
        self.viewer.overlay_data = rescaled
        self.viewer.overlay_colormap = "gray"
        self.viewer.overlay_interpolation = "linear"
        self.viewer.overlay_maxrad = 1.0
        self.viewer.overlay_alpha = 1.0
        self.viewer.overlay_scale = scale
        self.viewer.saveCurrentOverlay()
        '''

        self.viewer.getNextOverlay()
        self.timestr = datetime.datetime.now().isoformat()
        print("timestamp", self.timestr)
        self.setWindowTitle("wind2d  "+self.timestr)

    # Not used at this time
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
    def __init__(self, name, data, maxrad, colormap="viridis", interpolation="linear", alpha=None, scale=None):
        self.name = name
        self.data = data
        self.colormap = colormap
        self.interpolation = interpolation
        self.maxrad = maxrad
        self.alpha = alpha
        self.scale = scale

    @staticmethod
    def createFromOverlay(overlay):
        no = Overlay(overlay.name, overlay.data, overlay.maxrad, overlay.colormap, overlay.interpolation, overlay.alpha)
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
        self.warp_dot_size = 3
        self.umb = None
        self.overlays = []
        self.overlay_data = None
        self.overlay_name = ""
        self.decimation = 1
        self.overlay_colormap = "viridis"
        self.overlay_interpolation = "linear"
        self.overlay_maxrad = None
        self.overlay_alpha = None
        self.overlay_scale = None
        self.overlay_defaults = None
        self.src_dots = None
        self.dest_dots = None
        self.cpu = torch.device('cpu')
        self.gpu = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
        self.safe_gpu = self.gpu
        if str(self.gpu) == "xpu":
            self.safe_gpu = self.cpu


    def setOverlayDefaults(self):
        self.overlay_defaults = Overlay("", None, self.overlay_maxrad, self.overlay_colormap, self.overlay_interpolation)

    def makeOverlayCurrent(self, overlay):
        self.saveCurrentOverlay()
        self.overlay_data = overlay.data
        self.overlay_name = overlay.name
        self.overlay_colormap = overlay.colormap
        self.overlay_interpolation = overlay.interpolation
        self.overlay_maxrad = overlay.maxrad
        self.overlay_alpha = overlay.alpha
        self.overlay_scale = overlay.scale

    def setOverlayByName(self, name):
        o = Overlay.findItemByName(self.overlays, name)
        if o is None:
            return
        self.makeOverlayCurrent(o)


    def getOverlayByName(self, name):
        o = Overlay.findItemByName(self.overlays, name)
        return o

    def saveCurrentOverlay(self):
        name = self.overlay_name
        no = Overlay(name, self.overlay_data, self.overlay_maxrad, self.overlay_colormap, self.overlay_interpolation, self.overlay_alpha, self.overlay_scale)
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
            # self.solveWindingOneStep()
            self.undeform()
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
            if e.modifiers() & Qt.ShiftModifier:
                self.warp_dot_size = (self.warp_dot_size+1)%4
                # 
            else:
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
            # .cpu() to prevent xpu crash
            stxt += "%.2f "%imi.cpu()
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
        ixys = np.zeros(wxys.shape, dtype=np.float32)
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
            image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED).astype(np.float32)
            image /= 65535.
        except Exception as e:
            print("Error while loading",fname,e)
            return
        self.image = torch.from_numpy(image).to(self.gpu)
        self.image_mtime = fname.stat().st_mtime
        self.setDefaults()
        self.umb = np.array((image.shape[1]/2, image.shape[0]/2))
        self.drawAll()

    # set is_cross True if op is cross product, False if
    # op is dot product
    # Creates a sparse matrix that represents the operator
    # vec2d cross grad  or  vec2d dot grad
    # depending on the is_cross flag
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
        # print("diag3", diag3.shape, diag3.dtype)
        # clean up memory
        n1d_flat = None

        vec2d_flat = vec2d[:nr, :nc].reshape(-1, 2)
        # print("vec2d_flat", vec2d_flat.shape)

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
        # print("ddx", ddx.shape, ddx.dtype)

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
        # print("ddy", ddy.shape, ddy.dtype)

        # clean up memory
        dy0 = None
        dy1 = None

        # print("ddx,ddy", ddx.max(axis=0), ddy.max(axis=0))

        uxg = np.concatenate((ddx, ddy), axis=0)
        # print("uxg", uxg.shape, uxg.dtype, uxg[:,0].max(), uxg[:,1].max())
        ddx = None
        ddy = None
        sparse_uxg = sparse.coo_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(nrf*ncf, nrf*ncf))
        # sparse_uxg = sparse.csc_array((uxg[:,2], (uxg[:,0], uxg[:,1])), shape=(nrf*ncf, nrf*ncf))
        # print("sparse_uxg", sparse_uxg.shape, sparse_uxg.dtype)
        return sparse_uxg


    @staticmethod
    def sparseDiagonal(shape):
        nrf, ncf = shape
        ix = np.arange(nrf*ncf)
        ones = np.full((nrf*ncf), 1.)
        sparse_diag = sparse.coo_array((ones, (ix, ix)), shape=(nrf*ncf, nrf*ncf))
        return sparse_diag

    # create a sparse matrix that represents the 2D grad operator.
    # if interleave is true, the output is a single sparse matrix
    # with interleaved x and y components of the grad.
    # if interleave is false, separate sparse matrices are created
    # for the x and y components of the grad
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
            # print("grad", grad.shape, grad.min(axis=0), grad.max(axis=0), grad.dtype)
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
        print("rad0 smoothing", smoothing_weight)
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
        # print("A", A.shape, A.dtype)

        b = -sparse_u_cross_grad @ basew.flatten()
        b[basew.size:] = 0.
        x = self.solveAxEqb(A, b)
        out = x.reshape(basew.shape)
        out += basew
        # print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (vecu.shape[1], vecu.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out


    def solveRadius1(self, rad0, smoothing_weight, cross_weight):
        st = self.main_window.st
        print("rad1 smoothing", smoothing_weight, "cross_weight", cross_weight)
        decimation = self.decimation
        # print("decimation", decimation)

        icw = 1.-cross_weight

        uvec = st.vector_u
        coh = st.coherence.copy()

        # TODO: for testing
        # mask = self.createMask()
        ## coh = coh.copy()*mask
        # coh *= mask

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
        # print("sgx", sgx.shape, "hxx", hxx.shape, "hxy", hxy.shape)

        # print("grad", sparse_grad.shape, "hess", sparse_hess.shape)
        sparse_umb = ImageViewer.sparseUmbilical(shape, np.array(self.umb)/decimation)
        # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*sparse_grad, sparse_umb))
        # sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, smoothing_weight*hxy, sparse_umb))
        sparse_all = sparse.vstack((icw*sparse_u_dot_g, cross_weight*sparse_u_cross_g, smoothing_weight*hxx, smoothing_weight*hyy, sparse_umb))

        A = sparse_all
        # print("A", A.shape, A.dtype)

        b = np.zeros((A.shape[0]), dtype=np.float64)
        # NOTE multiplication by decimation factor
        b[:rad0.size] = 1.*coh.flatten()*decimation*icw
        x = self.solveAxEqb(A, b)
        out = x.reshape(rad0.shape)
        # print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            out = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out

    def solveTheta(self, rad, uvec, coh, dot_weight, smoothing_weight, theta_weight):
        print("theta dot_weight", dot_weight, "smoothing", smoothing_weight, "theta_weight", theta_weight)
        st = self.main_window.st
        decimation = self.decimation
        # print("decimation", decimation)

        theta = self.createThetaArray()
        oldshape = rad.shape
        coh = coh[:,:,np.newaxis]
        weight = coh.copy()
        # TODO: for testing only!
        # mask = self.createMask()
        # coh *= mask
        # weight = coh*coh*coh
        # weight[:,:] = 1.
        wuvec = weight*uvec
        rwuvec = rad[:,:,np.newaxis]*wuvec
        if decimation > 1:
            wuvec = wuvec[::decimation, ::decimation, :]
            theta = theta[::decimation, ::decimation]
            weight = weight[::decimation, ::decimation, :]
            # Note that rad is divided by decimation
            rad = rad.copy()[::decimation, ::decimation] / decimation
            # recompute rwuvec to account for change in rad
            rwuvec = rad[:,:,np.newaxis]*wuvec
        shape = theta.shape
        sparse_grad = ImageViewer.sparseGrad(shape, rad)
        sparse_u_cross_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=True)
        sparse_u_dot_g = ImageViewer.sparseVecOpGrad(rwuvec, is_cross=False)
        sparse_theta = ImageViewer.sparseDiagonal(shape)
        sparse_all = sparse.vstack((sparse_u_cross_g, dot_weight*sparse_u_dot_g, smoothing_weight*sparse_grad, theta_weight*sparse_theta))
        # print("sparse_all", sparse_all.shape)

        umb = np.array(self.umb)
        decimated_umb = umb/decimation
        iumb = decimated_umb.astype(np.int32)
        # bc: branch cut
        bc_rad = rad[iumb[1], :iumb[0]]
        bc_rwuvec = rwuvec[iumb[1], :iumb[0]]
        bc_dot = 2*np.pi*bc_rwuvec[:,1]
        bc_grad = 2*np.pi*bc_rad
        bc_cross = 2*np.pi*bc_rwuvec[:,0]
        bc_f0 = shape[1]*iumb[1]
        bc_f1 = bc_f0 + iumb[0]

        b_dot = np.zeros((sparse_u_dot_g.shape[0]), dtype=np.float64)
        b_dot[bc_f0:bc_f1] += bc_dot.flatten()
        b_cross = weight.flatten()
        b_cross[bc_f0:bc_f1] += bc_cross.flatten()
        b_grad = np.zeros((sparse_grad.shape[0]), dtype=np.float64)
        b_grad[2*bc_f0+1:2*bc_f1+1:2] += bc_grad.flatten()
        b_theta = theta.flatten()
        b_all = np.concatenate((b_cross, dot_weight*b_dot, smoothing_weight*b_grad, theta_weight*b_theta))
        # print("b_all", b_all.shape)

        x = self.solveAxEqb(sparse_all, b_all)
        out = x.reshape(shape)
        # print("out", out.shape, out.min(), out.max())
        if decimation > 1:
            outl = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_LINEAR)
            outn = cv2.resize(out, (uvec.shape[1], uvec.shape[0]), interpolation=cv2.INTER_NEAREST)
        # return outl,outn
        return outl

    # given radius and theta, compute x and y
    def xformXY(self, rad, theta):
        x = rad*np.cos(theta)
        y = rad*np.sin(theta)
        xy = np.stack((x,y), axis=2)
        print(x.shape, y.shape, xy.shape)
        return xy

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

    def printArray(self, arr, ixy):
        la = arr[ixy[1]:ixy[1]+2, ixy[0]:ixy[0]+2]
        print(la)
        print(la[0,1]-la[0,0])
        print(la[1,0]-la[0,0])

    @staticmethod
    def solveAxEqb(A, b):
        print("solving Ax = b", A.shape, b.shape)
        At = A.transpose()
        AtA = At @ A
        # print("AtA", AtA.shape, sparse.issparse(AtA))
        # print("ata", ata.shape, ata.dtype, ata[ata!=0], np.argwhere(ata))
        asum = np.abs(AtA).sum(axis=0)
        # print("asum", np.argwhere(asum==0))
        Atb = At @ b
        # print("Atb", Atb.shape, sparse.issparse(Atb))

        lu = sparse.linalg.splu(AtA.tocsc())
        # print("lu created")
        x = lu.solve(Atb)
        print("x", x.shape, x.dtype, x.min(), x.max())
        return x

    # given an array filled with radius values, align
    # the structure tensor u vector with the gradient of
    # the radius
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

    '''
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
    '''

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
        print("array is C type", arr.flags['C_CONTIGUOUS'])
        nrrd.write(str(fname), arr, index_order='C')

    def loadOrCreateArray(self, part, fn, save_tiff=False):
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
            if save_tiff:
                print("saving tiff", part)
                tname = self.cache_file_base.with_name(self.cache_file_base.name + part + ".tif")
                cv2.imwrite(str(tname), (arr*65535).astype(np.uint16))
        return arr

    def createRadiusArray(self):
        umb = self.umb
        im = self.image
        # iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        # dtype=float32 to prevent xpu crash
        x0s = torch.arange(im.shape[0], device=self.gpu, dtype=torch.float32)
        x1s = torch.arange(im.shape[1], device=self.gpu, dtype=torch.float32)
        iys, ixs = torch.meshgrid((x0s, x1s), indexing='ij')
        print("ixs", ixs.shape, ixs.dtype, ixs.device)
        # print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])
        rad = torch.sqrt(radsq)
        # print("rad", rad.shape)
        return rad

    def createThetaArray(self):
        umb = self.umb
        umb[1] += .5
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        # print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        theta = np.arctan2(iys-umb[1], ixs-umb[0])
        # print("theta", theta.shape, theta.min(), theta.max())
        return theta

    # given a radius array, create u vectors from the
    # normalized gradients of that array.
    def synthesizeUVecArray(self, rad):
        gradx, grady = self.computeGrad(rad)
        uvec = np.stack((gradx, grady), axis=2)
        # print("suvec", uvec.shape)
        luvec = np.sqrt((uvec*uvec).sum(axis=2))
        lnz = luvec != 0
        # print("ll", uvec.shape, luvec.shape, lnz.shape)
        uvec[lnz] /= luvec[lnz][:,np.newaxis]
        coh = np.full(rad.shape, 1.)
        coh[:,-1] = 0
        coh[-1,:] = 0
        return uvec, coh

    # "E" stands for ellipse.  Create a synthetic radius array
    def createERadiusArray(self, ecc):
        umb = self.umb
        im = self.image
        iys, ixs = np.mgrid[:im.shape[0], :im.shape[1]]
        # print("mg", ixs.shape, iys.shape)
        # iys gives row ids, ixs gives col ids
        radsq = (ixs-umb[0])*(ixs-umb[0])+(iys-umb[1])*(iys-umb[1])/(ecc*ecc)
        rad = np.sqrt(radsq)
        # print("rad", rad.shape)
        return rad

    # "E" stands for ellipse.  Create a synthetic theta array
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
        # print("r factor", tlen/(2*np.pi))
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
        ctheta = np.arctan2((iys-umb[1])/ecc, ixs-umb[0])
        etheta = np.interp(ctheta, cth, eth)
        return etheta

    # For debugging: create a mask to be applied to
    # the input image or to the coherency array.
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

    def undeform(self):
        im = self.image
        if im is None:
            return

        rad = self.createRadiusArray()

        self.overlay_data = rad
        self.overlay_name = "rad"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

    # This is where the undeform operation takes place.
    # The name of the function doesn't really make sense.
    def solveWindingOneStep(self):
        im = self.image
        if im is None:
            return

        rad = self.createRadiusArray()

        smoothing_weight = .1
        rad0 = self.loadOrCreateArray(
                "_r0", lambda: self.solveRadius0(rad, smoothing_weight))

        self.alignUVVec(rad0)

        # copy uvec AFTER it has been aligned
        st = self.main_window.st
        uvec = st.vector_u.copy()
        coh = st.coherence.copy()

        self.overlay_data = rad0
        self.overlay_name = "rad0"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

        # hess
        smoothing_weight = .2
        # grad
        # smoothing_weight = .01
        cross_weight = 0.95
        rad1 = self.loadOrCreateArray(
                "_r1", lambda: self.solveRadius1(rad0, smoothing_weight, cross_weight))

        # TODO: For testing!
        # ecc = .5
        # rad1 = self.createERadiusArray(ecc)
        # # r factor
        # rad1 *= .7709

        # find which locations in the image have
        # high coherency and a low radius
        cargs = np.argsort(coh.flatten())
        min_coh = coh.flatten()[cargs[len(cargs)//4]]
        rargs = np.argsort(rad1.flatten())
        max_rad1 = rad1.flatten()[rargs[len(rargs)//4]]

        crb = np.logical_and(coh > min_coh, rad1 < max_rad1)
        crb = np.logical_and(crb, rad > 0)
        rs = rad[crb]
        r1s = rad1[crb]
        # using the locations found above, find the average
        # ratio between r1 (pre-deformation radius) and
        # current radius.
        mr1r = np.median(r1s/rs)
        print("mr1r", mr1r)
        # apply this ratio as a correction to r1
        rad1 /= mr1r

        self.overlay_data = rad1
        self.overlay_name = "rad1"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()

        if self.diagnostics:
            gradx, grady = self.computeGrad(rad1)
            gdu = gradx*uvec[:,:,0] + grady*uvec[:,:,1]
            self.overlay_name = "rad1 gdu"
            self.overlay_data = gdu
            self.overlay_maxrad = 4.
            self.overlay_colormap = "hsv"
            self.overlay_interpolation = "linear"
            self.saveCurrentOverlay()

        self.setOverlayByName("rad1")

        dot_weight = .001
        smoothing_weight = .4
        theta_weight = .0001

        # testing
        # ecc = .5
        # rad1 = self.createERadiusArray(ecc)
        # r factor
        # rad1 *= .7709

        # create synthetic u vector and coherence
        # using the gradient of r1
        # (the coherence will be 1 almost everywhere,
        # except at the edges)
        th0uvec, th0coh = self.synthesizeUVecArray(rad1)

        theta0 = self.loadOrCreateArray(
                  "_th0", lambda: self.solveTheta(rad1, th0uvec, th0coh, dot_weight, smoothing_weight, theta_weight))

        '''
        # testing
        ecc = .5
        theta0 = self.createEThetaArray(ecc)
        rad1 = self.createERadiusArray(ecc)
        # r factor
        rad1 *= .7709
        o = Overlay.findItemByName(self.overlays, "rad1")
        o.data = rad1
        '''

        self.overlay_data = theta0
        self.overlay_name = "theta0"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = 3.
        self.saveCurrentOverlay()

        gradx, grady = self.computeGrad(theta0)
        gradx *= rad1
        grady *= rad1
        grad = np.sqrt(gradx*gradx+grady*grady)

        # u cross (rad1 grad theta0)
        gxu = -gradx*th0uvec[:,:,1] + grady*th0uvec[:,:,0]
    
        if self.diagnostics:
            self.overlay_data = gradx
            self.overlay_name = "rad1 grad theta0 x"
            self.overlay_colormap = "hsv"
            self.overlay_interpolation = "linear"
            self.overlay_maxrad = 2.
            # self.saveCurrentOverlay()
    
            self.overlay_data = grady
            self.overlay_name = "rad1 grad theta0 y"
            # self.saveCurrentOverlay()
    
            self.overlay_data = grad
            self.overlay_name = "rad1 grad theta0"
            self.overlay_colormap = "hsv"
            self.overlay_interpolation = "linear"
            # self.saveCurrentOverlay()
    
            self.overlay_name = "th0 gxu"
            self.overlay_data = gxu
            self.overlay_colormap = "hsv"
            self.overlay_interpolation = "linear"
            self.saveCurrentOverlay()
    
            gdu = gradx*th0uvec[:,:,0] + grady*th0uvec[:,:,1]
            self.overlay_name = "th0 gdu"
            self.overlay_data = gdu
            self.overlay_colormap = "hsv"
            self.overlay_interpolation = "linear"
            # self.saveCurrentOverlay()

        cargs = np.argsort(coh.flatten())
        min_coh = coh.flatten()[cargs[len(cargs)//4]]
        rargs = np.argsort(rad1.flatten())
        max_rad1 = rad1.flatten()[rargs[len(rargs)//4]]

        crb = np.logical_and(coh > min_coh, rad1 < max_rad1)

        # gxu (defined above) should average out to 1.0 over the image;
        # find the deviation and apply it as a correction factor to rad1
        mgxu = np.median(gxu[crb])

        print("mgxu", mgxu)

        rad1 /= mgxu

        theta1 = self.loadOrCreateArray(
                  "_th1", lambda: self.solveTheta(rad1, th0uvec, th0coh, dot_weight, smoothing_weight, theta_weight))

        self.overlay_data = theta1
        self.overlay_name = "theta1"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = 3.
        self.saveCurrentOverlay()

        gradx, grady = self.computeGrad(theta1)
        gradx *= rad1
        grady *= rad1
        gxu = -gradx*th0uvec[:,:,1] + grady*th0uvec[:,:,0]

        if self.diagnostics:
            self.overlay_name = "th1 gxu"
            self.overlay_data = gxu
            self.overlay_colormap = "hsv"
            self.overlay_maxrad = 2.
            self.overlay_interpolation = "linear"
            self.saveCurrentOverlay()

        mgxu = np.median(gxu[crb])
        print("mgxu", mgxu)

        self.setOverlayByName("theta1")

        rad = self.createRadiusArray()
        theta = self.createThetaArray()

        # present day xy of image pixels
        src = self.xformXY(rad, theta)

        '''
        self.overlay_data = src[:,:,0]
        self.overlay_name = "src x"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()
        self.overlay_data = src[:,:,1]
        self.overlay_name = "src y"
        self.saveCurrentOverlay()
        '''

        # pre-deformation xy of image pixels
        dest = self.xformXY(rad1, theta1)

        '''
        self.overlay_data = dest[:,:,0]
        self.overlay_name = "dest x"
        self.overlay_colormap = "tab20"
        self.overlay_interpolation = "nearest"
        self.overlay_maxrad = self.umb_maxrad
        self.saveCurrentOverlay()
        self.overlay_data = dest[:,:,1]
        self.overlay_name = "dest y"
        self.saveCurrentOverlay()
        '''

        # deci = self.decimation
        # deci = 32
        # deci = 128
        # ndots = 512

        # "fast" and "slow" tested with 
        # --warp_decimation 32
        # resize = .75
        # scale = 2.

        # slice 2000
        # resize = .75

        # slice 3000
        # resize = .5

        # scale = 2.
        ishape = self.image.shape
        # deci = int(math.sqrt(ishape[0]*ishape[1]/ndots))

        # apply decimation and shift
        deci = self.warp_decimation
        ndots = ishape[0]*ishape[1] / (deci*deci)
        print("deci", deci, "ndots", ndots)
        srcd = src[::deci, ::deci].reshape(-1,2)
        b = np.logical_and(srcd[:,0] <= 0, np.abs(srcd[:,1]) <= self.decimation)
        srcd = srcd[~b]
        srcd += self.umb
        self.src_dots = srcd.copy()


        # srcd = srcd[100:150,100:150]
        # print(srcd[0])
        # apply decimation and shift
        # need to apply additional factors to make 
        # sure un-deformed image is fully visible
        destd = dest[::deci, ::deci].reshape(-1,2)
        destd = destd[~b]
        print("destd min max", destd.min(axis=0), destd.max(axis=0))
        dmin = destd.min(axis=0)
        dmax = destd.max(axis=0)
        dmin[dmin==0] = .01
        dmax[dmax==0] = .01
        imin = -self.umb
        imax = imin + np.array((ishape[1],ishape[0]))
        print("dmin, dmax", dmin, dmax)
        print("imin, imax", imin, imax)
        rmin = np.abs(imin/dmin).min()
        rmax = np.abs(imax/dmax).min()
        resize = min(rmin, rmax)*.95
        scale = 1./resize
        scale = min(scale, 2.)
        print("resize", resize, "scale", scale)

        destd *= resize
        destd += self.umb
        # print("destd min max", destd.min(axis=0), destd.max(axis=0))
        self.dest_dots = destd.copy()
        destd *= scale
        # print(destd[8])
        # destd = destd[100:150,100:150]
        '''
        print("estimating", srcd.shape, destd.shape)
        # xform.estimate(srcd, destd)
        xform = skimage.transform.PiecewiseAffineTransform()
        xform.estimate(destd, srcd)
        print("warping")
        oshape = (int(ishape[0]*scale), int(ishape[1]*scale))
        oim = skimage.transform.warp(self.image, xform, output_shape=oshape)
        '''
        '''
        ndots = 512
        resize = .75
        scale = 2.
        oim = self.loadOrCreateArray(
                  "_ud", lambda: self.warpImage(rad, theta, rad1, theta1, ndots, resize, scale))
        '''
        oim = self.loadOrCreateArray(
                  "_ud", lambda: self.warpImage(srcd, destd, scale), save_tiff=True)
        print("finished warping")

        self.overlay_data = oim
        self.overlay_name = "warped"
        self.overlay_colormap = "gray"
        self.overlay_interpolation = "linear"
        self.overlay_maxrad = 1.
        self.overlay_alpha = 1.
        self.overlay_scale = scale
        self.saveCurrentOverlay()
        # o = self.getOverlayByName("warped")
        # o.alpha = 1.

    def warpImage(self, srcd, destd, scale):
    # def warpImage(self, rad, theta, rad1, theta1, ndots=512, resize=.75, scale=2.):
        '''
        src = self.xformXY(rad, theta)
        dest = self.xformXY(rad1, theta1)
        deci = int(math.sqrt(ishape[0]*ishape[1]/ndots))
        print("deci", deci, "ndots", ndots)
        srcd = src[::deci, ::deci].reshape(-1,2)
        srcd += self.umb
        self.src_dots = srcd.copy()
        # srcd = srcd[100:150,100:150]
        # print(srcd[0])
        destd = dest[::deci, ::deci].reshape(-1,2)
        destd *= .75
        destd += self.umb
        self.dest_dots = destd.copy()
        scale = 2.
        destd *= scale
        # print(destd[8])
        # destd = destd[100:150,100:150]
        print("estimating", srcd.shape, destd.shape)
        # xform.estimate(srcd, destd)
        '''
        ishape = self.image.shape
        # xform = PiecewiseAffineTransform()
        xform = FastPiecewiseAffineTransform()
        print("estimating")
        xform.estimate(destd, srcd)
        oshape = (int(ishape[0]*scale), int(ishape[1]*scale))
        print("warping", oshape, oshape[0]*oshape[1])
        oim = skimage.transform.warp(self.image, xform, output_shape=oshape)
        return oim

    # input: 2D float array, range 0.0 to 1.0
    # output: RGB array, uint8, with colors determined by the
    # colormap and alpha, zoomed in based on the current
    # window size, center, and zoom factor
    def dataToZoomedRGB(self, data, alpha=1., colormap="gray", interpolation="linear", scale=1.):
        if scale is None:
            scale = 1.
        if colormap in self.colormaps:
            colormap = self.colormaps[colormap]
        cm = cmap.Colormap(colormap, interpolation=interpolation)

        # data = gdata.cpu().numpy()
        
        iw = data.shape[1]
        ih = data.shape[0]
        z = self.zoom / scale
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
        cx *= scale
        cy *= scale

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
            # cdata = data.cpu().numpy()
            # zslc = cv2.resize(cdata[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            idata = data[y1s:y2s,x1s:x2s].unsqueeze(0).unsqueeze(0)
            odata = F.interpolate(idata, size=(y2-y1, x2-x1)).squeeze(0).squeeze(0)
            # take remainder to allow wrapping of color maps.
            # use 1+epsilon so that 1.0 (max possible value) does not
            # get mapped to 0.
            odata = torch.clamp(torch.remainder(odata, 1.0001), min=0., max=1.0)
            zslc = odata.cpu().numpy()
            # cslc = cm(np.remainder(zslc, 1.))
            cslc = cm(zslc)
            outrgb[y1:y2, x1:x2, :] = (255*cslc[:,:,:3]*alpha).astype(np.uint8)
        return outrgb

    def drawPoints(self, ipoints, outrgb):
        points = self.ixysToWxys(ipoints)
        points = points.reshape(-1,1,1,2)
        '''
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
        '''
        color = (255,0,0)

        cv2.polylines(outrgb, points, True, color, self.warp_dot_size)
        # cv2.circle(outrgb, points[0,0,0], 3, (255,0,255), -1)

    def drawAll(self):
        if self.image is None:
            return
        self.setStatusTextFromMousePosition()
        total_alpha = .8
        if self.overlay_data is None:
            main_alpha = total_alpha
            # overlay_alpha = 0.
        elif self.overlay_alpha is not None:
            # overlay_alpha = total_alpha*self.overlay_alpha
            # main_alpha = total_alpha - overlay_alpha
            main_alpha = total_alpha*(1. - self.overlay_alpha)
        else:
            main_alpha = .5*total_alpha
        overlay_alpha = total_alpha - main_alpha

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
            outrgb += self.dataToZoomedRGB(other_data, alpha=overlay_alpha, colormap=self.overlay_colormap, interpolation=self.overlay_interpolation, scale=self.overlay_scale)

        ww = self.width()
        wh = self.height()

        scale = 1.
        if self.overlay_scale is not None:
            scale = self.overlay_scale


        if st is not None and self.dip_bars_visible and scale == 1:
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
            # dpir = dpi[:,:,::-1].copy()
            # TODO: or do they?
            dpir = dpi.copy()
            gdpir = torch.from_numpy(dpir).to(self.gpu)
            uvs = st.interpolator(st.vector_u, gdpir, mode='nearest', device=self.safe_gpu).cpu().numpy()
            vvs = st.interpolator(st.vector_v, gdpir, mode='nearest', device=self.safe_gpu).cpu().numpy()
            coherence = st.interpolator(st.linearity, gdpir, mode='nearest', device=self.safe_gpu).cpu().numpy()
            # print ("dpi", dpi.shape, dpi.dtype, dpi[0,5])
            # uvs = st.vector_u_interpolator(dpir)
            # vvs = st.vector_v_interpolator(dpir)
            # print("vvs", vvs.shape, vvs.dtype, vvs[0,5])
            # coherence = st.coherence_interpolator(dpir)
            # coherence = st.linearity_interpolator(dpir)
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

        if self.warp_dot_size > 0:
            # if scale == 1.:
            if self.overlay_name == "warped":
                if self.dest_dots is not None:
                    self.drawPoints(self.dest_dots, outrgb)
            else:
                if self.src_dots is not None:
                    self.drawPoints(self.src_dots, outrgb)


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
    parser.add_argument("--diagnostics",
                        action="store_true",
                        help="Create diagnostic overlays")
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
                        default=8,
                        help="decimation factor (default is no decimation)")
    parser.add_argument("--warp_decimation",
                        type=int,
                        default=32,
                        help="decimation factor for warping")

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
