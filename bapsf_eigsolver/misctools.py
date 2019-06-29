#!/usr/bin/env /u/local/apps/python/2.6.1/bin/python
"""
Collection of useful tools and definitions
"""

import matplotlib.pyplot as plt

# -----------------------------------------------------------
class attrdict(dict):
    """
    Dictionary with keys accessible as object attributes.
    Example: d["key"] is equivalent to d.key, both for read and write operations.
    """
    def __getattr__(self, name):
        """
        Maps values to attributes. Only called if there isn't an attribute with this name
        """
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(name)


    def __setattr__(self, item, value):
        """
        Maps attributes to values.
        """
        if item in self.__dict__:   # normal attributes handled as usual
            dict.__setattr__(self, item, value)
        else:
            self.__setitem__(item, value)


# -----------------------------------------------------------
def GetSeriesRunDirs(prefix, path):
    # find the directories with name prefixXX in dir "path"
    import os, re

    dirList = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # Get all dirs that start with prefix followed by 2 digits
    matchList = [dir for dir in dirList 
                    if re.match(prefix+r"\d\d", dir) and (len(dir)==len(prefix)+2)]
    
    return matchList




# ----------------------------------------------------
def plot4dir(v1, v2=None, p = (0,0,0,0), msg="", axislabels="xyzt"):
    """
    1D plots of 4D arrays v1 (and v2 if specified) around the point (ix,iy,iz,it)
    """

    ix,iy,iz,it = p

    plt.ioff()  # turn off matplotlib interactive regime for faster plotting

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(v1[:,iy,iz,it],'g-')
    plt.xlabel(axislabels[0])
    if v2 != None: plt.plot(v2[:,iy,iz,it],'r--') 
    plt.plot([ix],[v1[p]],'o')

    plt.subplot(2,2,2)
    plt.plot(v1[ix,:,iz,it],'g-')
    plt.xlabel(axislabels[1])
    if v2 != None: plt.plot(v2[ix,:,iz,it],'r--') 
    plt.plot([iy],[v1[p]],'o')

    plt.subplot(2,2,3)
    plt.plot(v1[ix,iy,:,it],'g-')
    plt.xlabel(axislabels[2])
    if v2 != None: plt.plot(v2[ix,iy,:,it],'r--') 
    plt.plot([iz],[v1[p]],'o')

    plt.subplot(2,2,4)
    plt.plot(v1[ix,iy,iz,:],'g-')
    plt.xlabel(axislabels[3])
    if v2 != None: plt.plot(v2[ix,iy,iz,:],'r--') 
    plt.plot([it],[v1[p]],'o')

    if msg: msg = ",  " + msg
    plt.suptitle("Central point (i%s,i%s,i%s,i%s)=%s%s" % (axislabels[0],axislabels[1],
                                                       axislabels[2],axislabels[3],
                                                       str(p), msg))
    
    plt.ion() # turn back on the interactive regime
    plt.show() # show the plot

# ----------------------------------------------------
def plot3dir(v1, v2=None, p = (0,0,0), msg = "", axislabels="xyt"):
    """
    1D plots of 3D arrays v1 (and v2 if specified)
    """

    ix,iy,iz = p

    plt.ioff()  # turn off matplotlib interactive regime for faster plotting

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(v1[:,iy,iz],'g-')
    plt.xlabel(axislabels[0])
    if v2 != None: plt.plot(v2[:,iy,iz],'r--') 
    plt.plot([ix],[v1[p]],'o')

    plt.subplot(2,2,2)
    plt.plot(v1[ix,:,iz],'g-')
    plt.xlabel(axislabels[1])
    if v2 != None: plt.plot(v2[ix,:,iz],'r--') 
    plt.plot([iy],[v1[p]],'o')

    plt.subplot(2,2,3)
    plt.plot(v1[ix,iy,:],'g-')
    plt.xlabel(axislabels[2])
    if v2 != None: plt.plot(v2[ix,iy,:],'r--') 
    plt.plot([iz],[v1[p]],'o')

    if msg: msg = ",  " + msg
    plt.suptitle("Central point (i%s,i%s,i%s)=%s%s" % (axislabels[0],axislabels[1],axislabels[2],
                                                   str(p), msg))
    
    plt.ion() # turn back on the interactive regime
    plt.show() # show the plot

# ----------------------------------------------------
def plot2dir(v1, v2=None, p = (0,0), msg = "", axislabels="xt"):
    """
    1D plots of 2D arrays v1 (and v2 if specified)
    """
    plt.ioff()  # turn off matplotlib interactive regime for faster plotting

    ix,iy = p

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(v1[:,iy],'g-')
    plt.xlabel(axislabels[0])
    if v2 != None: plt.plot(v2[:,iy],'r--') 
    plt.plot([ix],[v1[p]],'o')

    plt.subplot(2,1,2)
    plt.plot(v1[ix,:],'g-')
    plt.xlabel(axislabels[1])
    if v2 != None: plt.plot(v2[ix,:],'r--') 
    plt.plot([iy],[v1[p]],'o')

    if msg: msg = ",  " + msg
    plt.suptitle("Central point (i%s,i%s)=%s%s" % (axislabels[0],axislabels[1],
                                               str(p), msg))

    plt.ion() # turn back on the interactive regime
    plt.show() # show the plot


# ----------------------------------------------------
def plot1dir(v1, v2=None, p = (0), msg = "", axislabels="x"):
    """
    1D plots of 1D arrays v1 (and v2 if specified)
    """

    plt.ioff()  # turn off matplotlib interactive regime for faster plotting

    ix = p

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(v1[:],'g-')
    plt.xlabel(axislabels[0])
    if v2 != None: plt.plot(v2[:],'r--') 
    plt.plot([ix],[v1[p]],'o')


    if msg: msg = ",  " + msg
    plt.suptitle("Central point (i%s)=%s%s" % (axislabels[0],
                                               str(p), msg))

    plt.ion() # turn back on the interactive regime
    plt.show() # show the plot
    
# ----------------------------------------------------
def ppmatrix(M, digits=3, idiv=None, imod=None, trace=False):
    """Print a real 2D matrix without line wrapping """
    
    sformat = "%%%d.%de" % (digits+8, digits)
    cutoff = abs(M).max()*1.e-20

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (not idiv) or (idiv and (i % idiv == imod) and (j % idiv == imod)):
                if trace:
                    if abs(M[i,j]) > cutoff:
                        print(" * ", end=' ')
                    else:
                        print(" 0 ", end=' ')
                else:
                    print(sformat % M[i,j], end=' ')
        if (not idiv) or (idiv and (i % idiv == imod)):
            print("\n")

# ----------------------------------------------------
def ppcmatrix(M, digits=3):
    """Print a complex 2D matrix without line wrapping """
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            sformat = "%%%d.%de + %%%d.%dej" % (digits+8, digits, digits+6, digits)
            print(sformat % (M[i,j].real, M[i,j].imag), end=' ')
        print("\n")

# ----------------------------------------------------
class DebugClass(object):
    """
    Primitive debug object: print a debug message and a counter
    To use: call dbg() anywhere in the code, or with a message dbg('message')
    """

    def __init__(self):
        object.__init__(self)
        self.counter = 0

    def p(self, msg=''):
        print("%s %s %d" % ("DEBUG:", msg, self.counter))

    def __call__(self, msg=''):
        self.p(msg)
        self.counter += 1

dbg = DebugClass()


# -----------------------------------------------------------------------#
def Calc_radial_integral(rvec,f):
    """
    Take in a radius vector and radial profile and calculate its radial integral
    """
    
    ra = rvec[0]
    rb = rvec[-1]

    Nr = len(rvec)

    h = (rb-ra)/(Nr-1)

    f_r = [f[ix]*rvec[ix] for ix in range(Nr)]

    numerator = 2*sum(f_r)*h

    denominator = rb**2-ra**2

    return numerator/denominator


# -----------------------------------------------------------------------#
def get_BOUT_FLUC_files():
    """
    Return a list of all of the BOUT_FLUC* files in the current directory
    """

    import os

    BOUT_FLUC_filelist = []
    allfiles = []

    for root, dir, files in os.walk("."):
        allfiles.append(files)

    for fdir in allfiles:
        for filename in fdir:
            if filename[:9] == "BOUT_FLUC":
                BOUT_FLUC_filelist.append(filename)

    BOUT_FLUC_filelist.sort()

    return BOUT_FLUC_filelist

# -----------------------------------------------------------------------#
def Cross_Correlate(x,y,L,covariance = False):
    """
    Compute the Cross Correlation between two series
    """
    
    N = len(x)
    xmxavg = []
    ymyavg = []
    xmxavgsq = []
    ymyavgsq = []
    numerator = 0.0
    total_xmxavgsq = 0.0
    total_ymyavgsq = 0.0
    denominator = 0.0

    xavg = sum(x)/float(N)
    yavg = sum(y)/float(N)


    if L<0:
        for ix in range(N-abs(L)):
            xmxavg.append(x[ix+abs(L)] - xavg)
            xmxavgsq.append(xmxavg[ix]**2)
            
            ymyavg.append(y[ix] - yavg)
            ymyavgsq.append(ymyavg[ix]**2)
            
            numerator += xmxavg[ix]*ymyavg[ix]
            total_xmxavgsq += xmxavgsq[ix]
            total_ymyavgsq += ymyavgsq[ix]


    if L>= 0:
        for ix in range(N-L):
            xmxavg.append(x[ix] - xavg)
            xmxavgsq.append(xmxavg[ix]**2)
            
            ymyavg.append(y[ix+L] - yavg)
            ymyavgsq.append(ymyavg[ix]**2)
            
            numerator += xmxavg[ix]*ymyavg[ix]
            total_xmxavgsq += xmxavgsq[ix]
            total_ymyavgsq += ymyavgsq[ix]

    denominator = (total_xmxavgsq*total_ymyavgsq)**(0.5)


    if covariance:
        return numerator/N
    else:
        return numerator/denominator

# -----------------------------------------------------------------------#
def Smooth_func(f,n):
    """
    Smooth a function by averaging a certain number of consecutive points
    """

    flength = len(f)
    smoothf_length = flength/n

    smooth_f = []
    for i in range(smoothf_length):
        smooth_f += [0]

    for i in range(smoothf_length):
        for j in range(n):
            smooth_f[i] += f[j+i*n]/n

    return smooth_f


# -----------------------------------------------------------------------#

class BlockingMouseInput(object):
    """Class that stops the program execution until mouse click(s)"""

    callback = None
    verbose = False
    def __call__(self, fig, n=1, verbose=False, anywhere=False):
        """Blocking call to retrieve n coordinates through mouse clicks."""
        import time, sys
        
        assert isinstance(n, int), "Requires an integer argument"

        # Ensure that the current figure is shown
        fig.show()
        # connect the click events to the on_click function call
        self.callback = fig.canvas.mpl_connect('button_press_event',
                                               self.on_click)

        # initialize the list of click coordinates
        self.clicks = []
        self.verbose = verbose
        self.anywhere = anywhere # allow to click anywhere on the figure, 
                                 # not only within the axes

        # wait for n clicks
        print("Waiting for mouse click...", end=' ')
        sys.stdout.flush()
        counter = 0
        while len(self.clicks) < n:
            fig.canvas.flush_events()
            # rest for a moment
            time.sleep(0.01)
#        print "\r"+" "*40+"\n"


        # All done! Disconnect the event and return what we have
        fig.canvas.mpl_disconnect(self.callback)
        self.callback = None
        return self.clicks

    def on_click(self, event):
        """Event handler to process mouse click"""

        # if it's a valid click, append the coordinates to the list
        if event.inaxes:
            self.clicks.append((event.xdata, event.ydata))
            if self.verbose:
                print("\rInput %i: %f, %f" % (len(self.clicks),
                                    event.xdata, event.ydata))
        elif self.anywhere:
            self.clicks.append((0, 0))

 
# -----------------------------------------------------------

def PasteImage(imlarge, imsmall, x,y):
    """ 
    Insert the small image into the large one at coordinates (x,y) (top left corner)
    Return the modified large image
    """


#    box = (30, 10, 530, 440)
#    imsmall = imsmall.crop(box)

    w,h = imsmall.size
    w,h = imsmall.size
    imlarge.paste(imsmall, (x,y, x+w,y+h))

    return imlarge
    
# ----------------------------------------------------------

def DrawRec(im, box):
    """ 
    Draw a rectangle on the image
    """

    import Image, ImageDraw

    draw = ImageDraw.Draw(im)
    draw.rectangle(box, fill='white')
    del draw 

    return im

# ----------------------------------------------------------



class Progress(object):
    """
    A simple progress indicator: print a message and a percentage counter, same line.
    To use: create the object to start, then .update(msg=msg, percent=percentage), 
    then .end(msg=msg) to output final time
    """

    def __init__(self):
        import time
        object.__init__(self)
        self.tstart = time.clock()

    def update(self, msg='', percent=0):
        import sys
        print("\r%s%2d%%" % (msg, round(percent)), end=' ')
        sys.stdout.flush()

    def end(self, msg=''):
        import time
        self.tend = time.clock()
        print("\r%s (t=%.2gs)" % (msg, (self.tend-self.tstart)))

