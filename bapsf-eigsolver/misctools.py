#!/usr/bin/env /u/local/apps/python/2.6.1/bin/python
"""
Collection of useful tools and definitions
"""


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


# -----------------------------------------------------------
 
class BlockingMouseInput(object):
    """Class that stops the program execution until mouse click(s)"""

    callback = None
    verbose = False
    def __call__(self, fig, n=1, verbose=False):
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

 
