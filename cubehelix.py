"""Cubehelix module. I have this here only because cubehelix is broken on the IAS python installs"""
import matplotlib.colors as colors
import numpy as np

def cubehelix(gamma=1.0, s=0.5, r=-1.5, h=1.0):
    """Return custom data dictionary of (r,g,b) conversion functions, which
    can be used with :func:`register_cmap`, for the cubehelix color scheme.

    Unlike most other color schemes cubehelix was designed by D.A. Green to
    be monotonically increasing in terms of perceived brightness.
    Also, when printed on a black and white postscript printer, the scheme
    results in a greyscale with monotonically increasing brightness.
    This color scheme is named cubehelix because the r,g,b values produced
    can be visualised as a squashed helix around the diagonal in the
    r,g,b color cube.

    For a unit color cube (i.e. 3-D coordinates for r,g,b each in the
    range 0 to 1) the color scheme starts at (r,g,b) = (0,0,0), i.e. black,
    and finishes at (r,g,b) = (1,1,1), i.e. white. For some fraction *x*,
    between 0 and 1, the color is the corresponding grey value at that
    fraction along the black to white diagonal (x,x,x) plus a color
    element. This color element is calculated in a plane of constant
    perceived intensity and controlled by the following parameters.

    Optional keyword arguments:

      =========   =======================================================
      Keyword     Description
      =========   =======================================================
      gamma       gamma factor to emphasise either low intensity values
                  (gamma < 1), or high intensity values (gamma > 1);
                  defaults to 1.0.
      s           the start color; defaults to 0.5 (i.e. purple).
      r           the number of r,g,b rotations in color that are made
                  from the start to the end of the color scheme; defaults
                  to -1.5 (i.e. -> B -> G -> R -> B).
      h           the hue parameter which controls how saturated the
                  colors are. If this parameter is zero then the color
                  scheme is purely a greyscale; defaults to 1.0.
      =========   =======================================================

    """

    def get_color_function(p0, p1):
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            xg = x**gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = h * xg * (1 - xg) / 2

            phi = 2 * np.pi * (s / 3 + r * x)

            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    return {
            'red': get_color_function(-0.14861, 1.78277),
            'green': get_color_function(-0.29227, -0.90649),
            'blue': get_color_function(1.97294, 0.0),
    }

_cubehelix_data = cubehelix()

cubehelix = colors.LinearSegmentedColormap("cubehelix", _cubehelix_data, 256)

def _reverser(f):
    def freversed(x):
        return f(1-x)
    return freversed


def revcmap(data):
    """Can only handle specification *data* in dictionary format."""
    data_r = {}
    for key, val in data.iteritems():
        if callable(val):
            valnew = _reverser(val)
                # This doesn't work: lambda x: val(1-x)
                # The same "val" (the first one) is used
                # each time, so the colors are identical
                # and the result is shades of gray.
        else:
            # Flip x and exchange the y values facing x = 0 and x = 1.
            valnew = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(val)]
        data_r[key] = valnew
    return data_r

cubehelix_r = colors.LinearSegmentedColormap("cubehelix_r", revcmap(_cubehelix_data), 256)