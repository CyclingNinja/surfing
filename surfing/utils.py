import numpy as np

from PIL import Image

def im_resize(im, size):
    """
    resize an image array using PIL; general tool

    Parameters
    ----------
    im: pil image
    size: tuple

    Returns
    -------
    numpy array
    """
    pil_im = Image.fromarray(uint8(im))

    return np.array(pil_im.resize(size))


def histogram_equilizer(im, nbr_bins=256):
    """
    Histogram equalization of a greyscale image

    Parameters
    ----------
    im : pil image
    nbr_bins: default: 256, bins to equalize across

    Returns
    -------
    numpy array
    """

    # this may be the wrong implementation of histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    # use linear interpolatipon for new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf
