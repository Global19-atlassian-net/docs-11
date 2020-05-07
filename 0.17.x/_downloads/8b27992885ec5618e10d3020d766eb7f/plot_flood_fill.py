"""
=========================
Flood fill (paint bucket)
=========================

The flood fill algorithm [1]_ is the equivalent of the "paint bucket" tool of
raster graphics programs (for example The Gimp). Starting from a seed point,
connected points of the same value (up to a tolerance parameter) are found.

With the `morphology.flood_fill` function, the points found by the algorithm
are set to a new value passed to the function. It is also possible to return
only the binary mask of flooded points. The mask can be post-processed with
other functions, as in the example below.

Since flood fill operates on single-channel images, we transform here the image
to the HSV (Hue Saturation Value) space in order to flood pixels of similar
hue.

.. [1] https://en.wikipedia.org/wiki/Flood_fill
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color, morphology

img = data.astronaut()
img_hsv = color.rgb2hsv(img)

img_hsv_copy = np.copy(img_hsv)
# Fill hue channel with a different color using flood fill
_ = morphology.flood_fill(img_hsv_copy[..., 0], (313, 160), 0.5,
                          tolerance=0.016, in_place=True)

# 
mask = morphology.flood(img_hsv[..., 0], (313, 160), tolerance=0.016)
mask = morphology.remove_small_holes(mask, 40)
mask = np.logical_and(mask, img_hsv[..., 1] > 0.4)
mask = morphology.binary_opening(mask, np.ones((3, 3)))
img_hsv[mask, 0] = 0.5

fig, ax = plt.subplots(1, 2)
ax[0].imshow(color.hsv2rgb(img_hsv_copy))
ax[0].axis('off')
ax[1].imshow(img)
ax[1].contour(mask)
ax[1].axis('off')

plt.show()
