"""
===============================
Libraries
===============================

bla
"""


from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import numpy as np

image = data.coins()[50:-50, 50:-50]

plt.imshow(image)


##############################################################################
# Advanced example
# ----------------


# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border and remove small objects
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image)

fig = px.imshow(image_label_overlay)
#fig.update_traces(hoverinfo='skip')

for i, region in enumerate(regionprops(label_image)):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        fig.add_trace(go.Scatter(
            x=[minc, maxc, maxc, minc, minc], 
            y=[maxr, maxr, minr, minr, maxr], 
            fill='toself', hoveron='fills',
            text='<b>Area:</b> %.0f <br><b>Eccentricity:</b> %.2f'\
                    %(region.area, region.eccentricity),
            name='', showlegend=False))

fig
