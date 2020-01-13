'''
Created on 10 янв. 2020 г.

@author: 32kda
'''
from musket_core import augmenters 
from imgaug.augmenters.geometric import Affine

@augmenters.augmenter
class AffineWithPadding(Affine):
    
    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=1, cval=0, mode="constant",
                 fit_output=False, backend="auto",
                 name=None, deterministic=False, random_state=None):
        super(AffineWithPadding, self).__init__(scale, translate_percent, translate_px, rotate, shear, order, cval, mode, fit_output, backend, name, deterministic, random_state)
        self._mode_heatmaps = mode
        self._mode_segmentation_maps = mode

