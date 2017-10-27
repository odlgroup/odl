"""Example of the Cambridge images."""

import odl.contrib.datasets.images as images

images.brain_phantom(shape=[200, 200]
                     ).show('brain_phantom (resized)')
images.resolution_phantom(shape=[100, 123]
                          ).show('resolution_phantom (resized)')
images.building().show('building')  # this is shown as gray scales ATM
images.building(gray=True).show('building (gray)')
images.rings(gray=True).show('rings (gray)')
images.rings(shape=[100, 100]).show('rings (resized)')  # shown as gray scales
images.blurring_kernel(shape=[9, 9]).show('blurring_kernel (resized)')
