"""Example of the Cambridge images."""

import odl.contrib.datasets.images as images
import numpy as np
import odl


def convert_to_odl(image):
    """Convert image to ODL object."""

    shape = image.shape

    if len(shape) == 2:
        space = odl.uniform_discr([0, 0], shape, shape)
    elif len(shape) == 3:
        d = shape[2]
        shape = shape[:2]
        image = np.transpose(image, (2, 0, 1))
        space = odl.uniform_discr([0, 0], shape, shape) ** d

    image = space.element(image)

    return image


im = convert_to_odl(images.brain_phantom(shape=[200, 200]))
im.show('brain_phantom (resized)')

im = convert_to_odl(images.resolution_phantom(shape=[100, 123]))
im.show('resolution_phantom (resized)')

im = convert_to_odl(images.building())
im.show('building')  # this is shown as gray scales ATM

im = convert_to_odl(images.building(gray=True))
im.show('building (gray)')

im = convert_to_odl(images.rings(gray=True))
im.show('rings (gray)')

im = convert_to_odl(images.rings(shape=[100, 100]))
im.show('rings (resized)')  # this is shown as gray scales ATM

im = convert_to_odl(images.blurring_kernel(shape=[9, 9]))
im.show('blurring_kernel (resized)')
