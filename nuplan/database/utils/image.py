from __future__ import annotations

from functools import cached_property
from typing import BinaryIO, cast

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image as PilImage


class Image:
    """
    A class to represent an image. This class is an analogue to LidarPointCloud. It is a class for manipulating and
    transforming an image. Any transformation functions (flip, scale, translate) should be added to this class in the
    future.
    """

    def __init__(self, image: PilImage.Image) -> None:
        """
        Constructor for the Image class.
        :param image: An image of type PIL.Image.Image.
        """
        self._image = image

    @property
    def as_pil(self) -> PilImage.Image:
        """
        Returns the image of type PIL.Image.Image in uint8, RGB format.
        :return: An image of type PIL.Image.Image.
        """
        return self._image

    @cached_property
    def as_numpy(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a numpy array in uint8, RGB format.
        :return: An image as a numpy array.
        """
        return self.as_numpy_nocache()

    def as_numpy_nocache(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a numpy array in uint8, RGB format. A non caching variation to save on memory if needed.
        :return: An image as a numpy array.
        """
        return np.array(self._image, dtype=np.uint8)

    @cached_property
    def as_cv2(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a CV2 image in uint8, BGR format. It is a numpy array under the hood.
        This function is a convenience for to be used with cv2.imshow().
        :return: An image as a CV2 image.
        """
        return self.as_cv2_nocache()

    def as_cv2_nocache(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a CV2 image in uint8, BGR format. It is a numpy array under the hood. This function
        is a convenience for to be used with cv2.imshow(). A non caching variation to save on memory if needed.
        :return: An image as a CV2 image.
        """
        return cast(npt.NDArray[np.uint8], cv2.cvtColor(np.array(self._image, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    @classmethod
    def from_buffer(cls, blob: BinaryIO) -> Image:
        """
        Instantiates Image from buffer.
        :param blob: Data to load.
        :return: An Image object.
        """
        return cls(PilImage.open(blob))
