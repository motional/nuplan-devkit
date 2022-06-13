from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.database.maps_db.metadata import MapLayerMeta


class MapLayer:
    """
    Wraps a map layer numpy array and provides methods for computing the distance to the foreground and
     determining if points are on the foreground.
    """

    def __init__(
        self,
        data: npt.NDArray[np.uint8],
        metadata: MapLayerMeta,
        joint_distance: Optional[npt.NDArray[np.float64]] = None,
        transform_matrix: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """
        Initiates MapLayer.
        :param data: Map layer as a binary numpy array with one channel.
        :param metadata: Map layer metadata.
        :param joint_distance:
            Same shape as `mask`.
            For every valid (row, col) in `joint_distance`, the *magnitude* of the value `joint_distance[row][col]` is
             the l2 distance on the ground plane from `mask[row][col]` to the nearest value in `mask` not equal to
              `mask[row][col]`.

            The *sign* of `joint_distance[row][col]` is positive if `mask[row][col] == 0`, and
             negative if `mask[row][col] == 1`.
        :param transform_matrix: Matrix for converting from physical coordinates to pixel coordinates.
        """
        # TODO: Some map layers have foreground set to 255 instead of one.
        if metadata.is_binary and np.amax(data) == 255:
            data = data.copy()
            data[data == 255] = 1

        self.data = data
        self.metadata = metadata
        self.nrows, self.ncols = data.shape[-2:]
        self.joint_distance = joint_distance
        self.foreground = 1
        self.background = 0

        if transform_matrix is None:
            # Use `n_rows - 1` so (0, 0) in physical space becomes (0, n_rows - 1)
            # in pixel space (the bottom-left pixel).
            transform_matrix = np.array(
                [
                    [1.0 / self.metadata.precision, 0, 0, 0],
                    [0, -1.0 / self.metadata.precision, 0, self.nrows - 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        self._transform_matrix = transform_matrix

    @property
    def precision(self) -> float:
        """
        Returns map resolution in meters per pixel. Typically set to 0.1, meaning that 10 pixels
            correspond to 1 meter.
        :return: Meters per pixel.
        """
        return self.metadata.precision  # type: ignore

    def mask(self, dilation: float = 0) -> npt.NDArray[np.float64]:
        """
        Returns full map layer content optionally including dilation.
        :param dilation: Max distance from the foreground. Should be not less than 0.
        :return: A full map layer content as a numpy array.
        """
        return self.crop(slice(0, self.nrows), slice(0, self.ncols), dilation)

    def crop(self, rows: slice, cols: slice, dilation: float = 0) -> npt.NDArray[np.float64]:
        """
        Returns the map data in the rows and cols specified.
        :param rows: Range of rows to include in the crop.
        :param cols: Range of columns to include in the crop.
        :param dilation: If greater than 0, all pixels within dilation distance of the foreground will be made
         foreground pixels.
        :return: A full map layer content as a numpy array.
        """
        assert dilation >= 0, "Negative dilation not supported."
        if dilation == 0:
            return self.data[rows, cols]  # type: ignore
        else:
            assert self.metadata.can_dilate
            return self.joint_distance[rows, cols] <= dilation  # type: ignore

    @property
    def transform_matrix(self) -> npt.NDArray[np.float64]:
        """
        Matrix for transforming physical coordinates into pixel coordinates.
        Physical coordinates use bottom-left origin, while pixel coordinates use upper-left origin.
        :return: <np.ndarray: 4, 4>, the transform matrix.
        """
        return self._transform_matrix

    def to_pixel_coords(self, x: Any, y: Any) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """
        Gets the image coordinates given the x-y coordinates of points.
        :param x: Global x coordinates. Can be a scalar, list or a numpy array.
        :param y: Global y coordinates. Can be a scalar, list or a numpy array.
        :return: (px <np.int32: x.shape>, py <np.int32: y.shape>). Pixel coordinates in map.
        """
        x = np.atleast_1d(np.array(x))
        y = np.atleast_1d(np.array(y))

        assert x.shape == y.shape
        assert x.ndim == y.ndim == 1

        pts = np.stack([x, y, np.zeros(x.shape), np.ones(x.shape)])  # type: ignore
        pixel_coords = np.round(np.dot(self.transform_matrix, pts)).astype(np.int32)

        return pixel_coords[0, :], pixel_coords[1, :]

    def _is_in_bounds(self, px: npt.NDArray[np.int32], py: npt.NDArray[np.int32]) -> npt.NDArray[np.bool_]:
        """
        Determines whether points in pixel space are within the dimensions of this map.
        :param px: pixel coordinates.
        :param py: pixel coordinates.
        :return: <np.bool: px.shape> with True to indicate points in pixel space are within the dimensions of this map.
        """
        in_bounds = np.full(px.shape, True)
        in_bounds[px < 0] = False
        in_bounds[px >= self.ncols] = False
        in_bounds[py < 0] = False
        in_bounds[py >= self.nrows] = False

        return in_bounds

    def _dilated_distance(
        self, px: npt.NDArray[np.float64], py: npt.NDArray[np.float64], dilation: float
    ) -> npt.NDArray[np.float64]:
        """
        Gives the distance to the dilated mask. A positive distance means outside the mask,
        a negative means inside. px and py are in pixel coordinates and should be in bound.
        :param px: pixel coordinates.
        :param py: pixel coordinates.
        :param dilation: dilation in meters.
        :return: The distance matrix to the dilated mask.
        """
        # Dilating the mask (the 1's) makes the 1-to-nearest-0 distances larger,
        # and we return those as negative numbers. Dilation makes the
        # 0-to-nearest-1 distances smaller, and we return those as positive
        # values. If the point is off the mask, but dilation extends the mask
        # *over* the point, we want to return (dilation - distance_to_nearest_1)
        # as a negative number. Subtracting dilation works in all three cases.
        return self.joint_distance[py, px] - dilation  # type: ignore

    def is_on_mask(self, x: Any, y: Any, dilation: float = 0.0) -> npt.NDArray[np.bool_]:
        """
        Determines whether the points are on the mask (foreground of the layer).
        :param x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param dilation: Specifies the threshold on the distance from the drivable_area mask.
            The drivable_area mask is dilated to include points which are within this distance from itself.
        :return: <np.bool: x.shape>, True if the points are on the mask, otherwise False.
        """
        px, py = self.to_pixel_coords(x, y)
        on_mask = np.zeros(px.size, dtype=bool)  # type: ignore
        in_bounds = self._is_in_bounds(px, py)

        if dilation > 0:
            assert self.metadata.can_dilate
            on_mask[in_bounds] = self._dilated_distance(px[in_bounds], py[in_bounds], dilation) < 0
        else:
            on_mask[in_bounds] = self.data[py[in_bounds], px[in_bounds]] == self.foreground

        return on_mask

    def dist_to_mask(self, x: Any, y: Any, dilation: float = 0.0) -> npt.NDArray[np.float32]:
        """
        Returns the physical distance of the closest 'mask boundary' to physical point (x, y).
        If (x, y) is *on* mask, returns distance to nearest point *off* mask as a *negative* value.
        If (x, y) is *off* mask, returns distance to nearest point *on* mask as a *positive* value.
        :param x: Physical x. Can be a scalar, list or a numpy array of x coordinates.
        :param y: Physical y. Can be a scalar, list or a numpy array of x coordinates.
        :param dilation: Specifies the threshold on the distance from the drivable_area mask.
             The drivable_area mask is dilated to include points which are within this distance from itself.
        :return: <np.float32: x.shape>, Distance to nearest mask boundary, or NAN if out of bounds in pixel space.
        """
        assert self.metadata.can_dilate

        px, py = self.to_pixel_coords(x, y)

        in_bounds = self._is_in_bounds(px, py)

        distance = np.full(px.shape, np.nan, dtype=np.float32)  # type: ignore
        distance[in_bounds] = self._dilated_distance(px[in_bounds], py[in_bounds], dilation)

        return distance
