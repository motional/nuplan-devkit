"""
Shared tools for visualizing stuff.
"""

import colorsys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw


def rainbow(nbr_colors: int, normalized: bool = False) -> List[Tuple[Any, ...]]:
    """
    Returns colors that are maximally different in HSV color space.
    :param nbr_colors: Number of colors to generate.
    :param normalized: Whether to normalize colors in 0-1. Else it is between 0-255.
    :return: <[(R <TYPE>, G <TYPE>, B <TYPE>)]>. Color <TYPE> varies depending on whether they are normalized.
    """
    hsv_tuples = [(x * 1.0 / nbr_colors, 0.5, 1) for x in range(nbr_colors)]
    colors = 255 * np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)))
    if normalized:
        colors = colors / 255.0  # type: ignore
        return list(colors)
    else:
        return [tuple([int(c) for c in color]) for color in colors]


def _color_prep(
    ncolors: Optional[int] = None,
    alpha: int = 128,
    colors: Optional[Union[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int, int]]]] = None,
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Prepares colors for image_with_boxes and draw_masks.
    :param ncolors: Total number of colors.
    :param alpha: Alpha-matting value to use for fill (0-255).
    :param colors: {id: (R, G, B) OR (R, G, B, A)}.
    :return: {id: (R, G, B, A)}.
    """
    if colors is None:
        assert ncolors is not None, 'If no colors are supplied, need to include ncolors'
        colors = [tuple(color) + (alpha,) for color in rainbow(ncolors - 1)]  # type: ignore
    else:
        if ncolors is not None:
            assert ncolors == len(colors), 'Number of supplied colors {} disagrees with supplied ncolor: {}'.format(
                len(colors), ncolors
            )
        for _id, color in colors.items():
            if isinstance(color, list):
                # Convert to tuple.
                color = tuple(color)
            if len(color) == 3:
                # If no alpha supplied, fill in global alpha.
                color = color + (alpha,)  # type: ignore
            colors[_id] = color  # type: ignore

    return colors  # type: ignore


def image_with_boxes(
    img: npt.NDArray[np.uint8],
    boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    labels: Optional[List[int]] = None,
    ncolors: Optional[int] = None,
    alpha: int = 128,
    labelset: Optional[Dict[int, str]] = None,
    scores: Optional[List[float]] = None,
    colors: Optional[Union[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int, int]]]] = None,
) -> Image:
    """
    Simple plotting function to view image with boxes.
    :param img: <np.uint8: nrows, ncols, 3>. Input image.
    :param boxes: [(xmin, ymin, xmax, ymax)]. Bounding boxes.
    :param labels: Box3D labels.
    :param ncolors: Total number of colors needed (ie number of foreground classes).
    :param alpha: Alpha-matting value to use for fill (0-255).
    :param labelset: {id: name}. Maps label ids to names.
    :param scores: Prediction scores.
    :param colors: {id: (R, G, B) OR (R, G, B, A)}.
    :return: Image instance with overlaid boxes.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if not boxes or len(boxes) == 0:
        return img

    if not labels:
        labels = [1] * len(boxes)

    if not scores:
        scores = [None] * len(boxes)  # type: ignore

    colors = _color_prep(ncolors, alpha, colors)

    draw = ImageDraw.Draw(img, 'RGBA')

    for box, label, score in zip(boxes, labels, scores):
        color = colors[label]
        bbox = [int(b) for b in box]
        draw.rectangle(bbox, outline=color[:3], fill=color)

        # make the rectangles slightly thicker
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] - 1, bbox[3] - 1], outline=color[:3], fill=None)

        text = labelset[label] if labelset else '{:.0f}'.format(label)
        if score:
            text += ': {:.0f}'.format(100 * score)

        draw.text((box[0], box[1]), text)

    return img


def build_color_mask(
    target: npt.NDArray[np.uint8], colors: Dict[int, Tuple[int, int, int, int]]
) -> npt.NDArray[np.uint8]:
    """
    Builds color mask based on color dictionary.
    :param target: <np.uint8: nrows, ncols>. Same size as image. Indicates the label of each pixel.
    :param colors: {id: (R, G, B, A)}. Color dictionary.
    :return: Color mask.
    """
    nrows, ncols = target.shape
    color_mask = np.zeros(shape=(nrows, ncols, 4), dtype='uint8')  # type: ignore
    for i in np.unique(target):
        color_mask[target == i] = colors[i]

    return color_mask


def draw_masks(
    img: Image,
    target: npt.NDArray[np.uint8],
    ncolors: Optional[int] = None,
    colors: Optional[Union[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int, int]]]] = None,
    alpha: int = 128,
) -> None:
    """
    Utility function for overlaying masks on images.
    :param img: Input image.
    :param target: <np.uint8: nrows, ncols>. Same size as image. Indicates the label of each pixel.
    :param ncolors: Total number of colors needed (ie number of foreground classes).
    :param colors: {id: (R, G, B) OR (R, G, B, A)}.
    :param alpha: Alpha-matting value to use for fill (0-255).
    """
    assert isinstance(img, Image.Image), 'img should be PIL type.'
    alpha_img = img.convert('RGBA')
    colors_prep = _color_prep(ncolors, alpha, colors)

    # Build a 'color mask' of the colors we want to overlay on the original image to show the pixel segmentation.
    color_mask = build_color_mask(target, colors_prep)

    # Alpha-composite the color mask onto the original image.
    color_mask_image = Image.fromarray(color_mask, mode='RGBA')
    alpha_img.alpha_composite(color_mask_image)
    # Modify the input image in-place to be consistent with the other rendering functions in this repo.
    img.paste(alpha_img.convert('RGB'))


def clean_ax(this_ax: plt.Axes) -> plt.Axes:
    """
    Standardizes the matplotlib axes for better visualization.
    :param this_ax: Default axes.
    :return: Standardized axes.
    """
    this_ax.get_xaxis().tick_bottom()
    this_ax.get_yaxis().tick_left()
    this_ax.spines["top"].set_visible(False)
    this_ax.spines["bottom"].set_visible(False)
    this_ax.spines["right"].set_visible(False)
    this_ax.spines["left"].set_visible(False)

    return this_ax


def pil_grid(images: List[Image.Image], max_horiz: int) -> Image.Image:
    """
    Automatically creates a mosaic from a list of PIL images.
    :param images: List of images in PIL form.
    :param max_horiz: Maximum number of images in the column.
    :return: Mosaic-like image.
    """
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)  # type: ignore
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))

    return im_grid
