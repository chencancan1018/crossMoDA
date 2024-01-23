import os
import itertools
import warnings
import re
from copy import deepcopy
import collections
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union, Any, Callable, List, Optional, Tuple, Type, Mapping, cast

import numpy as np
import torch
import enum
from .type_definitions import NdarrayOrTensor, NdarrayTensor, DtypeLike, IndexSelection
from .enums import NumpyPadMode, PytorchPadMode, GridSampleMode, TraceKeys, InterpolateMode, TransformBackends
from mmcv.utils import TORCH_VERSION, digit_version

from .deprecate_utils import optional_import, min_version

def get_random_patch(
    dims: Sequence[int], patch_size: Sequence[int], rand_state: Optional[np.random.RandomState] = None
) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.

    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_state: a random state object to generate random numbers from

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    rand_int = np.random.randint if rand_state is None else rand_state.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

    # create the slices for each dimension which define the patch in the source array
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))

def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))

def compute_divisible_spatial_size(spatial_shape: Sequence[int], k: Union[Sequence[int], int]):
    """
    Compute the target spatial size which should be divisible by `k`.

    Args:
        spatial_shape: original spatial shape.
        k: the target k for each spatial dimension.
            if `k` is negative or 0, the original size is preserved.
            if `k` is an int, the same `k` be applied to all the input spatial dimensions.

    """
    k = fall_back_tuple(k, (1,) * len(spatial_shape))
    new_size = []
    for k_d, dim in zip(k, spatial_shape):
        new_dim = int(np.ceil(dim / k_d) * k_d) if k_d > 0 else dim
        new_size.append(new_dim)

    return new_size

def convert_pad_mode(dst: NdarrayOrTensor, mode: Union[NumpyPadMode, PytorchPadMode, str]):
    """
    Utility to convert padding mode between numpy array and PyTorch Tensor.

    Args:
        dst: target data to convert padding mode for, should be numpy array or PyTorch Tensor.
        mode: current padding mode.

    """
    mode = mode.value if isinstance(mode, (NumpyPadMode, PytorchPadMode)) else mode
    if isinstance(dst, torch.Tensor):
        if mode == "wrap":
            mode = "circular"
        if mode == "edge":
            mode = "replicate"
        return look_up_option(mode, PytorchPadMode)
    if isinstance(dst, np.ndarray):
        if mode == "circular":
            mode = "wrap"
        if mode == "replicate":
            mode = "edge"
        return look_up_option(mode, NumpyPadMode)
    raise ValueError(f"unsupported data type: {type(dst)}.")

def generate_label_classes_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    label_spatial_shape: Sequence[int],
    indices: Sequence[NdarrayOrTensor],
    ratios: Optional[List[Union[float, int]]] = None,
    rand_state: Optional[np.random.RandomState] = None,
    allow_smaller: bool = False,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the specified ratios of label classes.
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        indices: sequence of pre-computed foreground indices of every class in 1 dimension.
        ratios: ratios of every class in the label to generate crop centers, including background class.
            if None, every class will have the same ratio to generate crop centers.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    if num_samples < 1:
        raise ValueError("num_samples must be an int number and greater than 0.")
    ratios_: List[Union[float, int]] = ([1] * len(indices)) if ratios is None else ratios
    if len(ratios_) != len(indices):
        raise ValueError("random crop ratios must match the number of indices of classes.")
    if any(i < 0 for i in ratios_):
        raise ValueError("ratios should not contain negative number.")

    for i, array in enumerate(indices):
        if len(array) == 0:
            warnings.warn(f"no available indices of class {i} to crop, set the crop ratio of this class to zero.")
            ratios_[i] = 0

    centers = []
    classes = rand_state.choice(len(ratios_), size=num_samples, p=np.asarray(ratios_) / np.sum(ratios_))
    for i in classes:
        # randomly select the indices of a class based on the ratios
        indices_to_use = indices[i]
        random_int = rand_state.randint(len(indices_to_use))
        center = unravel_index(indices_to_use[random_int], label_spatial_shape).tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return centers

def unravel_index(idx, shape) -> NdarrayOrTensor:
    """`np.unravel_index` with equivalent implementation for torch.

    Args:
        idx: index to unravel
        shape: shape of array/tensor

    Returns:
        Index unravelled for given shape
    """
    if isinstance(idx, torch.Tensor):
        coord = []
        for dim in reversed(shape):
            coord.append(idx % dim)
            idx = floor_divide(idx, dim)
        return torch.stack(coord[::-1])
    return np.asarray(np.unravel_index(idx, shape))

def floor_divide(a: NdarrayOrTensor, b) -> NdarrayOrTensor:
    """`np.floor_divide` with equivalent implementation for torch.

    As of pt1.8, use `torch.div(..., rounding_mode="floor")`, and
    before that, use `torch.floor_divide`.

    Args:
        a: first array/tensor
        b: scalar to divide by

    Returns:
        Element-wise floor division between two arrays/tensors.
    """
    if isinstance(a, np.ndarray):
        return np.floor_divide(a, b)
    elif isinstance(a, torch.Tensor):
        if digit_version(TORCH_VERSION)[:2] >= digit_version('1.8'):
            return torch.div(a, b, rounding_mode="floor")
        return torch.floor_divide(a, b)
    else:
        raise TypeError(f"The type of {a} must be np.ndarray or torch.Tensor!")
    # return np.floor_divide(a, b)

def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: Optional[np.random.RandomState] = None,
    allow_smaller: bool = False,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape).tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return centers

def correct_crop_centers(
    centers: List[int],
    spatial_size: Union[Sequence[int], int],
    label_spatial_shape: Sequence[int],
    allow_smaller: bool = False,
):
    """
    Utility to correct the crop center if the crop size and centers are not compatible with the image size.

    Args:
        centers: pre-computed crop centers of every dim, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if any(np.subtract(label_spatial_shape, spatial_size) < 0):
        if not allow_smaller:
            raise ValueError("The size of the proposed random crop ROI is larger than the image size.")
        spatial_size = tuple(min(l, s) for l, s in zip(label_spatial_shape, spatial_size))

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1
    valid_centers = []
    for c, v_s, v_e in zip(centers, valid_start, valid_end):
        center_i = min(max(c, v_s), v_e - 1)
        valid_centers.append(int(center_i))
    return valid_centers

def is_positive(img):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return img > 0

def generate_spatial_bounding_box(
    img: NdarrayOrTensor,
    select_fn: Callable = is_positive,
    channel_indices: Optional[IndexSelection] = None,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    """
    Generate the spatial bounding box of foreground in the image with start-end positions (inclusive).
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...], [-1, -1, ...] if there's no positive intensity.

    Args:
        img: a "channel-first" image of shape (C, spatial_dim1[, spatial_dim2, ...]) to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img[list(ensure_tuple(channel_indices))] if channel_indices is not None else img
    data = select_fn(data).any(0)
    ndim = len(data.shape)
    margin = ensure_tuple_rep(margin, ndim)
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data
        if len(ax) != 0:
            dt = any_np_pt(dt, ax)

        if not dt.any():
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        arg_max = where(dt == dt.max())[0]
        min_d = max(arg_max[0] - margin[di], 0)
        max_d = arg_max[-1] + margin[di] + 1

        box_start[di] = min_d.detach().cpu().item() if isinstance(min_d, torch.Tensor) else min_d  # type: ignore
        box_end[di] = max_d.detach().cpu().item() if isinstance(max_d, torch.Tensor) else max_d  # type: ignore

    return box_start, box_end

def map_classes_to_indices(
    label: NdarrayOrTensor,
    num_classes: Optional[int] = None,
    image: Optional[NdarrayOrTensor] = None,
    image_threshold: float = 0.0,
) -> List[NdarrayOrTensor]:
    """
    Filter out indices of every class of the input label data, return the indices after fattening.
    It can handle both One-Hot format label and Argmax format label, must provide `num_classes` for
    Argmax label.

    For example:
    ``label = np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])`` and `num_classes=3`, will return a list
    which contains the indices of the 3 classes:
    ``[np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])]``

    Args:
        label: use the label data to get the indices of every class.
        num_classes: number of classes for argmax label, not necessary for One-Hot label.
        image: if image is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.

    """
    img_flat: Optional[NdarrayOrTensor] = None
    if image is not None:
        img_flat = ravel((image > image_threshold).any(0))

    indices: List[NdarrayOrTensor] = []
    # assuming the first dimension is channel
    channels = len(label)

    num_classes_: int = channels
    if channels == 1:
        if num_classes is None:
            raise ValueError("if not One-Hot format label, must provide the num_classes.")
        num_classes_ = num_classes

    for c in range(num_classes_):
        label_flat = ravel(any_np_pt(label[c : c + 1] if channels > 1 else label == c, 0))
        label_flat = img_flat & label_flat if img_flat is not None else label_flat
        # no need to save the indices in GPU, otherwise, still need to move to CPU at runtime when crop by indices
        cls_indices: NdarrayOrTensor = convert_data_type(nonzero(label_flat), device=torch.device("cpu"))[0]
        indices.append(cls_indices)

    return indices

def weighted_patch_samples(
    spatial_size: Union[int, Sequence[int]],
    w: NdarrayOrTensor,
    n_samples: int = 1,
    r_state: Optional[np.random.RandomState] = None,
) -> List:
    """
    Computes `n_samples` of random patch sampling locations, given the sampling weight map `w` and patch `spatial_size`.

    Args:
        spatial_size: length of each spatial dimension of the patch.
        w: weight map, the weights must be non-negative. each element denotes a sampling weight of the spatial location.
            0 indicates no sampling.
            The weight map shape is assumed ``(spatial_dim_0, spatial_dim_1, ..., spatial_dim_n)``.
        n_samples: number of patch samples
        r_state: a random state container

    Returns:
        a list of `n_samples` N-D integers representing the spatial sampling location of patches.

    """
    if w is None:
        raise ValueError("w must be an ND array.")
    if r_state is None:
        r_state = np.random.RandomState()
    img_size = np.asarray(w.shape, dtype=int)
    win_size = np.asarray(fall_back_tuple(spatial_size, img_size), dtype=int)

    s = tuple(slice(w // 2, m - w + w // 2) if m > w else slice(m // 2, m // 2 + 1) for w, m in zip(win_size, img_size))
    v = w[s]  # weight map in the 'valid' mode
    v_size = v.shape
    v = ravel(v)
    if (v < 0).any():
        v -= v.min()  # shifting to non-negative
    v = cumsum(v)
    if not v[-1] or not isfinite(v[-1]) or v[-1] < 0:  # uniform sampling
        idx = r_state.randint(0, len(v), size=n_samples)
    else:
        r, *_ = convert_to_dst_type(r_state.random(n_samples), v)
        idx = searchsorted(v, r * v[-1], right=True)  # type: ignore
    idx, *_ = convert_to_dst_type(idx, v, dtype=torch.int)  # type: ignore
    # compensate 'valid' mode
    diff = np.minimum(win_size, img_size) // 2
    diff, *_ = convert_to_dst_type(diff, v)  # type: ignore
    return [unravel_index(i, v_size) + diff for i in idx]

def searchsorted(a: NdarrayTensor, v: NdarrayOrTensor, right=False, sorter=None, **kwargs) -> NdarrayTensor:
    """
    `np.searchsorted` with equivalent implementation for torch.

    Args:
        a: numpy array or tensor, containing monotonically increasing sequence on the innermost dimension.
        v: containing the search values.
        right: if False, return the first suitable location that is found, if True, return the last such index.
        sorter: if `a` is numpy array, optional array of integer indices that sort array `a` into ascending order.
        kwargs: if `a` is PyTorch Tensor, additional args for `torch.searchsorted`, more details:
            https://pytorch.org/docs/stable/generated/torch.searchsorted.html.

    """
    side = "right" if right else "left"
    if isinstance(a, np.ndarray):
        return np.searchsorted(a, v, side, sorter)  # type: ignore
    return torch.searchsorted(a, v, right=right, **kwargs)  # type: ignore

def any_np_pt(x: NdarrayOrTensor, axis: Union[int, Sequence[int]]) -> NdarrayOrTensor:
    """`np.any` with equivalent implementation for torch.

    For pytorch, convert to boolean for compatibility with older versions.

    Args:
        x: input array/tensor
        axis: axis to perform `any` over

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, np.ndarray):
        return np.any(x, axis)  # type: ignore

    # pytorch can't handle multiple dimensions to `any` so loop across them
    axis = [axis] if not isinstance(axis, Sequence) else axis
    for ax in axis:
        try:
            x = torch.any(x, ax)
        except RuntimeError:
            # older versions of pytorch require the input to be cast to boolean
            x = torch.any(x.bool(), ax)
    return x

def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        return (vals,)

    return tuple(vals)

def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    new_tup = ensure_tuple(tup) + (pad_val,) * dim
    return new_tup[:dim]

def no_collation(x):
    """
    No any collation operation.
    """
    return x

def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default

def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if isinstance(obj, torch.Tensor):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def fall_back_tuple(
    user_provided: Any, default: Union[Sequence, np.ndarray], func: Callable = lambda x: x and x > 0
) -> Tuple[Any, ...]:
    """
    Refine `user_provided` according to the `default`, and returns as a validated tuple.

    The validation is done for each element in `user_provided` using `func`.
    If `func(user_provided[idx])` returns False, the corresponding `default[idx]` will be used
    as the fallback.

    Typically used when `user_provided` is a tuple of window size provided by the user,
    `default` is defined by data, this function returns an updated `user_provided` with its non-positive
    components replaced by the corresponding components from `default`.

    Args:
        user_provided: item to be validated.
        default: a sequence used to provided the fallbacks.
        func: a Callable to validate every components of `user_provided`.

    Examples::

        >>> fall_back_tuple((1, 2), (32, 32))
        (1, 2)
        >>> fall_back_tuple(None, (32, 32))
        (32, 32)
        >>> fall_back_tuple((-1, 10), (32, 32))
        (32, 10)
        >>> fall_back_tuple((-1, None), (32, 32))
        (32, 32)
        >>> fall_back_tuple((1, None), (32, 32))
        (1, 32)
        >>> fall_back_tuple(0, (32, 32))
        (32, 32)
        >>> fall_back_tuple(range(3), (32, 64, 48))
        (32, 1, 2)
        >>> fall_back_tuple([0], (32, 32))
        ValueError: Sequence must have length 2, got length 1.

    """
    ndim = len(default)
    user = ensure_tuple_rep(user_provided, ndim)
    return tuple(  # use the default values if user provided is not valid
        user_c if func(user_c) else default_c for default_c, user_c in zip(default, user)
    )

def look_up_option(opt_str, supported: Union[Collection, enum.EnumMeta], default="no_default"):
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f"Unrecognized option type: {type(opt_str)}:{opt_str}.")
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, enum.EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in cast(Iterable[enum.Enum], supported)}:
            # such as: "example" in MyEnum
            return supported(opt_str)
        if isinstance(opt_str, enum.Enum) and opt_str in supported:
            # such as: MyEnum.EXAMPLE in MyEnum
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        # such as: MyDict[key]
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str

    if default != "no_default":
        return default

    # find a close match
    set_to_check: set
    if isinstance(supported, enum.EnumMeta):
        set_to_check = {item.value for item in cast(Iterable[enum.Enum], supported)}
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f"No options available: {supported}.")
    edit_dists = {}
    opt_str = f"{opt_str}"
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f"{key}", opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist

    supported_msg = f"Available options are {set_to_check}.\n"
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)  # type: ignore
        raise ValueError(
            f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n"
            + f"'{opt_str}' is not a valid option.\n"
            + supported_msg
        )
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)

def damerau_levenshtein_distance(s1: str, s2: str):
    """
    Calculates the Damerau–Levenshtein distance between two strings for spelling correction.
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    if s1 == s2:
        return 0
    string_1_length = len(s1)
    string_2_length = len(s2)
    if not s1:
        return string_2_length
    if not s2:
        return string_1_length
    d = {(i, -1): i + 1 for i in range(-1, string_1_length + 1)}
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i, s1i in enumerate(s1):
        for j, s2j in enumerate(s2):
            cost = 0 if s1i == s2j else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1, d[(i, j - 1)] + 1, d[(i - 1, j - 1)] + cost  # deletion  # insertion  # substitution
            )
            if i and j and s1i == s2[j - 1] and s1[i - 1] == s2j:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]

def map_binary_to_indices(
    label: NdarrayOrTensor, image: Optional[NdarrayOrTensor] = None, image_threshold: float = 0.0
) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
    """
    Compute the foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 3, 5, 6, 7])`` and ``background indices = np.array([0, 4, 8])``

    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.
    """

    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    label_flat = ravel(any_np_pt(label, 0))  # in case label has multiple dimensions
    fg_indices = nonzero(label_flat)
    if image is not None:
        img_flat = ravel(any_np_pt(image > image_threshold, 0))
        img_flat, *_ = convert_to_dst_type(img_flat, label, dtype=img_flat.dtype)
        bg_indices = nonzero(img_flat & ~label_flat)
    else:
        bg_indices = nonzero(~label_flat)

    # no need to save the indices in GPU, otherwise, still need to move to CPU at runtime when crop by indices
    fg_indices, *_ = convert_data_type(fg_indices, device=torch.device("cpu"))
    bg_indices, *_ = convert_data_type(bg_indices, device=torch.device("cpu"))
    return fg_indices, bg_indices

def extreme_points_to_image(
    points: List[Tuple[int, ...]],
    label: NdarrayOrTensor,
    sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.0,
    rescale_min: float = -1.0,
    rescale_max: float = 1.0,
) -> torch.Tensor:
    """
    Please refer to :py:class:`monai.transforms.AddExtremePointsChannel` for the usage.

    Applies a gaussian filter to the extreme points image. Then the pixel values in points image are rescaled
    to range [rescale_min, rescale_max].

    Args:
        points: Extreme points of the object/organ.
        label: label image to get extreme points from. Shape must be
            (1, spatial_dim1, [, spatial_dim2, ...]). Doesn't support one-hot labels.
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        rescale_min: minimum value of output data.
        rescale_max: maximum value of output data.
    """
    # points to image
    # points_image = torch.zeros(label.shape[1:], dtype=torch.float)
    points_image = torch.zeros_like(torch.as_tensor(label[0]), dtype=torch.float)
    for p in points:
        points_image[p] = 1.0

    if isinstance(sigma, Sequence):
        sigma = [torch.as_tensor(s, device=points_image.device) for s in sigma]
    else:
        sigma = torch.as_tensor(sigma, device=points_image.device)

    # add channel and add batch
    points_image = points_image.unsqueeze(0).unsqueeze(0)
    gaussian_filter = GaussianFilter(label.ndim - 1, sigma=sigma)
    points_image = gaussian_filter(points_image).squeeze(0).detach()

    # rescale the points image to [rescale_min, rescale_max]
    min_intensity = points_image.min()
    max_intensity = points_image.max()
    points_image = (points_image - min_intensity) / (max_intensity - min_intensity)
    return points_image * (rescale_max - rescale_min) + rescale_min

def get_extreme_points(
    img: NdarrayOrTensor, rand_state: Optional[np.random.RandomState] = None, background: int = 0, pert: float = 0.0
) -> List[Tuple[int, ...]]:
    """
    Generate extreme points from an image. These are used to generate initial segmentation
    for annotation models. An optional perturbation can be passed to simulate user clicks.

    Args:
        img:
            Image to generate extreme points from. Expected Shape is ``(spatial_dim1, [, spatial_dim2, ...])``.
        rand_state: `np.random.RandomState` object used to select random indices.
        background: Value to be consider as background, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Returns:
        A list of extreme points, its length is equal to 2 * spatial dimension of input image.
        The output format of the coordinates is:

        [1st_spatial_dim_min, 1st_spatial_dim_max, 2nd_spatial_dim_min, ..., Nth_spatial_dim_max]

    Raises:
        ValueError: When the input image does not have any foreground pixel.
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore
    indices = where(img != background)
    if np.size(indices[0]) == 0:
        raise ValueError("get_extreme_points: no foreground object in mask!")

    def _get_point(val, dim):
        """
        Select one of the indices within slice containing val.

        Args:
            val : value for comparison
            dim : dimension in which to look for value
        """
        idx = where(indices[dim] == val)[0]
        idx = idx.cpu() if isinstance(idx, torch.Tensor) else idx
        idx = rand_state.choice(idx)
        pt = []
        for j in range(img.ndim):
            # add +- pert to each dimension
            val = int(indices[j][idx] + 2.0 * pert * (rand_state.rand() - 0.5))
            val = max(val, 0)
            val = min(val, img.shape[j] - 1)
            pt.append(val)
        return pt

    points = []
    for i in range(img.ndim):
        points.append(tuple(_get_point(indices[i].min(), i)))
        points.append(tuple(_get_point(indices[i].max(), i)))

    return points

def where(condition: NdarrayOrTensor, x=None, y=None) -> NdarrayOrTensor:
    """
    Note that `torch.where` may convert y.dtype to x.dtype.
    """
    result: NdarrayOrTensor
    if isinstance(condition, np.ndarray):
        if x is not None:
            result = np.where(condition, x, y)
        else:
            result = np.where(condition)  # type: ignore
    else:
        if x is not None:
            x = torch.as_tensor(x, device=condition.device)
            y = torch.as_tensor(y, device=condition.device, dtype=x.dtype)
            result = torch.where(condition, x, y)
        else:
            result = torch.where(condition)  # type: ignore
    return result
def ravel(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.ravel` with equivalent implementation for torch.

    Args:
        x: array/tensor to ravel

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "ravel"):  # `ravel` is new in torch 1.8.0
            return x.ravel()
        return x.flatten().contiguous()
    return np.ravel(x)

def nonzero(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.nonzero` with equivalent implementation for torch.

    Args:
        x: array/tensor

    Returns:
        Index unravelled for given shape
    """
    if isinstance(x, np.ndarray):
        return np.nonzero(x)[0]
    return torch.nonzero(x).flatten()

def convert_to_dst_type(
    src: Any, dst: NdarrayTensor, dtype: Union[DtypeLike, torch.dtype, None] = None, wrap_sequence: bool = False
) -> Tuple[NdarrayTensor, type, Optional[torch.device]]:
    """
    Convert source data to the same data type and device as the destination data.
    If `dst` is an instance of `torch.Tensor` or its subclass, convert `src` to `torch.Tensor` with the same data type as `dst`,
    if `dst` is an instance of `numpy.ndarray` or its subclass, convert to `numpy.ndarray` with the same data type as `dst`,
    otherwise, convert to the type of `dst` directly.

    Args:
        src: source data to convert type.
        dst: destination data that convert to the same data type as it.
        dtype: an optional argument if the target `dtype` is different from the original `dst`'s data type.
        wrap_sequence: if `False`, then lists will recursively call this function. E.g., `[1, 2]` -> `[array(1), array(2)]`.
            If `True`, then `[1, 2]` -> `array([1, 2])`.

    See Also:
        :func:`convert_data_type`
    """
    device = dst.device if isinstance(dst, torch.Tensor) else None
    if dtype is None:
        dtype = dst.dtype

    output_type: Any
    if isinstance(dst, torch.Tensor):
        output_type = torch.Tensor
    elif isinstance(dst, np.ndarray):
        output_type = np.ndarray
    else:
        output_type = type(dst)
    return convert_data_type(data=src, output_type=output_type, device=device, dtype=dtype, wrap_sequence=wrap_sequence)

def convert_data_type(
    data: Any,
    output_type: Optional[Type[NdarrayTensor]] = None,
    device: Optional[torch.device] = None,
    dtype: Union[DtypeLike, torch.dtype] = None,
    wrap_sequence: bool = False,
) -> Tuple[NdarrayTensor, type, Optional[torch.device]]:
    """
    Convert to `torch.Tensor`/`np.ndarray` from `torch.Tensor`/`np.ndarray`/`float`/`int` etc.

    Args:
        data: data to be converted
        output_type: `torch.Tensor` or `np.ndarray` (if `None`, unchanged)
        device: if output is `torch.Tensor`, select device (if `None`, unchanged)
        dtype: dtype of output data. Converted to correct library type (e.g.,
            `np.float32` is converted to `torch.float32` if output type is `torch.Tensor`).
            If left blank, it remains unchanged.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
    Returns:
        modified data, orig_type, orig_device

    Note:
        When both `output_type` and `dtype` are specified with different backend
        (e.g., `torch.Tensor` and `np.float32`), the `output_type` will be used as the primary type,
        for example::

            >>> convert_data_type(1, torch.Tensor, dtype=np.float32)
            (1.0, <class 'torch.Tensor'>, None)

    """
    orig_type: type
    if isinstance(data, torch.Tensor):
        orig_type = torch.Tensor
    elif isinstance(data, np.ndarray):
        orig_type = np.ndarray
    else:
        orig_type = type(data)

    orig_device = data.device if isinstance(data, torch.Tensor) else None

    output_type = output_type or orig_type

    dtype_ = get_equivalent_dtype(dtype, output_type)

    data_: NdarrayTensor
    if issubclass(output_type, torch.Tensor):
        data_ = convert_to_tensor(data, dtype=dtype_, device=device, wrap_sequence=wrap_sequence)
        return data_, orig_type, orig_device
    if issubclass(output_type, np.ndarray):
        data_ = convert_to_numpy(data, dtype=dtype_, wrap_sequence=wrap_sequence)
        return data_, orig_type, orig_device
    raise ValueError(f"Unsupported output type: {output_type}")

def convert_to_tensor(
    data, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, wrap_sequence: bool = False
):
    """
    Utility to convert the input data to a PyTorch Tensor. If passing a dictionary, list or tuple,
    recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device, memory_format=torch.contiguous_format)  # type: ignore
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            return torch.as_tensor(data, dtype=dtype, device=device)  # type: ignore
    elif isinstance(data, list):
        list_ret = [convert_to_tensor(i, dtype=dtype, device=device) for i in data]
        return torch.as_tensor(list_ret, dtype=dtype, device=device) if wrap_sequence else list_ret  # type: ignore
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_tensor(i, dtype=dtype, device=device) for i in data)
        return torch.as_tensor(tuple_ret, dtype=dtype, device=device) if wrap_sequence else tuple_ret  # type: ignore
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

    return data


def convert_to_numpy(data, dtype: DtypeLike = None, wrap_sequence: bool = False):
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().to(dtype=get_equivalent_dtype(dtype, torch.Tensor), device="cpu").numpy()
    elif isinstance(data, (np.ndarray, float, int, bool)):
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_numpy(i, dtype=dtype) for i in data]
        return np.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_numpy(i, dtype=dtype) for i in data)
        return np.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data

def convert_inverse_interp_mode(trans_info: List, mode: str = "nearest", align_corners: Optional[bool] = None):
    """
    Change the interpolation mode when inverting spatial transforms, default to "nearest".
    This function modifies trans_info's `TraceKeys.EXTRA_INFO`.

    See also: :py:class:`monai.transform.inverse.InvertibleTransform`

    Args:
        trans_info: transforms inverse information list, contains context of every invertible transform.
        mode: target interpolation mode to convert, default to "nearest" as it's usually used to save the mode output.
        align_corners: target align corner value in PyTorch interpolation API, need to align with the `mode`.

    """
    interp_modes = [i.value for i in InterpolateMode] + [i.value for i in GridSampleMode]

    # set to string for DataLoader collation
    align_corners_ = TraceKeys.NONE if align_corners is None else align_corners

    for item in ensure_tuple(trans_info):
        if TraceKeys.EXTRA_INFO in item:
            orig_mode = item[TraceKeys.EXTRA_INFO].get("mode", None)
            if orig_mode is not None:
                if orig_mode[0] in interp_modes:
                    item[TraceKeys.EXTRA_INFO]["mode"] = [mode for _ in range(len(mode))]
                elif orig_mode in interp_modes:
                    item[TraceKeys.EXTRA_INFO]["mode"] = mode
            if "align_corners" in item[TraceKeys.EXTRA_INFO]:
                if issequenceiterable(item[TraceKeys.EXTRA_INFO]["align_corners"]):
                    item[TraceKeys.EXTRA_INFO]["align_corners"] = [align_corners_ for _ in range(len(mode))]
                else:
                    item[TraceKeys.EXTRA_INFO]["align_corners"] = align_corners_
    return trans_info

_torch_to_np_dtype = {
    torch.bool: np.dtype(bool),
    torch.uint8: np.dtype(np.uint8),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.complex64: np.dtype(np.complex64),
    torch.complex128: np.dtype(np.complex128),
}

_np_to_torch_dtype = {value: key for key, value in _torch_to_np_dtype.items()}

def dtype_torch_to_numpy(dtype):
    """Convert a torch dtype to its numpy equivalent."""
    return look_up_option(dtype, _torch_to_np_dtype)


def dtype_numpy_to_torch(dtype):
    """Convert a numpy dtype to its torch equivalent."""
    # np dtypes can be given as np.float32 and np.dtype(np.float32) so unify them
    dtype = np.dtype(dtype) if isinstance(dtype, (type, str)) else dtype
    return look_up_option(dtype, _np_to_torch_dtype)


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.

    Example::

        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))

    """
    if dtype is None:
        return None
    if data_type is torch.Tensor:
        if isinstance(dtype, torch.dtype):
            # already a torch dtype and target `data_type` is torch.Tensor
            return dtype
        return dtype_numpy_to_torch(dtype)
    if not isinstance(dtype, torch.dtype):
        # assuming the dtype is ok if it is not a torch dtype and target `data_type` is not torch.Tensor
        return dtype
    return dtype_torch_to_numpy(dtype)

def concatenate(to_cat: Sequence[NdarrayOrTensor], axis: int = 0, out=None) -> NdarrayOrTensor:
    """`np.concatenate` with equivalent implementation for torch (`torch.cat`)."""
    if isinstance(to_cat[0], np.ndarray):
        return np.concatenate(to_cat, axis, out)  # type: ignore
    return torch.cat(to_cat, dim=axis, out=out)  # type: ignore

def in1d(x, y):
    """`np.in1d` with equivalent implementation for torch."""
    if isinstance(x, np.ndarray):
        return np.in1d(x, y)
    return (x[..., None] == torch.tensor(y, device=x.device)).any(-1).view(-1)

def isfinite(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.isfinite` with equivalent implementation for torch."""
    if not isinstance(x, torch.Tensor):
        return np.isfinite(x)
    return torch.isfinite(x)

def unravel_indices(idx, shape) -> NdarrayOrTensor:
    """Computing unravel coordinates from indices.

    Args:
        idx: a sequence of indices to unravel
        shape: shape of array/tensor

    Returns:
        Stacked indices unravelled for given shape
    """
    lib_stack = torch.stack if isinstance(idx[0], torch.Tensor) else np.stack
    return lib_stack([unravel_index(i, shape) for i in idx])  # type: ignore

def cumsum(a: NdarrayOrTensor, axis=None, **kwargs) -> NdarrayOrTensor:
    """
    `np.cumsum` with equivalent implementation for torch.

    Args:
        a: input data to compute cumsum.
        axis: expected axis to compute cumsum.
        kwargs: if `a` is PyTorch Tensor, additional args for `torch.cumsum`, more details:
            https://pytorch.org/docs/stable/generated/torch.cumsum.html.

    """

    if isinstance(a, np.ndarray):
        return np.cumsum(a, axis)
    if axis is None:
        return torch.cumsum(a[:], 0, **kwargs)
    return torch.cumsum(a, dim=axis, **kwargs)

def maximum(a: NdarrayOrTensor, b: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.maximum` with equivalent implementation for torch.

    `torch.maximum` only available from pt>1.6, else use `torch.stack` and `torch.max`.

    Args:
        a: first array/tensor
        b: second array/tensor

    Returns:
        Element-wise maximum between two arrays/tensors.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # is torch and has torch.maximum (pt>1.6)
        if hasattr(torch, "maximum"):  # `maximum` is new in torch 1.7.0
            return torch.maximum(a, b)
        return torch.stack((a, b)).max(dim=0)[0]
    return np.maximum(a, b)

def moveaxis(x: NdarrayOrTensor, src: Union[int, Sequence[int]], dst: Union[int, Sequence[int]]) -> NdarrayOrTensor:
    """`moveaxis` for pytorch and numpy, using `permute` for pytorch version < 1.7"""
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "movedim"):  # `movedim` is new in torch 1.7.0
            # torch.moveaxis is a recent alias since torch 1.8.0
            return torch.movedim(x, src, dst)  # type: ignore
        return _moveaxis_with_permute(x, src, dst)
    return np.moveaxis(x, src, dst)

def _moveaxis_with_permute(
    x: torch.Tensor, src: Union[int, Sequence[int]], dst: Union[int, Sequence[int]]
) -> torch.Tensor:
    # get original indices
    indices = list(range(x.ndim))
    len_indices = len(indices)
    for s, d in zip(ensure_tuple(src), ensure_tuple(dst)):
        # make src and dst positive
        # remove desired index and insert it in new position
        pos_s = len_indices + s if s < 0 else s
        pos_d = len_indices + d if d < 0 else d
        indices.pop(pos_s)
        indices.insert(pos_d, pos_s)
    return x.permute(indices)

def allclose(a: NdarrayTensor, b: NdarrayOrTensor, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool:
    """`np.allclose` with equivalent implementation for torch."""
    b, *_ = convert_to_dst_type(b, a)
    if isinstance(a, np.ndarray):
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)  # type: ignore

def meshgrid_ij(*tensors):
    if digit_version(TORCH_VERSION)[:2] >= digit_version("1.10"):
        return torch.meshgrid(*tensors, indexing="ij")
    return torch.meshgrid(*tensors)

import torch.nn.functional as F
from torch import instance_norm, nn

class SavitzkyGolayFilter(nn.Module):
    """
    Convolve a Tensor along a particular axis with a Savitzky-Golay kernel.

    Args:
        window_length: Length of the filter window, must be a positive odd integer.
        order: Order of the polynomial to fit to each window, must be less than ``window_length``.
        axis (optional): Axis along which to apply the filter kernel. Default 2 (first spatial dimension).
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'`` or
        ``'circular'``. Default: ``'zeros'``. See torch.nn.Conv1d() for more information.
    """

    def __init__(self, window_length: int, order: int, axis: int = 2, mode: str = "zeros"):

        super().__init__()
        if order >= window_length:
            raise ValueError("order must be less than window_length.")

        self.axis = axis
        self.mode = mode
        self.coeffs = self._make_coeffs(window_length, order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to filter. Must be real, in shape ``[Batch, chns, spatial1, spatial2, ...]`` and
                have a device type of ``'cpu'``.
        Returns:
            torch.Tensor: ``x`` filtered by Savitzky-Golay kernel with window length ``self.window_length`` using
            polynomials of order ``self.order``, along axis specified in ``self.axis``.
        """

        # Make input a real tensor on the CPU
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError("Invalid axis for shape of x.")

        # Create list of filter kernels (1 per spatial dimension). The kernel for self.axis will be the savgol coeffs,
        # while the other kernels will be set to [1].
        n_spatial_dims = len(x.shape) - 2
        spatial_processing_axis = self.axis - 2
        new_dims_before = spatial_processing_axis
        new_dims_after = n_spatial_dims - spatial_processing_axis - 1
        kernel_list = [self.coeffs.to(device=x.device, dtype=x.dtype)]
        for _ in range(new_dims_before):
            kernel_list.insert(0, torch.ones(1, device=x.device, dtype=x.dtype))
        for _ in range(new_dims_after):
            kernel_list.append(torch.ones(1, device=x.device, dtype=x.dtype))

        return separable_filtering(x, kernel_list, mode=self.mode)

    @staticmethod
    def _make_coeffs(window_length, order):

        half_length, rem = divmod(window_length, 2)
        if rem == 0:
            raise ValueError("window_length must be odd.")

        idx = torch.arange(window_length - half_length - 1, -half_length - 1, -1, dtype=torch.float, device="cpu")
        a = idx ** torch.arange(order + 1, dtype=torch.float, device="cpu").reshape(-1, 1)
        y = torch.zeros(order + 1, dtype=torch.float, device="cpu")
        y[0] = 1.0
        return torch.lstsq(y, a).solution.squeeze()

def separable_filtering(x: torch.Tensor, kernels: List[torch.Tensor], mode: str = "zeros") -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.

    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all spatial dimensions), or
            a list of `spatial_dims` number of kernels.
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.

    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import separable_filtering
        >>> img = torch.randn(2, 4, 32, 32)  # batch_size 2, channels 4, 32x32 2D images
        # applying a [-1, 0, 1] filter along each of the spatial dimensions.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, torch.tensor((-1., 0., 1.)))
        # applying `[-1, 0, 1]`, `[1, 0, -1]` filters along two spatial dimensions respectively.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, [torch.tensor((-1., 0., 1.)), torch.tensor((1., 0., -1.))])

    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    _kernels = [s.to(x) for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode

    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims - 1, spatial_dims, _paddings, n_chs)

def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: List[torch.Tensor],
    pad_mode: str,
    d: int,
    spatial_dims: int,
    paddings: List[int],
    num_channels: int,
) -> torch.Tensor:

    if d < 0:
        return input_

    s = [1] * len(input_.shape)
    s[d + 2] = -1
    _kernel = kernels[d].reshape(s)

    # if filter kernel is unity, don't convolve
    if _kernel.numel() == 1 and _kernel[0] == 1:
        return _separable_filtering_conv(input_, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels)

    _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
    _padding = [0] * spatial_dims
    _padding[d] = paddings[d]
    conv_type = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

    # translate padding for input to torch.nn.functional.pad
    _reversed_padding_repeated_twice: List[List[int]] = [[p, p] for p in reversed(_padding)]
    _sum_reversed_padding_repeated_twice: List[int] = sum(_reversed_padding_repeated_twice, [])
    padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)

    return conv_type(
        input=_separable_filtering_conv(padded_input, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels),
        weight=_kernel,
        groups=num_channels,
    )

if digit_version(TORCH_VERSION)[:2] >= digit_version('1.7'):

    import torch.fft as fft

    class HilbertTransform(nn.Module):
        """
        Determine the analytical signal of a Tensor along a particular axis.
        Requires PyTorch 1.7.0+ and the PyTorch FFT module (which is not included in NVIDIA PyTorch Release 20.10).

        Args:
            axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
            n: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.
        """

        def __init__(self, axis: int = 2, n: Union[int, None] = None) -> None:
            super().__init__()
            self.axis = axis
            self.n = n

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
            Returns:
                torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
                FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
            """

            # Make input a real tensor
            x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
            if torch.is_complex(x):
                raise ValueError("x must be real.")
            x = x.to(dtype=torch.float)

            if (self.axis < 0) or (self.axis > len(x.shape) - 1):
                raise ValueError("Invalid axis for shape of x.")

            n = x.shape[self.axis] if self.n is None else self.n
            if n <= 0:
                raise ValueError("N must be positive.")
            x = torch.as_tensor(x, dtype=torch.complex64)
            # Create frequency axis
            f = torch.cat(
                [
                    torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
                    torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
                ]
            )
            xf = fft.fft(x, n=n, dim=self.axis)
            # Create step function
            u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
            u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
            new_dims_before = self.axis
            new_dims_after = len(xf.shape) - self.axis - 1
            for _ in range(new_dims_before):
                u.unsqueeze_(0)
            for _ in range(new_dims_after):
                u.unsqueeze_(-1)

            ht = fft.ifft(xf * 2 * u, dim=self.axis)

            # Apply transform
            return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)

    class Fourier:
        """
        Helper class storing Fourier mappings
        """
        @staticmethod
        def shift_fourier(x: NdarrayOrTensor, spatial_dims: int, n_dims: Optional[int] = None) -> NdarrayOrTensor:
            """
            Applies fourier transform and shifts the zero-frequency component to the
            center of the spectrum. Only the spatial dimensions get transformed.

            Args:
                x: Image to transform.
                spatial_dims: Number of spatial dimensions.



            Returns
                k: K-space data.
            """
            if n_dims is not None:
                spatial_dims = n_dims
            dims = tuple(range(-spatial_dims, 0))
            k: NdarrayOrTensor
            if isinstance(x, torch.Tensor):
                if hasattr(torch.fft, "fftshift"):  # `fftshift` is new in torch 1.8.0
                    k = torch.fft.fftshift(torch.fft.fftn(x, dim=dims), dim=dims)
                else:
                    # if using old PyTorch, will convert to numpy array and return
                    k = np.fft.fftshift(np.fft.fftn(x.cpu().numpy(), axes=dims), axes=dims)
            else:
                k = np.fft.fftshift(np.fft.fftn(x, axes=dims), axes=dims)
            return k

        @staticmethod
        def inv_shift_fourier(k: NdarrayOrTensor, spatial_dims: int, n_dims: Optional[int] = None) -> NdarrayOrTensor:
            """
            Applies inverse shift and fourier transform. Only the spatial
            dimensions are transformed.

            Args:
                k: K-space data.
                spatial_dims: Number of spatial dimensions.

            Returns:
                x: Tensor in image space.
            """
            if n_dims is not None:
                spatial_dims = n_dims
            dims = tuple(range(-spatial_dims, 0))
            out: NdarrayOrTensor
            if isinstance(k, torch.Tensor):
                if hasattr(torch.fft, "ifftshift"):  # `ifftshift` is new in torch 1.8.0
                    out = torch.fft.ifftn(torch.fft.ifftshift(k, dim=dims), dim=dims, norm="backward").real
                else:
                    # if using old PyTorch, will convert to numpy array and return
                    out = np.fft.ifftn(np.fft.ifftshift(k.cpu().numpy(), axes=dims), axes=dims).real
            else:
                out = np.fft.ifftn(np.fft.ifftshift(k, axes=dims), axes=dims).real
            return out
else:
    print("Fourier and HilbertTransform requires pytorch version >= 1.7.0 or later.")

class GaussianFilter(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor],
        truncated: float = 4.0,
        approx: str = "erf",
        requires_grad: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std. could be a single value, or `spatial_dims` number of values.
            truncated: spreads how many stds.
            approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

                - ``erf`` approximation interpolates the error function;
                - ``sampled`` uses a sampled Gaussian kernel;
                - ``scalespace`` corresponds to
                  https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
                  based on the modified Bessel functions.

            requires_grad: whether to store the gradients for sigma.
                if True, `sigma` will be the initial value of the parameters of this module
                (for example `parameters()` iterator could be used to get the parameters);
                otherwise this module will fix the kernels using `sigma` as the std.
        """
        if issequenceiterable(sigma):
            if len(sigma) != spatial_dims:  # type: ignore
                raise ValueError
        else:
            sigma = [deepcopy(sigma) for _ in range(spatial_dims)]  # type: ignore
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(s, dtype=torch.float, device=s.device if isinstance(s, torch.Tensor) else None),
                requires_grad=requires_grad,
            )
            for s in sigma  # type: ignore
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].
        """
        _kernel = [gaussian_1d(s, truncated=self.truncated, approx=self.approx) for s in self.sigma]
        return separable_filtering(x=x, kernels=_kernel)

def gaussian_1d(
    sigma: torch.Tensor, truncated: float = 4.0, approx: str = "erf", normalize: bool = False
) -> torch.Tensor:
    """
    one dimensional Gaussian kernel.

    Args:
        sigma: std of the kernel
        truncated: tail length
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

            - ``erf`` approximation interpolates the error function;
            - ``sampled`` uses a sampled Gaussian kernel;
            - ``scalespace`` corresponds to
              https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
              based on the modified Bessel functions.

        normalize: whether to normalize the kernel with `kernel.sum()`.

    Raises:
        ValueError: When ``truncated`` is non-positive.

    Returns:
        1D torch tensor

    """
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=sigma.device if isinstance(sigma, torch.Tensor) else None)
    device = sigma.device
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        t = 0.70710678 / torch.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=sigma.device)
        out = torch.exp(-0.5 / (sigma * sigma) * x**2)
        if not normalize:  # compute the normalizer
            out = out / (2.5066282 * sigma)
    elif approx.lower() == "scalespace":
        sigma2 = sigma * sigma
        out_pos: List[Optional[torch.Tensor]] = [None] * (tail + 1)
        out_pos[0] = _modified_bessel_0(sigma2)
        out_pos[1] = _modified_bessel_1(sigma2)
        for k in range(2, len(out_pos)):
            out_pos[k] = _modified_bessel_i(k, sigma2)
        out = out_pos[:0:-1]
        out.extend(out_pos)
        out = torch.stack(out) * torch.exp(-sigma2)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / out.sum() if normalize else out  # type: ignore

def _modified_bessel_0(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if isinstance(x, torch.Tensor) else None)
    if torch.abs(x) < 3.75:
        y = x * x / 14.0625
        return polyval([0.45813e-2, 0.360768e-1, 0.2659732, 1.2067492, 3.0899424, 3.5156229, 1.0], y)
    ax = torch.abs(x)
    y = 3.75 / ax
    _coef = [
        0.392377e-2,
        -0.1647633e-1,
        0.2635537e-1,
        -0.2057706e-1,
        0.916281e-2,
        -0.157565e-2,
        0.225319e-2,
        0.1328592e-1,
        0.39894228,
    ]
    return polyval(_coef, y) * torch.exp(ax) / torch.sqrt(ax)


def _modified_bessel_1(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if isinstance(x, torch.Tensor) else None)
    if torch.abs(x) < 3.75:
        y = x * x / 14.0625
        _coef = [0.32411e-3, 0.301532e-2, 0.2658733e-1, 0.15084934, 0.51498869, 0.87890594, 0.5]
        return torch.abs(x) * polyval(_coef, y)
    ax = torch.abs(x)
    y = 3.75 / ax
    _coef = [
        -0.420059e-2,
        0.1787654e-1,
        -0.2895312e-1,
        0.2282967e-1,
        -0.1031555e-1,
        0.163801e-2,
        -0.362018e-2,
        -0.3988024e-1,
        0.39894228,
    ]
    ans = polyval(_coef, y) * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < 0.0 else ans

def _modified_bessel_i(n: int, x: torch.Tensor) -> torch.Tensor:
    if n < 2:
        raise ValueError(f"n must be greater than 1, got n={n}.")
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if isinstance(x, torch.Tensor) else None)
    if x == 0.0:
        return x
    device = x.device
    tox = 2.0 / torch.abs(x)
    ans, bip, bi = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    m = int(2 * (n + np.floor(np.sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        if abs(bi) > 1.0e10:
            ans = ans * 1.0e-10
            bi = bi * 1.0e-10
            bip = bip * 1.0e-10
        if j == n:
            ans = bip
    ans = ans * _modified_bessel_0(x) / bi
    return -ans if x < 0.0 and (n % 2) == 1 else ans


def polyval(coef, x) -> torch.Tensor:
    """
    Evaluates the polynomial defined by `coef` at `x`.

    For a 1D sequence of coef (length n), evaluate::

        y = coef[n-1] + x * (coef[n-2] + ... + x * (coef[1] + x * coef[0]))

    Args:
        coef: a sequence of floats representing the coefficients of the polynomial
        x: float or a sequence of floats representing the variable of the polynomial

    Returns:
        1D torch tensor
    """
    device = x.device if isinstance(x, torch.Tensor) else None
    coef = torch.as_tensor(coef, dtype=torch.float, device=device)
    if coef.ndim == 0 or (len(coef) < 1):
        return torch.zeros(x.shape)
    x = torch.as_tensor(x, dtype=torch.float, device=device)
    ans = coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans  # type: ignore

exposure, has_skimage = optional_import("skimage.exposure")

def equalize_hist(
    img: np.ndarray, mask: Optional[np.ndarray] = None, num_bins: int = 256, min: int = 0, max: int = 255
) -> np.ndarray:
    """
    Utility to equalize input image based on the histogram.
    If `skimage` installed, will leverage `skimage.exposure.histogram`, otherwise, use
    `np.histogram` instead.

    Args:
        img: input image to equalize.
        mask: if provided, must be ndarray of bools or 0s and 1s, and same shape as `image`.
            only points at which `mask==True` are used for the equalization.
        num_bins: number of the bins to use in histogram, default to `256`. for more details:
            https://numpy.org/doc/stable/reference/generated/numpy.histogram.html.
        min: the min value to normalize input image, default to `0`.
        max: the max value to normalize input image, default to `255`.

    """

    orig_shape = img.shape
    hist_img = img[np.array(mask, dtype=bool)] if mask is not None else img
    if has_skimage:
        hist, bins = exposure.histogram(hist_img.flatten(), num_bins)
    else:
        hist, bins = np.histogram(hist_img.flatten(), num_bins)
        bins = (bins[:-1] + bins[1:]) / 2

    cum = hist.cumsum()
    # normalize the cumulative result
    cum = rescale_array(arr=cum, minv=min, maxv=max)

    # apply linear interpolation
    img = np.interp(img.flatten(), bins, cum)
    return img.reshape(orig_shape)

def rescale_array(
    arr: NdarrayOrTensor,
    minv: Optional[float] = 0.0,
    maxv: Optional[float] = 1.0,
    dtype: Union[DtypeLike, torch.dtype] = np.float32,
) -> NdarrayOrTensor:
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    If either `minv` or `maxv` is None, it returns `(a - min_a) / (max_a - min_a)`.

    Args:
        arr: input array to rescale.
        minv: minimum value of target rescaled array.
        maxv: maxmum value of target rescaled array.
        dtype: if not None, convert input array to dtype before computation.

    """
    if dtype is not None:
        arr, *_ = convert_data_type(arr, dtype=dtype)
    mina = arr.min()
    maxa = arr.max()

    if mina == maxa:
        return arr * minv if minv is not None else arr

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    if (minv is None) or (maxv is None):
        return norm
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default

def clip(a: NdarrayOrTensor, a_min, a_max) -> NdarrayOrTensor:
    """`np.clip` with equivalent implementation for torch."""
    result: NdarrayOrTensor
    if isinstance(a, np.ndarray):
        result = np.clip(a, a_min, a_max)
    else:
        result = torch.clamp(a, a_min, a_max)
    return result


def percentile(
    x: NdarrayOrTensor, q, dim: Optional[int] = None, keepdim: bool = False, **kwargs
) -> Union[NdarrayOrTensor, float, int]:
    """`np.percentile` with equivalent implementation for torch.

    Pytorch uses `quantile`, but this functionality is only available from v1.7.
    For earlier methods, we calculate it ourselves. This doesn't do interpolation,
    so is the equivalent of ``numpy.percentile(..., interpolation="nearest")``.
    For more details, please refer to:
    https://pytorch.org/docs/stable/generated/torch.quantile.html.
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html.

    Args:
        x: input data
        q: percentile to compute (should in range 0 <= q <= 100)
        dim: the dim along which the percentiles are computed. default is to compute the percentile
            along a flattened version of the array. only work for numpy array or Tensor with PyTorch >= 1.7.0.
        keepdim: whether the output data has dim retained or not.
        kwargs: if `x` is numpy array, additional args for `np.percentile`, more details:
            https://numpy.org/doc/stable/reference/generated/numpy.percentile.html.

    Returns:
        Resulting value (scalar)
    """
    if np.isscalar(q):
        if not 0 <= q <= 100:  # type: ignore
            raise ValueError
    elif any(q < 0) or any(q > 100):
        raise ValueError
    result: Union[NdarrayOrTensor, float, int]
    if isinstance(x, np.ndarray):
        result = np.percentile(x, q, axis=dim, keepdims=keepdim, **kwargs)
    else:
        q = torch.tensor(q, device=x.device)
        if hasattr(torch, "quantile"):  # `quantile` is new in torch 1.7.0
            result = torch.quantile(x, q / 100.0, dim=dim, keepdim=keepdim)
        else:
            # Note that ``kthvalue()`` works one-based, i.e., the first sorted value
            # corresponds to k=1, not k=0. Thus, we need the `1 +`.
            k = 1 + (0.01 * q * (x.numel() - 1)).round().int()
            if k.numel() > 1:
                r = [x.view(-1).kthvalue(int(_k)).values.item() for _k in k]
                result = torch.tensor(r, device=x.device)
            else:
                result = x.view(-1).kthvalue(int(k)).values.item()

    return result


def where(condition: NdarrayOrTensor, x=None, y=None) -> NdarrayOrTensor:
    """
    Note that `torch.where` may convert y.dtype to x.dtype.
    """
    result: NdarrayOrTensor
    if isinstance(condition, np.ndarray):
        if x is not None:
            result = np.where(condition, x, y)
        else:
            result = np.where(condition)  # type: ignore
    else:
        if x is not None:
            x = torch.as_tensor(x, device=condition.device)
            y = torch.as_tensor(y, device=condition.device, dtype=x.dtype)
            result = torch.where(condition, x, y)
        else:
            result = torch.where(condition)  # type: ignore
    return result

def fill_holes(
    img_arr: np.ndarray, applied_labels: Optional[Iterable[int]] = None, connectivity: Optional[int] = None
) -> np.ndarray:
    from scipy import ndimage
    """
    Fill the holes in the provided image.

    The label 0 will be treated as background and the enclosed holes will be set to the neighboring class label.
    What is considered to be an enclosed hole is defined by the connectivity.
    Holes on the edge are always considered to be open (not enclosed).

    Note:

        The performance of this method heavily depends on the number of labels.
        It is a bit faster if the list of `applied_labels` is provided.
        Limiting the number of `applied_labels` results in a big decrease in processing time.

        If the image is one-hot-encoded, then the `applied_labels` need to match the channel index.

    Args:
        img_arr: numpy array of shape [C, spatial_dim1[, spatial_dim2, ...]].
        applied_labels: Labels for which to fill holes. Defaults to None,
            that is filling holes for all labels.
        connectivity: Maximum number of orthogonal hops to
            consider a pixel/voxel as a neighbor. Accepted values are ranging from  1 to input.ndim.
            Defaults to a full connectivity of ``input.ndim``.

    Returns:
        numpy array of shape [C, spatial_dim1[, spatial_dim2, ...]].
    """
    channel_axis = 0
    num_channels = img_arr.shape[channel_axis]
    is_one_hot = num_channels > 1
    spatial_dims = img_arr.ndim - 1
    structure = ndimage.generate_binary_structure(spatial_dims, connectivity or spatial_dims)

    # Get labels if not provided. Exclude background label.
    applied_labels = set(applied_labels or (range(num_channels) if is_one_hot else np.unique(img_arr)))
    background_label = 0
    applied_labels.discard(background_label)

    for label in applied_labels:
        tmp = np.zeros(img_arr.shape[1:], dtype=bool)
        ndimage.binary_dilation(
            tmp,
            structure=structure,
            iterations=-1,
            mask=np.logical_not(img_arr[label]) if is_one_hot else img_arr[0] != label,
            origin=0,
            border_value=1,
            output=tmp,
        )
        if is_one_hot:
            img_arr[label] = np.logical_not(tmp)
        else:
            img_arr[0, np.logical_not(tmp)] = label

    return img_arr

def get_largest_connected_component_mask(img: NdarrayTensor, connectivity: Optional[int] = None) -> NdarrayTensor:
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used. for more details:
            https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.
    """
    from skimage import measure
    img_arr = convert_data_type(img, np.ndarray)[0]
    largest_cc: np.ndarray = np.zeros(shape=img_arr.shape, dtype=img_arr.dtype)
    img_arr = measure.label(img_arr, connectivity=connectivity)
    if img_arr.max() != 0:
        largest_cc[...] = img_arr == (np.argmax(np.bincount(img_arr.flat)[1:]) + 1)

    return convert_to_dst_type(largest_cc, dst=img, dtype=largest_cc.dtype)[0]

def apply_filter(x: torch.Tensor, kernel: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Filtering `x` with `kernel` independently for each batch and channel respectively.

    Args:
        x: the input image, must have shape (batch, channels, H[, W, D]).
        kernel: `kernel` must at least have the spatial shape (H_k[, W_k, D_k]).
            `kernel` shape must be broadcastable to the `batch` and `channels` dimensions of `x`.
        kwargs: keyword arguments passed to `conv*d()` functions.

    Returns:
        The filtered `x`.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import apply_filter
        >>> img = torch.rand(2, 5, 10, 10)  # batch_size 2, channels 5, 10x10 2D images
        >>> out = apply_filter(img, torch.rand(3, 3))   # spatial kernel
        >>> out = apply_filter(img, torch.rand(5, 3, 3))  # channel-wise kernels
        >>> out = apply_filter(img, torch.rand(2, 5, 3, 3))  # batch-, channel-wise kernels

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
    batch, chns, *spatials = x.shape
    n_spatial = len(spatials)
    if n_spatial > 3:
        raise NotImplementedError(f"Only spatial dimensions up to 3 are supported but got {n_spatial}.")
    k_size = len(kernel.shape)
    if k_size < n_spatial or k_size > n_spatial + 2:
        raise ValueError(
            f"kernel must have {n_spatial} ~ {n_spatial + 2} dimensions to match the input shape {x.shape}."
        )
    kernel = kernel.to(x)
    # broadcast kernel size to (batch chns, spatial_kernel_size)
    kernel = kernel.expand(batch, chns, *kernel.shape[(k_size - n_spatial) :])
    kernel = kernel.reshape(-1, 1, *kernel.shape[2:])  # group=1
    x = x.view(1, kernel.shape[0], *spatials)
    conv = [F.conv1d, F.conv2d, F.conv3d][n_spatial - 1]
    if "padding" not in kwargs:
        if digit_version(TORCH_VERSION)[:2] >= digit_version("1.10"):
            kwargs["padding"] = "same"
        else:
            # even-sized kernels are not supported
            kwargs["padding"] = [(k - 1) // 2 for k in kernel.shape[2:]]

    if "stride" not in kwargs:
        kwargs["stride"] = 1
    output = conv(x, kernel, groups=kernel.shape[0], bias=None, **kwargs)
    return output.view(batch, chns, *output.shape[2:])

def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])
    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

AFFINE_TOL = 1e-3
nib, _ = optional_import("nibabel")

def compute_shape_offset(
    spatial_shape: Union[np.ndarray, Sequence[int]], in_affine: NdarrayOrTensor, out_affine: NdarrayOrTensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given input and output affine, compute appropriate shapes
    in the output space based on the input array's shape.
    This function also returns the offset to put the shape
    in a good position with respect to the world coordinate system.

    Args:
        spatial_shape: input array's shape
        in_affine (matrix): 2D affine matrix
        out_affine (matrix): 2D affine matrix
    """
    shape = np.array(spatial_shape, copy=True, dtype=float)
    sr = len(shape)
    in_affine_ = convert_data_type(to_affine_nd(sr, in_affine), np.ndarray)[0]
    out_affine_ = convert_data_type(to_affine_nd(sr, out_affine), np.ndarray)[0]
    in_coords = [(0.0, dim - 1.0) for dim in shape]
    corners: np.ndarray = np.asarray(np.meshgrid(*in_coords, indexing="ij")).reshape((len(shape), -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    corners = in_affine_ @ corners
    try:
        inv_mat = np.linalg.inv(out_affine_)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Affine {out_affine_} is not invertible") from e
    corners_out = inv_mat @ corners
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(corners_out.ptp(axis=1) + 1.0)
    mat = inv_mat[:-1, :-1]
    k = 0
    for i in range(corners.shape[1]):
        min_corner = np.min(mat @ corners[:-1, :] - mat @ corners[:-1, i : i + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            k = i
            break
    offset = corners[:-1, k]
    return out_shape.astype(int, copy=False), offset

def to_affine_nd(r: Union[np.ndarray, int], affine: NdarrayTensor, dtype=np.float64) -> NdarrayTensor:
    """
    Using elements from affine, to create a new affine matrix by
    assigning the rotation/zoom/scaling matrix and the translation vector.

    When ``r`` is an integer, output is an (r+1)x(r+1) matrix,
    where the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(r, len(affine) - 1)`.

    When ``r`` is an affine matrix, the output has the same shape as ``r``,
    and the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(len(r) - 1, len(affine) - 1)`.

    Args:
        r (int or matrix): number of spatial dimensions or an output affine to be filled.
        affine (matrix): 2D affine matrix
        dtype: data type of the output array.

    Raises:
        ValueError: When ``affine`` dimensions is not 2.
        ValueError: When ``r`` is nonpositive.

    Returns:
        an (r+1) x (r+1) matrix (tensor or ndarray depends on the input ``affine`` data type)

    """
    affine_np = convert_data_type(affine, output_type=np.ndarray, dtype=dtype, wrap_sequence=True)[0]
    affine_np = affine_np.copy()
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=dtype, copy=True)
    if new_affine.ndim == 0:
        sr: int = int(new_affine.astype(np.uint))
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=dtype)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    output, *_ = convert_to_dst_type(new_affine, affine, dtype=dtype)
    return output

def reorient_spatial_axes(
    data_shape: Sequence[int], init_affine: NdarrayOrTensor, target_affine: NdarrayOrTensor
) -> Tuple[np.ndarray, NdarrayOrTensor]:
    """
    Given the input ``init_affine``, compute the orientation transform between
    it and ``target_affine`` by rearranging/flipping the axes.

    Returns the orientation transform and the updated affine (tensor or ndarray
    depends on the input ``affine`` data type).
    Note that this function requires external module ``nibabel.orientations``.
    """
    init_affine_, *_ = convert_data_type(init_affine, np.ndarray)
    target_affine_, *_ = convert_data_type(target_affine, np.ndarray)
    start_ornt = nib.orientations.io_orientation(init_affine_)
    target_ornt = nib.orientations.io_orientation(target_affine_)
    try:
        ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
    except ValueError as e:
        raise ValueError(f"The input affine {init_affine} and target affine {target_affine} are not compatible.") from e
    new_affine = init_affine_ @ nib.orientations.inv_ornt_aff(ornt_transform, data_shape)
    new_affine, *_ = convert_to_dst_type(new_affine, init_affine)
    return ornt_transform, new_affine

def to_affine_nd(r: Union[np.ndarray, int], affine: NdarrayTensor, dtype=np.float64) -> NdarrayTensor:
    """
    Using elements from affine, to create a new affine matrix by
    assigning the rotation/zoom/scaling matrix and the translation vector.

    When ``r`` is an integer, output is an (r+1)x(r+1) matrix,
    where the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(r, len(affine) - 1)`.

    When ``r`` is an affine matrix, the output has the same shape as ``r``,
    and the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(len(r) - 1, len(affine) - 1)`.

    Args:
        r (int or matrix): number of spatial dimensions or an output affine to be filled.
        affine (matrix): 2D affine matrix
        dtype: data type of the output array.

    Raises:
        ValueError: When ``affine`` dimensions is not 2.
        ValueError: When ``r`` is nonpositive.

    Returns:
        an (r+1) x (r+1) matrix (tensor or ndarray depends on the input ``affine`` data type)

    """
    affine_np = convert_data_type(affine, output_type=np.ndarray, dtype=dtype, wrap_sequence=True)[0]
    affine_np = affine_np.copy()
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=dtype, copy=True)
    if new_affine.ndim == 0:
        sr: int = int(new_affine.astype(np.uint))
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=dtype)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    output, *_ = convert_to_dst_type(new_affine, affine, dtype=dtype)
    return output

def zoom_affine(affine: np.ndarray, scale: Union[np.ndarray, Sequence[float]], diagonal: bool = True):
    """
    To make column norm of `affine` the same as `scale`.  If diagonal is False,
    returns an affine that combines orthogonal rotation and the new scale.
    This is done by first decomposing `affine`, then setting the zoom factors to
    `scale`, and composing a new affine; the shearing factors are removed.  If
    diagonal is True, returns a diagonal matrix, the scaling factors are set
    to the diagonal elements.  This function always return an affine with zero
    translations.

    Args:
        affine (nxn matrix): a square matrix.
        scale: new scaling factor along each dimension. if the components of the `scale` are non-positive values,
            will use the corresponding components of the original pixdim, which is computed from the `affine`.
        diagonal: whether to return a diagonal scaling matrix.
            Defaults to True.

    Raises:
        ValueError: When ``affine`` is not a square matrix.
        ValueError: When ``scale`` contains a nonpositive scalar.

    Returns:
        the updated `n x n` affine.

    """

    affine = np.array(affine, dtype=float, copy=True)
    if len(affine) != len(affine[0]):
        raise ValueError(f"affine must be n x n, got {len(affine)} x {len(affine[0])}.")
    scale_np = np.array(scale, dtype=float, copy=True)

    d = len(affine) - 1
    # compute original pixdim
    norm = affine_to_spacing(affine, r=d)
    if len(scale_np) < d:  # defaults based on affine
        scale_np = np.append(scale_np, norm[len(scale_np) :])
    scale_np = scale_np[:d]
    scale_np = np.asarray(fall_back_tuple(scale_np, norm))

    scale_np[scale_np == 0] = 1.0
    if diagonal:
        return np.diag(np.append(scale_np, [1.0]))
    rzs = affine[:-1, :-1]  # rotation zoom scale
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    s = np.sign(np.diag(zs)) * np.abs(scale_np)
    # construct new affine with rotation and zoom
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(s)
    return new_affine

def affine_to_spacing(affine: NdarrayTensor, r: int = 3, dtype=float, suppress_zeros: bool = True) -> NdarrayTensor:
    """
    Computing the current spacing from the affine matrix.

    Args:
        affine: a d x d affine matrix.
        r: indexing based on the spatial rank, spacing is computed from `affine[:r, :r]`.
        dtype: data type of the output.
        suppress_zeros: whether to surpress the zeros with ones.

    Returns:
        an `r` dimensional vector of spacing.
    """
    _affine, *_ = convert_to_dst_type(affine[:r, :r], dst=affine, dtype=dtype)
    if isinstance(_affine, torch.Tensor):
        spacing = torch.sqrt(torch.sum(_affine * _affine, dim=0))
    else:
        spacing = np.sqrt(np.sum(_affine * _affine, axis=0))
    if suppress_zeros:
        spacing[spacing == 0] = 1.0
    spacing_, *_ = convert_to_dst_type(spacing, dst=affine, dtype=dtype)
    return spacing_

def to_norm_affine(
    affine: torch.Tensor, src_size: Sequence[int], dst_size: Sequence[int], align_corners: bool = False
) -> torch.Tensor:
    """
    Given ``affine`` defined for coordinates in the pixel space, compute the corresponding affine
    for the normalized coordinates.

    Args:
        affine: Nxdxd batched square matrix
        src_size: source image spatial shape
        dst_size: target image spatial shape
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample

    Raises:
        TypeError: When ``affine`` is not a ``torch.Tensor``.
        ValueError: When ``affine`` is not Nxdxd.
        ValueError: When ``src_size`` or ``dst_size`` dimensions differ from ``affine``.

    """
    if not isinstance(affine, torch.Tensor):
        raise TypeError(f"affine must be a torch.Tensor but is {type(affine).__name__}.")
    if affine.ndimension() != 3 or affine.shape[1] != affine.shape[2]:
        raise ValueError(f"affine must be Nxdxd, got {tuple(affine.shape)}.")
    sr = affine.shape[1] - 1
    if sr != len(src_size) or sr != len(dst_size):
        raise ValueError(f"affine suggests {sr}D, got src={len(src_size)}D, dst={len(dst_size)}D.")

    src_xform = normalize_transform(src_size, affine.device, affine.dtype, align_corners)
    dst_xform = normalize_transform(dst_size, affine.device, affine.dtype, align_corners)
    return src_xform @ affine @ torch.inverse(dst_xform)

def normalize_transform(
    shape: Sequence[int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Compute an affine matrix according to the input shape.
    The transform normalizes the homogeneous image coordinates to the
    range of `[-1, 1]`.

    Args:
        shape: input spatial shape
        device: device on which the returned affine will be allocated.
        dtype: data type of the returned affine
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
    """
    norm = torch.tensor(shape, dtype=torch.float64, device=device)  # no in-place change
    if align_corners:
        norm[norm <= 1.0] = 2.0
        norm = 2.0 / (norm - 1.0)
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        norm[:-1, -1] = -1.0
    else:
        norm[norm <= 0.0] = 2.0
        norm = 2.0 / norm
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        norm[:-1, -1] = 1.0 / torch.tensor(shape, dtype=torch.float64, device=device) - 1.0
    norm = norm.unsqueeze(0).to(dtype=dtype)
    norm.requires_grad = False
    return norm

def map_spatial_axes(
    img_ndim: int, spatial_axes: Optional[Union[Sequence[int], int]] = None, channel_first: bool = True
) -> List[int]:
    """
    Utility to map the spatial axes to real axes in channel first/last shape.
    For example:
    If `channel_first` is True, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [1, 2, 3]
    [0, 1] -> [1, 2]
    [0, -1] -> [1, -1]
    If `channel_first` is False, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [0, 1, 2]
    [0, 1] -> [0, 1]
    [0, -1] -> [0, -2]

    Args:
        img_ndim: dimension number of the target image.
        spatial_axes: spatial axes to be converted, default is None.
            The default `None` will convert to all the spatial axes of the image.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints.
        channel_first: the image data is channel first or channel last, default to channel first.

    """
    if spatial_axes is None:
        spatial_axes_ = list(range(1, img_ndim) if channel_first else range(img_ndim - 1))

    else:
        spatial_axes_ = []
        for a in ensure_tuple(spatial_axes):
            if channel_first:
                spatial_axes_.append(a if a < 0 else a + 1)
            else:
                spatial_axes_.append(a - 1 if a < 0 else a)

    return spatial_axes_

def create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
    backend=TransformBackends.NUMPY,
) -> NdarrayOrTensor:
    """
    create a 2D or 3D rotation matrix

    Args:
        spatial_dims: {``2``, ``3``} spatial rank
        radians: rotation radians
            when spatial_dims == 3, the `radians` sequence corresponds to
            rotation in the 1st, 2nd, and 3rd dim respectively.
        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.

    Raises:
        ValueError: When ``radians`` is empty.
        ValueError: When ``spatial_dims`` is not one of [2, 3].

    """
    _backend = look_up_option(backend, TransformBackends)
    if _backend == TransformBackends.NUMPY:
        return _create_rotate(
            spatial_dims=spatial_dims, radians=radians, sin_func=np.sin, cos_func=np.cos, eye_func=np.eye
        )
    if _backend == TransformBackends.TORCH:
        return _create_rotate(
            spatial_dims=spatial_dims,
            radians=radians,
            sin_func=lambda th: torch.sin(torch.as_tensor(th, dtype=torch.float32, device=device)),
            cos_func=lambda th: torch.cos(torch.as_tensor(th, dtype=torch.float32, device=device)),
            eye_func=lambda rank: torch.eye(rank, device=device),
        )
    raise ValueError(f"backend {backend} is not supported")


def _create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    sin_func: Callable = np.sin,
    cos_func: Callable = np.cos,
    eye_func: Callable = np.eye,
) -> NdarrayOrTensor:
    radians = ensure_tuple(radians)
    if spatial_dims == 2:
        if len(radians) >= 1:
            sin_, cos_ = sin_func(radians[0]), cos_func(radians[0])
            out = eye_func(3)
            out[0, 0], out[0, 1] = cos_, -sin_
            out[1, 0], out[1, 1] = sin_, cos_
            return out  # type: ignore
        raise ValueError("radians must be non empty.")

    if spatial_dims == 3:
        affine = None
        if len(radians) >= 1:
            sin_, cos_ = sin_func(radians[0]), cos_func(radians[0])
            affine = eye_func(4)
            affine[1, 1], affine[1, 2] = cos_, -sin_
            affine[2, 1], affine[2, 2] = sin_, cos_
        if len(radians) >= 2:
            sin_, cos_ = sin_func(radians[1]), cos_func(radians[1])
            if affine is None:
                raise ValueError("Affine should be a matrix.")
            _affine = eye_func(4)
            _affine[0, 0], _affine[0, 2] = cos_, sin_
            _affine[2, 0], _affine[2, 2] = -sin_, cos_
            affine = affine @ _affine
        if len(radians) >= 3:
            sin_, cos_ = sin_func(radians[2]), cos_func(radians[2])
            if affine is None:
                raise ValueError("Affine should be a matrix.")
            _affine = eye_func(4)
            _affine[0, 0], _affine[0, 1] = cos_, -sin_
            _affine[1, 0], _affine[1, 1] = sin_, cos_
            affine = affine @ _affine
        if affine is None:
            raise ValueError("radians must be non empty.")
        return affine  # type: ignore

    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}, available options are [2, 3].")

def create_translate(
    spatial_dims: int,
    shift: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
    backend=TransformBackends.NUMPY,
) -> NdarrayOrTensor:
    """
    create a translation matrix

    Args:
        spatial_dims: spatial rank
        shift: translate pixel/voxel for every spatial dim, defaults to 0.
        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.
    """
    _backend = look_up_option(backend, TransformBackends)
    if _backend == TransformBackends.NUMPY:
        return _create_translate(spatial_dims=spatial_dims, shift=shift, eye_func=np.eye, array_func=np.asarray)
    if _backend == TransformBackends.TORCH:
        return _create_translate(
            spatial_dims=spatial_dims,
            shift=shift,
            eye_func=lambda x: torch.eye(torch.as_tensor(x), device=device),  # type: ignore
            array_func=lambda x: torch.as_tensor(x, device=device),
        )
    raise ValueError(f"backend {backend} is not supported")


def _create_translate(
    spatial_dims: int, shift: Union[Sequence[float], float], eye_func=np.eye, array_func=np.asarray
) -> NdarrayOrTensor:
    shift = ensure_tuple(shift)
    affine = eye_func(spatial_dims + 1)
    for i, a in enumerate(shift[:spatial_dims]):
        affine[i, spatial_dims] = a
    return array_func(affine)  # type: ignore

def create_control_grid(
    spatial_shape: Sequence[int],
    spacing: Sequence[float],
    homogeneous: bool = True,
    dtype: DtypeLike = float,
    device: Optional[torch.device] = None,
    backend=TransformBackends.NUMPY,
):
    """
    control grid with two additional point in each direction
    """
    torch_backend = look_up_option(backend, TransformBackends) == TransformBackends.TORCH
    ceil_func: Callable = torch.ceil if torch_backend else np.ceil  # type: ignore
    grid_shape = []
    for d, s in zip(spatial_shape, spacing):
        d = torch.as_tensor(d, device=device) if torch_backend else int(d)  # type: ignore
        if d % 2 == 0:
            grid_shape.append(ceil_func((d - 1.0) / (2.0 * s) + 0.5) * 2.0 + 2.0)
        else:
            grid_shape.append(ceil_func((d - 1.0) / (2.0 * s)) * 2.0 + 3.0)
    return create_grid(
        spatial_size=grid_shape, spacing=spacing, homogeneous=homogeneous, dtype=dtype, device=device, backend=backend
    )

def create_grid(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype: Union[DtypeLike, torch.dtype] = float,
    device: Optional[torch.device] = None,
    backend=TransformBackends.NUMPY,
):
    """
    compute a `spatial_size` mesh.

    Args:
        spatial_size: spatial size of the grid.
        spacing: same len as ``spatial_size``, defaults to 1.0 (dense grid).
        homogeneous: whether to make homogeneous coordinates.
        dtype: output grid data type, defaults to `float`.
        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.

    """
    _backend = look_up_option(backend, TransformBackends)
    _dtype = dtype or float
    if _backend == TransformBackends.NUMPY:
        return _create_grid_numpy(spatial_size, spacing, homogeneous, _dtype)
    if _backend == TransformBackends.TORCH:
        return _create_grid_torch(spatial_size, spacing, homogeneous, _dtype, device)
    raise ValueError(f"backend {backend} is not supported")


def _create_grid_numpy(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype: Union[DtypeLike, torch.dtype] = float,
):
    """
    compute a `spatial_size` mesh with the numpy API.
    """
    spacing = spacing or tuple(1.0 for _ in spatial_size)
    ranges = [np.linspace(-(d - 1.0) / 2.0 * s, (d - 1.0) / 2.0 * s, int(d)) for d, s in zip(spatial_size, spacing)]
    coords = np.asarray(np.meshgrid(*ranges, indexing="ij"), dtype=get_equivalent_dtype(dtype, np.ndarray))
    if not homogeneous:
        return coords
    return np.concatenate([coords, np.ones_like(coords[:1])])


def _create_grid_torch(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype=torch.float32,
    device: Optional[torch.device] = None,
):
    """
    compute a `spatial_size` mesh with the torch API.
    """
    spacing = spacing or tuple(1.0 for _ in spatial_size)
    ranges = [
        torch.linspace(
            -(d - 1.0) / 2.0 * s,
            (d - 1.0) / 2.0 * s,
            int(d),
            device=device,
            dtype=get_equivalent_dtype(dtype, torch.Tensor),
        )
        for d, s in zip(spatial_size, spacing)
    ]
    coords = meshgrid_ij(*ranges)
    if not homogeneous:
        return torch.stack(coords)
    return torch.stack([*coords, torch.ones_like(coords[0])])

def create_shear(
    spatial_dims: int,
    coefs: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
    backend=TransformBackends.NUMPY,
) -> NdarrayOrTensor:
    """
    create a shearing matrix

    Args:
        spatial_dims: spatial rank
        coefs: shearing factors, a tuple of 2 floats for 2D, a tuple of 6 floats for 3D),
            take a 3D affine as example::

                [
                    [1.0, coefs[0], coefs[1], 0.0],
                    [coefs[2], 1.0, coefs[3], 0.0],
                    [coefs[4], coefs[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.

    Raises:
        NotImplementedError: When ``spatial_dims`` is not one of [2, 3].

    """
    _backend = look_up_option(backend, TransformBackends)
    if _backend == TransformBackends.NUMPY:
        return _create_shear(spatial_dims=spatial_dims, coefs=coefs, eye_func=np.eye)
    if _backend == TransformBackends.TORCH:
        return _create_shear(
            spatial_dims=spatial_dims, coefs=coefs, eye_func=lambda rank: torch.eye(rank, device=device)
        )
    raise ValueError(f"backend {backend} is not supported")


def _create_shear(spatial_dims: int, coefs: Union[Sequence[float], float], eye_func=np.eye) -> NdarrayOrTensor:
    if spatial_dims == 2:
        coefs = ensure_tuple_size(coefs, dim=2, pad_val=0.0)
        out = eye_func(3)
        out[0, 1], out[1, 0] = coefs[0], coefs[1]
        return out  # type: ignore
    if spatial_dims == 3:
        coefs = ensure_tuple_size(coefs, dim=6, pad_val=0.0)
        out = eye_func(4)
        out[0, 1], out[0, 2] = coefs[0], coefs[1]
        out[1, 0], out[1, 2] = coefs[2], coefs[3]
        out[2, 0], out[2, 1] = coefs[4], coefs[5]
        return out  # type: ignore
    raise NotImplementedError("Currently only spatial_dims in [2, 3] are supported.")

def create_scale(
    spatial_dims: int,
    scaling_factor: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
    backend=TransformBackends.NUMPY,
) -> NdarrayOrTensor:
    """
    create a scaling matrix

    Args:
        spatial_dims: spatial rank
        scaling_factor: scaling factors for every spatial dim, defaults to 1.
        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.
    """
    _backend = look_up_option(backend, TransformBackends)
    if _backend == TransformBackends.NUMPY:
        return _create_scale(spatial_dims=spatial_dims, scaling_factor=scaling_factor, array_func=np.diag)
    if _backend == TransformBackends.TORCH:
        return _create_scale(
            spatial_dims=spatial_dims,
            scaling_factor=scaling_factor,
            array_func=lambda x: torch.diag(torch.as_tensor(x, device=device)),
        )
    raise ValueError(f"backend {backend} is not supported")


def _create_scale(
    spatial_dims: int, scaling_factor: Union[Sequence[float], float], array_func=np.diag
) -> NdarrayOrTensor:
    scaling_factor = ensure_tuple_size(scaling_factor, dim=spatial_dims, pad_val=1.0)
    return array_func(scaling_factor[:spatial_dims] + (1.0,))  # type: ignore

