from enum import Enum

__all__ = [
    "NumpyPadMode",
    "GridSampleMode",
    "InterpolateMode",
    "PytorchPadMode",
    "GridSamplePadMode",
    "Method",
    "TraceKeys",
    "PostFix",
    "ImageMetaKey",
    "TransformBackends",
]

class NumpyPadMode(Enum):
    """
    See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    CONSTANT = "constant"
    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"
    EMPTY = "empty"


class GridSampleMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    interpolation mode of `torch.nn.functional.grid_sample`

    Note:
        (documentation from `torch.nn.functional.grid_sample`)
        `mode='bicubic'` supports only 4-D input.
        When `mode='bilinear'` and the input is 5-D, the interpolation mode used internally will actually be trilinear.
        However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class InterpolateMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"

class PytorchPadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class GridSamplePadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"

class Method(Enum):
    """
    See also: :py:class:`monai.transforms.croppad.array.SpatialPad`
    """

    SYMMETRIC = "symmetric"
    END = "end"

class TraceKeys:
    """Extra meta data keys used for traceable transforms."""

    CLASS_NAME = "class"
    ID = "id"
    ORIG_SIZE = "orig_size"
    EXTRA_INFO = "extra_info"
    DO_TRANSFORM = "do_transforms"
    KEY_SUFFIX = "_transforms"
    NONE = "none"

from typing import Optional
class PostFix:
    """Post-fixes."""

    @staticmethod
    def _get_str(prefix, suffix):
        return suffix if prefix is None else f"{prefix}_{suffix}"

    @staticmethod
    def meta(key: Optional[str] = None):
        return PostFix._get_str(key, "meta_dict")

    @staticmethod
    def orig_meta(key: Optional[str] = None):
        return PostFix._get_str(key, "orig_meta_dict")

class ImageMetaKey:
    """
    Common key names in the meta data header of images
    """

    FILENAME_OR_OBJ = "filename_or_obj"
    PATCH_INDEX = "patch_index"

class TransformBackends(Enum):
    """
    Transform backends.
    """

    TORCH = "torch"
    NUMPY = "numpy"