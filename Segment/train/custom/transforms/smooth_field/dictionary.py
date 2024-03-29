from typing import Any, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from .array import (
    RandSmoothDeform,
    RandSmoothFieldAdjustContrast,
    RandSmoothFieldAdjustIntensity,
)
from ..transform import MapTransform, RandomizableTransform
from ..type_definitions import KeysCollection, NdarrayOrTensor
from ..enums import GridSampleMode, GridSamplePadMode, InterpolateMode, TransformBackends
from ..utils import ensure_tuple_rep


__all__ = [
    "RandSmoothFieldAdjustContrastd",
    "RandSmoothFieldAdjustIntensityd",
    "RandSmoothDeformd",
    "RandSmoothFieldAdjustContrastD",
    "RandSmoothFieldAdjustIntensityD",
    "RandSmoothDeformD",
    "RandSmoothFieldAdjustContrastDict",
    "RandSmoothFieldAdjustIntensityDict",
    "RandSmoothDeformDict",
]


InterpolateModeType = Union[InterpolateMode, str]
GridSampleModeType = Union[GridSampleMode, str]


class RandSmoothFieldAdjustContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothFieldAdjustContrast.

    The field is randomized once per invocation by default so the same field is applied to every selected key. The
    `mode` parameter specifying interpolation mode for the field can be a single value or a sequence of values with
    one for each key in `keys`.

    Args:
        keys: key names to apply the augment to
        spatial_size: size of input arrays, all arrays stated in `keys` must have same dimensions
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range for exponential field
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        rand_size: Sequence[int],
        pad: int = 0,
        mode: Union[InterpolateModeType, Sequence[InterpolateModeType]] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.5, 4.5),
        device: Optional[torch.device] = None,
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.mode = ensure_tuple_rep(mode, len(self.keys))

        self.trans = RandSmoothFieldAdjustContrast(
            spatial_size=spatial_size,
            rand_size=rand_size,
            pad=pad,
            mode=self.mode[0],
            align_corners=align_corners,
            prob=1.0,
            gamma=gamma,
            device=device,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustContrastd":
        super().set_random_state(seed, state)
        self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if self._do_transform:
            self.trans.randomize()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        self.randomize()

        if not self._do_transform:
            return data

        d = dict(data)

        for idx, key in enumerate(self.key_iterator(d)):
            self.trans.set_mode(self.mode[idx % len(self.mode)])
            d[key] = self.trans(d[key], False)

        return d


class RandSmoothFieldAdjustIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothFieldAdjustIntensity.

    The field is randomized once per invocation by default so the same field is applied to every selected key. The
    `mode` parameter specifying interpolation mode for the field can be a single value or a sequence of values with
    one for each key in `keys`.

    Args:
        keys: key names to apply the augment to
        spatial_size: size of input arrays, all arrays stated in `keys` must have same dimensions
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range of intensity multipliers
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        rand_size: Sequence[int],
        pad: int = 0,
        mode: Union[InterpolateModeType, Sequence[InterpolateModeType]] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.1, 1.0),
        device: Optional[torch.device] = None,
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.mode = ensure_tuple_rep(mode, len(self.keys))

        self.trans = RandSmoothFieldAdjustIntensity(
            spatial_size=spatial_size,
            rand_size=rand_size,
            pad=pad,
            mode=self.mode[0],
            align_corners=align_corners,
            prob=1.0,
            gamma=gamma,
            device=device,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustIntensityd":
        super().set_random_state(seed, state)
        self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.trans.randomize()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        self.randomize()

        if not self._do_transform:
            return data

        d = dict(data)

        for idx, key in enumerate(self.key_iterator(d)):
            self.trans.set_mode(self.mode[idx % len(self.mode)])
            d[key] = self.trans(d[key], False)

        return d


class RandSmoothDeformd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothDeform.

    The field is randomized once per invocation by default so the same field is applied to every selected key. The
    `field_mode` parameter specifying interpolation mode for the field can be a single value or a sequence of values
    with one for each key in `keys`. Similarly the `grid_mode` parameter can be one value or one per key.

    Args:
        keys: key names to apply the augment to
        spatial_size: input array size to which deformation grid is interpolated
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        field_mode: interpolation mode to use when upsampling the deformation field
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        def_range: value of the deformation range in image size fractions
        grid_dtype: type for the deformation grid calculated from the field
        grid_mode: interpolation mode used for sampling input using deformation grid
        grid_padding_mode: padding mode used for sampling input using deformation grid
        grid_align_corners: if True align the corners when sampling the deformation grid
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        rand_size: Sequence[int],
        pad: int = 0,
        field_mode: Union[InterpolateModeType, Sequence[InterpolateModeType]] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        def_range: Union[Sequence[float], float] = 1.0,
        grid_dtype=torch.float32,
        grid_mode: Union[GridSampleModeType, Sequence[GridSampleModeType]] = GridSampleMode.NEAREST,
        grid_padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        grid_align_corners: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.field_mode = ensure_tuple_rep(field_mode, len(self.keys))
        self.grid_mode = ensure_tuple_rep(grid_mode, len(self.keys))

        self.trans = RandSmoothDeform(
            rand_size=rand_size,
            spatial_size=spatial_size,
            pad=pad,
            field_mode=self.field_mode[0],
            align_corners=align_corners,
            prob=1.0,
            def_range=def_range,
            grid_dtype=grid_dtype,
            grid_mode=self.grid_mode[0],
            grid_padding_mode=grid_padding_mode,
            grid_align_corners=grid_align_corners,
            device=device,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothDeformd":
        super().set_random_state(seed, state)
        self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.trans.randomize()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        self.randomize()

        if not self._do_transform:
            return data

        d = dict(data)

        for idx, key in enumerate(self.key_iterator(d)):
            self.trans.set_field_mode(self.field_mode[idx % len(self.field_mode)])
            self.trans.set_grid_mode(self.grid_mode[idx % len(self.grid_mode)])

            d[key] = self.trans(d[key], False, self.trans.device)

        return d


RandSmoothDeformD = RandSmoothDeformDict = RandSmoothDeformd
RandSmoothFieldAdjustIntensityD = RandSmoothFieldAdjustIntensityDict = RandSmoothFieldAdjustIntensityd
RandSmoothFieldAdjustContrastD = RandSmoothFieldAdjustContrastDict = RandSmoothFieldAdjustContrastd
