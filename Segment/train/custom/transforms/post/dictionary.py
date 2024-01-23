import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union

import torch

from ..type_definitions import NdarrayOrTensor, KeysCollection
from ..enums import PostFix
from ..utils import ensure_tuple_rep
from .array import (
    Activations, AsDiscrete, FillHoles,
    KeepLargestConnectedComponent,
    LabelFilter, LabelToContour, MeanEnsemble,
    ProbNMS, VoteEnsemble,
)
from ..transform import MapTransform


__all__ = [
    "ActivationsD",
    "ActivationsDict",
    "Activationsd",
    "AsDiscreteD",
    "AsDiscreteDict",
    "AsDiscreted",
    "Ensembled",
    "EnsembleD",
    "EnsembleDict",
    "FillHolesD",
    "FillHolesDict",
    "FillHolesd",
    "KeepLargestConnectedComponentD",
    "KeepLargestConnectedComponentDict",
    "KeepLargestConnectedComponentd",
    "LabelFilterD",
    "LabelFilterDict",
    "LabelFilterd",
    "LabelToContourD",
    "LabelToContourDict",
    "LabelToContourd",
    "MeanEnsembleD",
    "MeanEnsembleDict",
    "MeanEnsembled",
    "ProbNMSD",
    "ProbNMSDict",
    "ProbNMSd",
    "VoteEnsembleD",
    "VoteEnsembleDict",
    "VoteEnsembled",
]

DEFAULT_POST_FIX = PostFix.meta()

class Activationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.
    """

    backend = Activations.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigmoid: Union[Sequence[bool], bool] = False,
        softmax: Union[Sequence[bool], bool] = False,
        other: Optional[Union[Sequence[Callable], Callable]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
            sigmoid: whether to execute sigmoid function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            softmax: whether to execute softmax function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            other: callable function to execute other activation layers,
                for example: `other = torch.tanh`. it also can be a sequence of Callable, each
                element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, sigmoid, softmax, other in self.key_iterator(d, self.sigmoid, self.softmax, self.other):
            d[key] = self.converter(d[key], sigmoid, softmax, other)
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`transforms.AsDiscrete`.
    """

    backend = AsDiscrete.backend

    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[Optional[int]], Optional[int]] = None,
        threshold: Union[Sequence[Optional[float]], Optional[float]] = None,
        rounding: Union[Sequence[Optional[str]], Optional[str]] = None,
        allow_missing_keys: bool = False,
        num_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        logit_thresh: Union[Sequence[float], float] = 0.5,  # deprecated
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        to_onehot_ = ensure_tuple_rep(to_onehot, len(self.keys))
        num_classes = ensure_tuple_rep(num_classes, len(self.keys))
        self.to_onehot = []
        for flag, val in zip(to_onehot_, num_classes):
            if isinstance(flag, bool):
                warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
                self.to_onehot.append(val if flag else None)
            else:
                self.to_onehot.append(flag)

        threshold_ = ensure_tuple_rep(threshold, len(self.keys))
        logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.threshold = []
        for flag, val in zip(threshold_, logit_thresh):
            if isinstance(flag, bool):
                warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
                self.threshold.append(val if flag else None)
            else:
                self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, argmax, to_onehot, threshold, rounding in self.key_iterator(
            d, self.argmax, self.to_onehot, self.threshold, self.rounding
        ):
            d[key] = self.converter(d[key], argmax, to_onehot, threshold, rounding)
        return d


class KeepLargestConnectedComponentd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`transforms.KeepLargestConnectedComponent`.
    """

    backend = KeepLargestConnectedComponent.backend

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels: Union[Sequence[int], int],
        is_onehot: Optional[bool] = None,
        independent: bool = True,
        connectivity: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            applied_labels: Labels for applying the connected component analysis on.
                If not OneHot. The pixel whose value is in this list will be analyzed.
                If the data is in OneHot format, this is used to determine which channels to apply.
            is_onehot: if `True`, treat the input data as OneHot format data, otherwise, not OneHot format data.
                default to None, which treats multi-channel data as OneHot and single channel data as not OneHot.
            independent: whether to treat ``applied_labels`` as a union of foreground labels.
                If ``True``, the connected component analysis will be performed on each foreground label independently
                and return the intersection of the largest components.
                If ``False``, the analysis will be performed on the union of foreground labels.
                default is `True`.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used. for more details:
                https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = KeepLargestConnectedComponent(
            applied_labels=applied_labels, is_onehot=is_onehot, independent=independent, connectivity=connectivity
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class LabelFilterd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`transforms.LabelFilter`.
    """

    backend = LabelFilter.backend

    def __init__(
        self, keys: KeysCollection, applied_labels: Union[Sequence[int], int], allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels: Label(s) to filter on.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelFilter(applied_labels)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class FillHolesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`transforms.FillHoles`.
    """

    backend = FillHoles.backend

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels: Optional[Union[Iterable[int], int]] = None,
        connectivity: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Initialize the connectivity and limit the labels for which holes are filled.

        Args:
            keys: keys of the corresponding items to be transformed.
            applied_labels (Optional[Union[Iterable[int], int]], optional): Labels for which to fill holes. Defaults to None,
                that is filling holes for all labels.
            connectivity (int, optional): Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. Defaults to a full
                connectivity of ``input.ndim``.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = FillHoles(applied_labels=applied_labels, connectivity=connectivity)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class LabelToContourd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`transforms.LabelToContour`.
    """

    backend = LabelToContour.backend

    def __init__(self, keys: KeysCollection, kernel_type: str = "Laplace", allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            kernel_type: the method applied to do edge detection, default is "Laplace".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelToContour(kernel_type=kernel_type)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class Ensembled(MapTransform):
    """
    Base class of dictionary-based ensemble transforms.

    """

    backend = list(set(VoteEnsemble.backend) & set(MeanEnsemble.backend))

    def __init__(
        self,
        keys: KeysCollection,
        ensemble: Callable[[Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]], NdarrayOrTensor],
        output_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
            ensemble: callable method to execute ensemble on specified data.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            TypeError: When ``ensemble`` is not ``callable``.
            ValueError: When ``len(keys) > 1`` and ``output_key=None``. Incompatible values.

        """
        super().__init__(keys, allow_missing_keys)
        if not callable(ensemble):
            raise TypeError(f"ensemble must be callable but is {type(ensemble).__name__}.")
        self.ensemble = ensemble
        if len(self.keys) > 1 and output_key is None:
            raise ValueError("Incompatible values: len(self.keys) > 1 and output_key=None.")
        self.output_key = output_key if output_key is not None else self.keys[0]

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        items: Union[List[NdarrayOrTensor], NdarrayOrTensor]
        if len(self.keys) == 1 and self.keys[0] in d:
            items = d[self.keys[0]]
        else:
            items = [d[key] for key in self.key_iterator(d)]

        if len(items) > 0:
            d[self.output_key] = self.ensemble(items)

        return d


class MeanEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`transforms.MeanEnsemble`.
    """

    backend = MeanEnsemble.backend

    def __init__(
        self,
        keys: KeysCollection,
        output_key: Optional[str] = None,
        weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            weights: can be a list or tuple of numbers for input data with shape: [E, C, H, W[, D]].
                or a Numpy ndarray or a PyTorch Tensor data.
                the `weights` will be added to input data from highest dimension, for example:
                1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
                2. if the `weights` has 2 dimensions, it will be added to `E` and `C` dimensions.
                it's a typical practice to add weights for different classes:
                to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
                so the input data shape can be: [3, 4, H, W, D].
                and add different `weights` for different classes, so the `weights` shape can be: [3, 4].
                for example: `weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]`.

        """
        ensemble = MeanEnsemble(weights=weights)
        super().__init__(keys, ensemble, output_key)


class VoteEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.VoteEnsemble`.
    """

    backend = VoteEnsemble.backend

    def __init__(
        self, keys: KeysCollection, output_key: Optional[str] = None, num_classes: Optional[int] = None
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            num_classes: if the input is single channel data instead of One-Hot, we can't get class number
                from channel, need to explicitly specify the number of classes to vote.

        """
        ensemble = VoteEnsemble(num_classes=num_classes)
        super().__init__(keys, ensemble, output_key)


class ProbNMSd(MapTransform):
    """
    Performs probability based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `box_size`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
            Defaults to 2.
        sigma: the standard deviation for gaussian filter.
            It could be a single value, or `spatial_dims` number of values. Defaults to 0.0.
        prob_threshold: the probability threshold, the function will stop searching if
            the highest probability is no larger than the threshold. The value should be
            no less than 0.0. Defaults to 0.5.
        box_size: the box size (in pixel) to be removed around the the pixel with the maximum probability.
            It can be an integer that defines the size of a square or cube,
            or a list containing different values for each dimensions. Defaults to 48.

    Return:
        a list of selected lists, where inner lists contain probability and coordinates.
        For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

    Raises:
        ValueError: When ``prob_threshold`` is less than 0.0.
        ValueError: When ``box_size`` is a list or tuple, and its length is not equal to `spatial_dims`.
        ValueError: When ``box_size`` has a less than 1 value.

    """

    backend = ProbNMS.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 2,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.0,
        prob_threshold: float = 0.5,
        box_size: Union[int, Sequence[int]] = 48,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob_nms = ProbNMS(
            spatial_dims=spatial_dims, sigma=sigma, prob_threshold=prob_threshold, box_size=box_size
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.prob_nms(d[key])
        return d

ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
FillHolesD = FillHolesDict = FillHolesd
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
LabelFilterD = LabelFilterDict = LabelFilterd
LabelToContourD = LabelToContourDict = LabelToContourd
MeanEnsembleD = MeanEnsembleDict = MeanEnsembled
ProbNMSD = ProbNMSDict = ProbNMSd
VoteEnsembleD = VoteEnsembleDict = VoteEnsembled
EnsembleD = EnsembleDict = Ensembled