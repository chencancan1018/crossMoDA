import os
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union

import numpy as np
import torch

__all__ = [
    "KeysCollection",
    "IndexSelection",
    "DtypeLike",
    "NdarrayTensor",
    "NdarrayOrTensor",
    "PathLike",
]


#: KeysCollection
#
# The KeyCollection type is used to for defining variables
# that store a subset of keys to select items from a dictionary.
# The container of keys must contain hashable elements.
# NOTE:  `Hashable` is not a collection, but is provided as a
#        convenience to end-users.  All supplied values will be
#        internally converted to a tuple of `Hashable`'s before
#        use
KeysCollection = Union[Collection[Hashable], Hashable]

#: IndexSelection
#
# The IndexSelection type is used to for defining variables
# that store a subset of indices to select items from a List or Array like objects.
# The indices must be integers, and if a container of indices is specified, the
# container must be iterable.
IndexSelection = Union[Iterable[int], int]

#: Type of datatypes: Adapted from https://github.com/numpy/numpy/blob/v1.21.4/numpy/typing/_dtype_like.py#L121
DtypeLike = Union[np.dtype, type, str, None]

#: NdarrayOrTensor: Union of numpy.ndarray and torch.Tensor to be used for typing
NdarrayOrTensor = Union[np.ndarray, torch.Tensor]

#: NdarrayTensor
#
# Generic type which can represent either a numpy.ndarray or a torch.Tensor
# Unlike Union can create a dependence between parameter(s) / return(s)
NdarrayTensor = TypeVar("NdarrayTensor", bound=NdarrayOrTensor)

#: PathLike: The PathLike type is used for defining a file path.
PathLike = Union[str, os.PathLike]