"""Samplers to sample a subset of objects from sequence of objects."""
import abc
from typing import Any, Iterable

import numpy as np
from monty.json import MSONable

from potdata.utils.dataops import serializable_slice, slice_sequence

__all__ = ["RandomSampler", "SliceSampler"]


class BaseSampler(MSONable):
    """Base class for data sampling."""

    @abc.abstractmethod
    def sample(
        self, data: Iterable, return_indices: bool = False
    ) -> list[Any] | list[int]:
        """Run the sampler to sample a subset of the data.

        Args:
            data: data to sample from.
            return_indices: If `True`, the returned value will be a list of indices
                of the original data. If `False`, the returned value will be a list of
                subset of the data.

        Returns:
            A list of data or indices.
        """

    @property
    @abc.abstractmethod
    def indices(self) -> list[int]:
        """
        Return the indices of the data that has been sampled.
        """


class RandomSampler(BaseSampler):
    """Randomly sample a subset.

    Args:
        size: number of data points to sample
        seed: random seed for the sampler.
    """

    def __init__(self, size: int, seed: int = 35):
        self.size = size
        self.seed = seed
        self._indices: list[int] = None
        np.random.seed(self.seed)

    def sample(
        self, data: Iterable, return_indices: bool = False
    ) -> list[Any] | list[int]:
        data = [x for x in data]

        if self.size > len(data):
            raise ValueError(
                f"Requested number of samples `{self.size}` larger than total "
                f"number f data points `{len(data)}`."
            )

        self._indices = list(
            np.random.randint(low=0, high=len(data), size=self.size).tolist()  # type: ignore
        )

        if return_indices:
            return self._indices  # ty
        else:
            return [data[i] for i in self._indices]

    @property
    def indices(self) -> list[int]:
        return self._indices


class SliceSampler(BaseSampler):
    """Sample a slice of the data points.

    Args:
        slicer: indices of the data to select or a slice object. If a list of int is
            provided, the corresponding indices are selected (e.g. `slicer=[0, 3, 5]`
            will select data points with indices 0, 3, and 5). Alternatively, a python
            slice object can be provided (e.g. `slicer=slice(0, None, 2)`
            will select data points with indices 0, 2, 4,...).
    """

    def __init__(self, slicer: list[int] | serializable_slice):
        self.slicer = slicer
        self._indices: list[int] = None

    def sample(
        self, data: Iterable, return_indices: bool = False
    ) -> list[Any] | list[int]:
        selected, self._indices = slice_sequence(data, self.slicer)

        if return_indices:
            return self._indices
        else:
            return selected

    @property
    def indices(self) -> list[int]:
        return self._indices
