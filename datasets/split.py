from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from matej.collections import ensure_iterable, shuffled, sum_
from matej.strings import print_conditional
import numpy as np


class Split(Sequence, ABC):
	"""
	Abstract class representing a `Dataset` split.
	"""

	def __init__(self, dataset, shuffle=True, verbose=True):
		"""
		Construct a `Dataset` split.

		:param Dataset dataset: The dataset to split
		"""

		self.dataset = dataset
		self.type = type(self.dataset)
		self._split = None
		self.new_split(shuffle)
		print_conditional(self, verbose)

	def __init_subclass__(cls, ignore_shuffle=False):
		cls.ignore_shuffle = ignore_shuffle

	def new_split(self, shuffle=True):
		"""
		Perform a new split with the same dataset and parameters.
		"""

		if shuffle and not self.ignore_shuffle:
			self.dataset = shuffled(self.dataset)
		self._split = self._new_split()

	@abstractmethod
	def _new_split(self):
		"""
		Override this in subclasses to perform and return the split.

		:return Iterable split: Iterable containing a Dataset for each fold of the split
		"""

	def __str__(self):
		return f"Splitting dataset into {len(self)} folds with sizes ({[len(f) for f in self]})."

	def __getitem__(self, index):
		try:
			# Return a single fold
			return self._split[index]
		except TypeError:
			# Return multiple folds as a single Dataset
			return sum_(self._split[i] for i in index)

	def __len__(self):
		return len(self._split)

	def __iter__(self):
		return iter(self._split)


class FoldSplit(Split):
	"""
	Split into folds of the same size (±1 if it cannot be split evenly).
	"""

	def __init__(self, dataset, n_folds=2, *args, **kw):
		"""
		:param int n_folds: Number of folds

		For other parameters, see `Split.__init__`
		"""

		self.n_folds = n_folds
		super().__init__(dataset, *args, **kw)

	def _new_split(self):
		return [self.type(fold) for fold in np.array_split(self.dataset, self.n_folds)]

	def __str__(self):
		return f"Splitting dataset into {self.n_folds} folds of size ~{len(self[0])}."


class RatioSplit(Split):
	"""
	Split into 2 or more folds at specific points.
	"""

	def __init__(self, dataset, ratio=.5, *args, **kw):
		"""
		:param ratio: The point(s) at which to split (should be on the interval [0, 1]) the dataset.
		:type  ratio: float or Iterable

		For other parameters, see `Split.__init__`
		"""

		self.split_points = [round(r * len(dataset)) for r in ensure_iterable(ratio)]
		super().__init__(dataset, *args, **kw)

	def _new_split(self):
		return [self.type(fold) for fold in np.split(self.dataset, self.split_points)]


_default = object()
class AttributeSplit(Split, ignore_shuffle=True):
	"""
	Split by an attribute of the Samples.
	"""

	def __init__(self, dataset, by, bins=_default, ordered_bins=False, *args, **kw):
		"""
		:param by: Key to group by. Can be a string (denoting the attribute name) or a function mapping a sample to a corresponding key value.
				   The key values should be hashable.
		:type by:  str or Callable
		:param bins: Value bins to group the samples into. Can be a collection of values [x1,...,xn] or a single value.
					 This will split the dataset into n+1 bins (x=x1, x=x2, ..., x=xn, x∉{x1,...,xn}).
					 By default the bins will be all values of `by` over the dataset.
		:param bool ordered_bins: If `True`, the dataset will instead be split into bins (x<=x1, x1<x<=x2, ..., x>xn).

		For other parameters, see `Split.__init__`
		"""

		self.f = partial(getattr, name=by) if isinstance(by, str) else by
		self.bins = bins
		self._ordered = ordered_bins
		super().__init__(dataset, *args, **kw)

	def _new_split(self):
		if self.bins is _default:
			self.bins = {self.f(s) for s in self.dataset}

		self.bins = ensure_iterable(self.bins, True)

		if self._ordered:
			self.bins = sorted(self.bins)
			return [self.type([
				s for s in self.dataset
				if (i == 0 or self.f(s) > self.bins[i-1])
				and (i == len(self.bins) or self.f(s) <= self.bins[i])
			]) for i in range(len(self.bins))]
		else:
			# We use _default for the last bin (x∉{x1,...,xn})
			split = {bin: [] for bin in self.bins + [_default]}
			for s in self.dataset:
				v = self.f(s)
				if v in split:
					split[v].append(s)
				else:
					split[_default].append(s)
			return [self.type(bin) for bin in split.values()]


class GPSplit(Split):
	def __init__(self, *args, **kw):
		self.gallery = None
		self.probe = None
		super().__init__(*args, **kw)

	def new_split(self, *args, **kw):
		super().new_split(*args, **kw)
		self.gallery, self.probe = self._split

	def __str__(self):
		return f"Placing {len(self.gallery)} images in gallery and {len(self.probe)} in probe."


class GPRatioSplit(RatioSplit, GPSplit):
	def _new_split(self):
		return self.type(self.dataset[:self.split_points[0]]), self.type(self.dataset[self.split_points[0]:])


class GazeSplit(GPSplit):
	def __init__(self, dataset, directions=4, *args, **kw):
		"""
		:param directions: Directions to include in gallery. If integer (1<=x<=4), x directions will be randomly sampled for each subject.
		                   If collection of directions, one sample for each of the directions will be used.
		:type directions:  int or Iterable[Direction]

		For other parameters, see `Split.__init__`
		"""
		self.dirs = directions
		super().__init__(dataset)

	def _new_split(self):
		g = []
		p = []

		for s in self.dataset:
			# int directions
			try:
				if sum(s.label == g.label for g in g) >= self.dirs:
					p.append(s)
					continue
			# collection of directions
			except TypeError:
				if s.gaze not in self.dirs:
					p.append(s)
					continue
			if any(s.label == g.label and s.direction == g.direction for g in g):
				p.append(s)
			else:
				g.append(s)

		self.gallery = self._dataset(g)
		self.probe = self._dataset(p)
		return self.gallery, self.probe
