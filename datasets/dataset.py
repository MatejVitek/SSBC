from abc import ABC
from collections import defaultdict
from enum import Enum
from matej.collections import shuffled
import numpy as np
import os
from pathlib import Path
import random
import re
from tqdm import tqdm


IMG_EXTS = '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.ppm'


class Direction(Enum):
	LEFT = L = 'l'
	RIGHT = R = 'r'
	STRAIGHT = S = CENTRE = C = 's'
	UP = U = 'u'


class _AbstractSample(ABC):
	@classmethod
	def __init_subclass__(cls, regex=r'(?P<id>\d+)', mask_regex=None, valid_extensions=IMG_EXTS, *args, **kw):
		super().__init_subclass__(*args, **kw)

		cls.match = re.compile(regex).fullmatch

		if mask_regex is None:
			mask_regex = r'(?P<basename>.+)_(?P<mask>[a-zA-Z]+)'
		cls.mask_match = re.compile(mask_regex).fullmatch

		cls.extensions = valid_extensions

class Sample(_AbstractSample):
	def __init__(self, f, masks=None):
		self.f = Path(f)
		self.bname = self.f.stem
		self.masks = masks

		self.__dict__.update(self.match(self.bname).groupdict())
		self._postprocess_attributes()

		self.label = self.id

	def _postprocess_attributes(self):
		self.id = int(self.id)

	def __eq__(self, other):
		if isinstance(other, Sample):
			return self.label == other.label and self.f == other.f
		return NotImplemented

	def __hash__(self):
		return hash((self.label, self.f))

	def __str__(self):
		return f"Sample {self.bname} ({self.f})"

	# Make Samples sortable by id, then bname
	def __lt__(self, other):
		return (self.id, self.bname) < (other.id, other.bname)

	@classmethod
	def valid(cls, f, mask=False):
		match = cls.mask_match if mask else cls.match
		bname, ext = Path(f).stem, Path(f).suffix
		return match(bname) and any(ext.lower() in (x.lower(), '.' + x.lower()) for x in cls.extensions)


class Dataset:
	sample_cls = Sample

	def __init__(self, data=None):
		"""
		Constructs a Dataset from a collection of Samples.

		:param Iterable[Sample] data: Collection of Samples to build the dataset from
		"""

		self.data = list(data) if data is not None else []
		for s in self:
			if not isinstance(s, self.sample_cls):
				raise TypeError(f"Bad data type: {type(s)}")

	def classes(self):
		return {s.label for s in self.data}

	@classmethod
	def from_dir(cls, dir, mask_dir=None, *args, both_eyes_same_class=True, mirrored_offset=0, **kw):
		"""
		Dataset initialiser that reads samples from a directory.

		:param str dir: Root directory for images
		:param str mask_dir: Root directory for masks.
		                     If it's a relative path (this includes `''`, which is equivalent to `'.'`), will resolve it from `dir`.
							 If `True`, will try to guess (`dir`/../Masks, `dir`/Masks, `dir`/../Annotations, etc.).
		                     If `None` or `False`, masks won't be read.

		For `both_eyes_same_class` and `mirrored_offset` see `Dataset.assign_class_labels`.
		For other parameters see `Dataset.__init__`.
		"""

		dir = Path(dir)

		if mask_dir is True:
			mask_dir = cls._guess_mask_dir(dir)
		elif mask_dir is not False and mask_dir is not None:
			mask_dir = Path(mask_dir)
			if not mask_dir.is_absolute():
				mask_dir = dir/mask_dir

		if mask_dir:
			data = cls._read_masked_samples(dir, mask_dir)
		else:
			data = cls._read_samples(dir)

		dataset = cls(data, *args, **kw)
		dataset.assign_class_labels(both_eyes_same_class, mirrored_offset)
		dataset.sort()
		print(f"Found {len(dataset.data)} images belonging to {len(dataset.classes())} classes.")

		return dataset

	@classmethod
	def _guess_mask_dir(cls, dir):
		for guess in ('Masks', 'masks', 'Annotations', 'annotations', 'GT', 'gt', 'GroundTruth', 'groundtruth', 'ground_truth', 'Ground Truth', 'Ground truth', 'ground truth', 'Labels', 'labels'):
			for mask_dir in dir.parent/guess, dir/guess:
				if mask_dir.is_dir():
					return mask_dir
		raise ValueError(f"No mask dir found in {dir.parent}")

	@classmethod
	def _read_masked_samples(cls, dir, mask_dir):
		if not dir.is_dir():
			raise ValueError(f"{dir} is not a directory.")
		if not mask_dir.is_dir():
			raise ValueError(f"{mask_dir} is not a directory.")

		# Find valid image files
		img_fs = {f.stem: f for f in cls._find_valid_files(dir, "Finding samples")}

		# Find valid mask files and save their matches
		valid_mask_matches = {f: cls.sample_cls.mask_match(f.stem) for f in cls._find_valid_files(mask_dir, "Finding masks", True)}

		# Invert the dictionary to a dict of dicts, mapping: basename -> mask channel -> full file name
		mask_fs = {}
		for f, match in valid_mask_matches.items():
			mask_fs.setdefault(match['basename'], {})[match['mask']] = f

		# If no masks found for a certain basename, pass an empty dict
		return [cls.sample_cls(img_fs[bname], mask_fs.get(bname, {})) for bname in img_fs]

	@classmethod
	def _read_samples(cls, dir):
		if not dir.is_dir():
			raise ValueError(f"{dir} is not a directory.")
		return [cls.sample_cls(f) for f in cls._find_valid_files(dir, "Finding samples")]

	@classmethod
	def _find_valid_files(cls, dir, tqdm_text, mask=False):
		return [
			Path(root)/fname
			for root, _, fnames in tqdm(os.walk(dir), desc=tqdm_text)
			for fname in fnames
			if cls.sample_cls.valid(fname, mask)
		]

	def assign_class_labels(self, both_eyes_same_class=True, mirrored_offset=0):
		"""
		Assign class labels to the samples in the dataset.

		:param bool both_eyes_same_class: Whether both eyes should be counted as the same class
		:param int mirrored_offset: Offset for mirrored identities. If 0, mirrored eyes will be counted as distinct classes.
		"""

		flipped = [mirrored_offset and s.id > mirrored_offset for s in self.data]
		adjusted_ids = [(s.id - mirrored_offset) if f else s.id for (s, f) in zip(self.data, flipped)]
		sorted_ids = sorted(set(adjusted_ids))

		for (s, f, id) in zip(self.data, flipped, adjusted_ids):
			if both_eyes_same_class:
				s.label = sorted_ids.index(id)
			# If id is mirrored (and we're counting mirrored images as same class), we need to reverse L/R for mirrored images.
			else:
				s.label = 2 * sorted_ids.index(id) + ((1 - s.eye) if f else s.eye)

	def split(self, ratio=(.6, .2, .2), shuffle=True, subject_disjoint='none'):
		"""
		Split dataset into train, validation, and test dataset.

		:param Iterable[float] ratio: The relative size of the train, validation, and test dataset. Can be 0 if any of them are not desired.
		:param bool shuffle: Whether to shuffle the dataset before splitting.
		:param str subject_disjoint: Which dataset(s) should be subject disjoint with train. Can be `'none'`, `'val'`, `'test'`, or `'all'`.
		                             Note that with subject disjoint sets the distribution ratio won't be exact.

		:return dict[str,Dataset] split: mapping from `'train'`, `'val'`, and `'test'` to respective `Dataset`s.
		"""

		subject_disjoint = subject_disjoint.lower()
		data = self.data
		if shuffle:
			data = shuffled(self.data)
		ratio = (np.array(ratio) / sum(ratio) * len(data)).astype(int)

		split = defaultdict(list)
		if subject_disjoint == 'none':
			ratio[1] += ratio[0]
			split['train'] = data[:ratio[0]]
			split['val'] = data[ratio[0]:ratio[1]]
			split['test'] = data[ratio[1]:]
		else:
			subjects = {s.id for s in data}
			if subject_disjoint in {'val', 'all'}:
				while len(split['val']) < ratio[1]:
					self._add_random_subject(split['val'], data, subjects)
			if subject_disjoint in {'test', 'all'}:
				while len(split['test']) < ratio[2]:
					self._add_random_subject(split['test'], data, subjects)
			for s in data:
				if s.id in subjects:
					subset = 'train' if subject_disjoint == 'all' or len(split['train']) < ratio[0] else 'val' if subject_disjoint == 'test' else 'test'
					split[subset].append(s)

		return {name: Dataset(data) for name, data in split.items() if data}

	@classmethod
	def _add_random_subject(cls, l, data, subjects):
		cls._add_subject(l, random.choice(tuple(subjects)), data, subjects)

	@staticmethod
	def _add_subject(l, id, data, subjects):
		subjects.remove(id)
		l.extend(s for s in data if s.id == id)

	def shuffle(self):
		random.shuffle(self.data)
		return self

	def sort(self):
		self.data.sort()
		return self

	def __getitem__(self, item):
		return self.data[item]

	def __iter__(self):
		return iter(self.data)

	def __len__(self):
		return len(self.data)

	def __add__(self, other):
		# This should make addition resolve to the lowest common superclass in subclasses.
		# This way you can for instance add two different Dataset subclasses and get a Dataset as a result.
		# Such addition may have unintended consequences, particularly if the subclasses add new class attributes.
		if isinstance(other, Dataset):
			mro_self, mro_other = type(self).mro(), type(other).mro()
			for c in mro_self:
				if c in mro_other:
					return c(list(set(self.data + other.data)))
		return NotImplemented

	def __eq__(self, other):
		if isinstance(other, Dataset):
			return set(self.data) == set(other.data)
		return NotImplemented

	def __str__(self):
		return f"Dataset{list(str(s) for s in self.data)}"
