from enum import Enum

from . import Dataset, Sample, Direction as Dir
from eyez.utils import EYEZ


class Phone(Enum):
	XPERIA = 1
	IPHONE = 2
	XIAOMI = 3


class Light(Enum):
	NATURAL = 'n'
	INDOOR = 'i'
	POOR = 'p'


class _ExtraInfo:
	def __init__(self, *args):
		self.gender = args[0]  # 'm' or 'f'
		self.age = int(args[1])  # int
		self.colour = args[2]  # str
		self.glasses_lenses = None if args[3].lower() == 'no' else args[3]  # 'g' = glasses, 'l' = lenses, None = neither
		self.dpt = float(args[4]), float(args[5])  # (left, right) float tuple
		self.cyl = float(args[6]), float(args[7])  # (left, right) float tuple
		self.smoker = args[8].lower() == 'y'  # bool
		self.eye_conditions = [] if not args[9] else args[9].split('/')  # list of str
		self.drops = args[10]  # str
		self.allergies = [] if not args[11] else args[11].split('/')  # list of str


class MOBIUSSample(Sample, regex=r'(?P<id>\d+)_(?P<phone>\d+)(?P<light>[nip])_(?P<eye>[LR])(?P<gaze>[lrsu])_(?P<n>(\d+)|(bad))'):
	def _postprocess_attributes(self):
		super()._postprocess_attributes()
		self.phone = Phone(int(self.phone))
		self.light = Light(self.light)
		self.eye = Dir(self.eye.lower())
		self.gaze = Dir(self.gaze)
		try:
			self.n = int(self.n)
		except ValueError:
			# '_bad' image
			pass


class MOBIUS(Dataset):
	sample_cls = MOBIUSSample

	@classmethod
	def from_dir(cls, dir, mask_dir=True, *args, extra_info_path=EYEZ/'MOBIUS/data.csv', **kw):
		dataset = super().from_dir(dir, mask_dir, *args, **kw)
		try:
			extra_info = cls.read_extra_info(extra_info_path)
			for s in dataset:
				s.__dict__.update(extra_info[s.id].__dict__)
		except IOError:
			import sys
			print("Warning: Extra info file not found. Only information parsed from file names will be available.", file=sys.stderr)
		return dataset

	@classmethod
	def read_extra_info(cls, extra_info_path=EYEZ/'MOBIUS/data.csv'):
		with open(extra_info_path, 'r') as f:
			# Skip header
			f.readline()
			return {
				# Split line and skip ID
				(id + 1): _ExtraInfo(*line.strip().split(',')[1:])
				for (id, line) in enumerate(f.readlines())
			}
