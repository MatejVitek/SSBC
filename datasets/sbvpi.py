from . import Dataset, Sample, Direction as Dir


class _ExtraInfo:
	def __init__(self, gender, age, colour):
		self.gender = gender
		self.age = int(age)
		self.colour = colour


class SBVPISample(Sample, regex=r'(?P<id>\d+)(?P<eye>[LR])_(?P<gaze>[lrsu])_(?P<n>\d+)'):
	def _postprocess_attributes(self):
		super()._postprocess_attributes()
		self.eye = Dir(self.eye.lower())
		self.gaze = Dir(self.gaze)
		self.n = int(self.n)


class SBVPI(Dataset):
	sample_cls = SBVPISample

	@classmethod
	def from_dir(cls, dir, mask_dir='', *args, extra_info_path='/path/to/SBVPI/SBVPI_Gender_Age_Colour.txt', **kw):
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
	def read_extra_info(cls, extra_info_path='/path/to/SBVPI/SBVPI_Gender_Age_Colour.txt'):
		with open(extra_info_path, 'r') as f:
			return {
				(id + 1 + mirror_offset): _ExtraInfo(*line.split())
				for (id, line) in enumerate(f.readlines())
				for mirror_offset in (0, 55)
			}
