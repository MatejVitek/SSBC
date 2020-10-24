#!/usr/bin/env python3

# Always import these
import os
import sys
from pathlib import Path
from ast import literal_eval
from matej.collections import DotDict, ensure_iterable
from matej import make_module_callable
from matej.parallel import tqdm_joblib
import argparse
from tkinter import *
from tkinter.colorchooser import askcolor
import tkinter.filedialog as filedialog
from joblib.parallel import Parallel, delayed
from tqdm import tqdm

# If you need EYEZ
ROOT = Path(__file__).absolute().parent.parent
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))
from eyez.utils import EYEZ

# Import whatever else is needed
from eyez.evaluation.segmentation import *
from eyez.evaluation.plot import def_tick_format
import itertools
import matplotlib.cm
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pickle
from PIL import Image
from random import shuffle
from scipy.interpolate import interp1d
import sklearn.metrics as skmetrics


# Constants
ATTR_EXP = 'light', 'phone', ('light', 'phone'), 'gaze'

# Auxiliary stuff
dict_product = lambda d: (dict(zip(d, x)) for x in itertools.product(*d.values()))  # {1: [a, n], 2: [as, fd]} -> [{1: a, 2: as}, {1: a, 2: fd}, {1: b, 2: as}, {1: b, 2: fd}]


class Main:
	def __init__(self, *args, **kw):
		# Default values
		root = EYEZ/'Segmentation/Results/Sclera/2020 SSBC'
		self.models = Path(args[0] if len(args) > 0 else kw.get('root', root/'Models'))
		self.gt = Path(args[1] if len(args) > 1 else kw.get('gt', root/'GT'))
		self.resize = kw.get('resize', (480, 360))
		self.dataset = kw.get('dataset', 'MOBIUS')
		self.k = kw.get('k', 5)
		self.reread = kw.get('reread', True)

		# Extra keyword arguments
		self.extra = DotDict(**kw)

	def __str__(self):
		return str(vars(self))

	def __call__(self):
		if not self.models.is_dir():
			raise ValueError(f"{self.models} is not a directory.")
		if not self.gt.is_dir():
			raise ValueError(f"{self.gt} is not a directory.")

		self.threshold = np.linspace(0, 1, self.extra.get('interp', self.extra.get('interp_points', 1000)))

		if self.dataset.lower() == 'mobius':
			from eyez.data.sets import MOBIUS
			dataset = MOBIUS
		elif self.dataset.lower() == 'sbvpi':
			from eyez.data.sets import SBVPI
			dataset = SBVPI
		else:
			from eyez.data.sets import Dataset
			dataset = Dataset

		dataset = dataset.from_dir(self.gt, mask_dir=None)
		dataset.shuffle()

		with tqdm_joblib(tqdm(desc="Reading GT", total=len(dataset))):
			gt = dict(Parallel(n_jobs=-1)(
				delayed(self._load_gt)(gt_sample)
				for gt_sample in dataset
			))

		for self._model in self.models.iterdir():
			self._predictions = self._model/'Predictions'
			self._binarised = self._model/'Binarised'
			if not self._predictions.is_dir():
				raise ValueError(f"{self._predictions} is not a directory.")
			if not self._binarised.is_dir():
				raise ValueError(f"{self._binarised} is not a directory.")

			# Check if all pickles already exist
			flat_attrs = tuple()
			for attr in ATTR_EXP:
				try:
					flat_attrs += attr
				except TypeError:
					flat_attrs += attr,
			unique_attr_values = {attr: {getattr(sample, attr) for sample in dataset} for attr in set(flat_attrs)}
			exp_attr_values = [{attr: unique_attr_values[attr] for attr in ensure_iterable(attrs, True)} for attrs in ATTR_EXP]
			attr_experiments = {
				', '.join(f'{attr.title()}={val.name.title()}' for attr, val in current_values.items()): current_values
				for current_exp in exp_attr_values
				for current_values in dict_product(current_exp)
			}
			all_names = ['Overall'] + attr_experiments
			if not self.extra.get('overwrite', False) and all((self._model/f'Pickles/{name}.pkl').is_file() for name in all_names):
				print(f"All pickles already exist, skipping {self._model.name}")
				continue

			#TODO: Move folds here and only load one fold's predictions at a time
			# We can't do this because experiment2 needs to have different splits. If we absolutely need this, we'll have to reread the images for each sub-experiment anew.
			# We can cache the images for each split until the end of the split - that way we'll only need to read some of the images anew.

			print(f"Evaluating model {self._model.name}")
			with tqdm_joblib(tqdm(desc="Reading predictions", total=len(dataset))):
				pred_bin = dict(Parallel(n_jobs=-1)(
					delayed(self._process_image)(gt_sample)
					for gt_sample in dataset
				))
			# This will filter out non-existing predictions, so the code will still work, but missing predictions should be addressed (otherwise evaluation is unfair)
			pred_bin_gt = {gt_sample: (*pred_bin[gt_sample], gt[gt_sample]) for gt_sample in dataset if pred_bin[gt_sample] is not None}

			# Overall
			self._experiment1(pred_bin_gt)

			# Split by lighting, phones, and gaze
			for attrs in ATTR_EXP:
				self._experiment2(pred_bin_gt, attrs)

	def _load_gt(self, gt_sample):
		gt = np.array((Image.open(gt_sample.f) if self.resize is None else Image.open(gt_sample.f).resize(self.resize)).convert('1'), dtype=np.bool_).flatten()
		return gt_sample, gt

	def _process_image(self, gt_sample):
		pred_f = self._predictions/gt_sample.f.name
		bin_f = self._binarised/gt_sample.f.name
		if not pred_f.is_file():
			pred_f = pred_f.with_suffix('.jpg')
			if not pred_f.is_file():
				print(f"Missing prediction file {pred_f}.", file=sys.stderr)
				return gt_sample, None
		if not bin_f.is_file():
			bin_f = bin_f.with_suffix('.jpg')
			if not bin_f.is_file():
				print(f"Missing binarised file {bin_f}.", file=sys.stderr)
				return gt_sample, None

		pred = np.array((Image.open(pred_f) if self.resize is None else Image.open(pred_f).resize(self.resize)).convert('L')).flatten() / 255
		bin_ = np.array((Image.open(bin_f) if self.resize is None else Image.open(bin_f).resize(self.resize)).convert('1'), dtype=np.bool_).flatten()
		return gt_sample, (pred, bin_)

	def _experiment1(self, pred_bin_gt):
		print("Experiment 1: Overall performance")
		self._compute(pred_bin_gt.values(), 'Overall')

	def _experiment2(self, pred_bin_gt, attrs):
		attrs = ensure_iterable(attrs, True)
		print(f"Experiment 2: Performance across different {', '.join(attr + 's' for attr in attrs)}")
		values = {attr: {getattr(sample, attr) for sample in pred_bin_gt} for attr in attrs}
		
		for current_values in dict_product(values):
			current_name = ", ".join(attr.title() + "=" + val.name.title() for attr, val in current_values.items())
			data = (pbg for sample, pbg in pred_bin_gt.items() if all(getattr(sample, attr) == val for attr, val in current_values.items()))
			self._compute(data, current_name)

	def _compute(self, data, save):
		save = self._model/f'Pickles/{save}.pkl'
		if not self.extra.get('overwrite', False) and save.is_file():
			return
		save.parent.mkdir(parents=True, exist_ok=True)
		try:
			eval_plt = self._evaluate_and_plot(list(data))
		except ValueError:
			print("Not enough images to split into folds", file=sys.stderr)
			return
		mean_std = Plot.mean_and_std(eval_plt[2], self.threshold)
		print(f"Saving data to {save}")
		with open(save, 'wb') as f:
			pickle.dump(eval_plt, f)
			pickle.dump(mean_std, f)

	def _evaluate_and_plot(self, pred_bin_gt):
		pred_eval = None
		bin_eval = None
		plots = []

		pred_bin_gt = np.array_split(pred_bin_gt, self.k)
		for i in range(self.k):
			print(f"Fold {i+1}")
			preds = np.concatenate([pred for pred, _, _ in pred_bin_gt[i]])
			bins = np.concatenate([bin_ for _, bin_, _ in pred_bin_gt[i]])
			gts = np.concatenate([gt for _, _, gt in pred_bin_gt[i]])
			pred_eval, bin_eval, plot = self._evaluate_and_plot_single_fold(preds, bins, gts, pred_eval, bin_eval)
			plots.append(plot)

		return pred_eval, bin_eval, plots

	def _evaluate_and_plot_single_fold(self, pred, bin_, gt, pred_eval=None, bin_eval=None):
		if pred_eval is None:
			pred_eval = BinarySegmentationEvaluation()
		if bin_eval is None:
			bin_eval = SegmentationEvaluation(F(), Precision(), Recall())

		# Probabilistic prediction
		print("Computing precision/recall curve")
		precisions, recalls, thresholds = skmetrics.precision_recall_curve(gt, pred)
		thresholds = np.append(thresholds, 1.)

		# Hack for edge cases (delete points with the same recall - this also deletes any points with precision=0, recall=0)
		# Get duplicate indices
		idx_sort = np.argsort(recalls)
		sorted_recalls_array = recalls[idx_sort]
		vals, idx_start, count = np.unique(sorted_recalls_array, return_counts=True, return_index=True)
		duplicates = list(filter(lambda x: x.size > 1, np.split(idx_sort, idx_start[1:])))
		if duplicates:
			# We need to delete everything but the one with maximum precision value
			for i, duplicate in enumerate(duplicates):
				duplicates[i] = sorted(duplicate, key=lambda idx: precisions[idx])[:-1]
			to_delete = np.concatenate(duplicates)
			recalls = np.delete(recalls, to_delete)
			precisions = np.delete(precisions, to_delete)
			thresholds = np.delete(thresholds, to_delete)

		print("Updating PR scores")
		# Find threshold with the best F1-score
		f1scores = 2 * precisions * recalls / (precisions + recalls)
		idx = f1scores.argmax()
		pred_eval.f1score.update(f1scores[idx])
		pred_eval.precision.update(precisions[idx])
		pred_eval.recall.update(recalls[idx])

		print("Computing IoU")
		pred_eval.iou.compute_and_update(gt, pred >= thresholds[idx])
		print("Computing AUC")
		pred_eval.auc.compute_and_update(precisions=precisions, recalls=recalls)

		# Binarised prediction
		for metric in bin_eval.values():
			print(f"Computing binarised {metric.name}")
			metric.compute_and_update(gt, bin_)

		plot = Plot(
			recalls,
			precisions,
			(recalls[idx], precisions[idx]),
			(bin_eval.recall.last(), bin_eval.precision.last())
		)

		return pred_eval, bin_eval, plot

	def process_command_line_options(self):
		ap = argparse.ArgumentParser(description="Evaluate segmentation results.")
		ap.add_argument('models', type=Path, nargs='?', default=self.models,
		                help="directory with all model predictions. Should contain a separate folder for each model with 'Predictions' and 'Binarised' inside.")
		ap.add_argument('gt', type=Path, nargs='?', default=self.gt, help="directory with ground truth masks")
		ap.add_argument('-d', '--dataset', type=str.lower, choices=('mobius', 'sbvpi', 'none'), help="dataset file naming protocol used")
		ap.add_argument('-k', type=int, help="number of folds to perform")
		ap.add_argument('-r', '--resize', type=int, nargs=2, help="width and height to resize the images to")
		ap.add_argument('--reread', action='store_true', help="reread predictions each fold (takes longer but uses less memory)")
		ap.parse_known_args(namespace=self)

		ap = argparse.ArgumentParser(description="Extra keyword arguments.")
		ap.add_argument('-e', '--extra', nargs=2, action='append', help="any extra keyword-value argument pairs")
		ap.add_argument('-o', '--overwrite', action='store_true', help="overwrite existing data")
		ap.parse_known_args(namespace=self.extra)

		if self.extra.extra:
			for key, value in self.extra.extra:
				try:
					self.extra[key] = literal_eval(value)
				except ValueError:
					self.extra[key] = value
			del self.extra['extra']

	def gui(self):
		gui = GUI(self)
		gui.mainloop()
		return gui.ok


class Plot:
	def __init__(self, recall, precision, f1_point=None, bin_point=None):
		self.recall = recall
		self.precision = precision
		self.f1_point = f1_point
		self.bin_point = bin_point

	@classmethod
	def mean_and_std(cls, plots, interp=1000):
		try:
			iter(interp)
		except TypeError:
			interp = np.linspace(0, 1, interp)

		# Interpolate precision to linspace recall for mean computation
		precision = np.vstack([
			#interp1d(plot.recall, plot.precision, fill_value='extrapolate')(interp)
			interp1d(plot.recall, plot.precision)(interp)
			#interp1d(plot.recall, plot.precision, fill_value=(1, 0))(interp)
			for plot in plots
		])
		bin_points = np.vstack([plot.bin_point for plot in plots])

		# Compute mean graph and standard deviations
		mean, std = precision.mean(0), precision.std(0)

		# Find max F1 point on mean graph
		f1 = F()
		idx = np.array([f1(precision=p, recall=r) for p, r in zip(mean, interp)]).argmax()

		return (
			Plot(interp, mean, (interp[idx], mean[idx]), bin_points.mean(0)),  # mean
			Plot(interp, mean - std),  # lower std
			Plot(interp, mean + std)   # upper std
		)


class GUI(Tk):
	def __init__(self, argspace, *args, **kw):
		super().__init__(*args, **kw)
		self.args = argspace
		self.ok = False

		self.frame = Frame(self)
		self.frame.pack(fill=BOTH, expand=YES)

		# In grid(), column default is 0, but row default is first empty row.
		row = 0
		self.models_lbl = Label(self.frame, text="Models:")
		self.models_lbl.grid(column=0, row=row, sticky='w')
		self.models_txt = Entry(self.frame, width=60)
		self.models_txt.insert(END, self.args.models)
		self.models_txt.grid(column=1, columnspan=3, row=row)
		self.models_btn = Button(self.frame, text="Browse", command=self.browse_models)
		self.models_btn.grid(column=4, row=row)

		row += 1
		self.gt_lbl = Label(self.frame, text="GT:")
		self.gt_lbl.grid(column=0, row=row, sticky='w')
		self.gt_txt = Entry(self.frame, width=60)
		self.gt_txt.insert(END, self.args.gt)
		self.gt_txt.grid(column=1, columnspan=3, row=row)
		self.gt_btn = Button(self.frame, text="Browse", command=self.browse_gt)
		self.gt_btn.grid(column=4, row=row)

		row += 1
		self.size_lbl = Label(self.frame, text="Size (WxH):")
		self.size_lbl.grid(column=0, row=row, sticky='w')
		self.width_txt = Entry(self.frame, width=10)
		self.width_txt.insert(END, self.args.resize[0])
		self.width_txt.grid(column=1, row=row)
		self.x_lbl = Label(self.frame, text="x")
		self.x_lbl.grid(column=2, row=row)
		self.height_txt = Entry(self.frame, width=10)
		self.height_txt.insert(END, self.args.resize[1])
		self.height_txt.grid(column=3, row=row)

		row += 1
		self.k_lbl = Label(self.frame, text="Folds:")
		self.k_lbl.grid(column=0, row=row, sticky='w')
		self.k_var = IntVar(value=self.args.k)
		self.k_spin = Spinbox(self.frame, from_=1, to=20, textvariable=self.k_var)
		self.k_spin.grid(column=1, row=row)

		row += 1
		self.chk_frame = Frame(self.frame)
		self.chk_frame.grid(row=row, columnspan=3, sticky='w')
		self.overwrite_var = BooleanVar()
		self.overwrite_var.set(False)
		self.overwrite_chk = Checkbutton(self.chk_frame, text="Overwrite", variable = self.overwrite_var)
		self.overwrite_chk.grid(sticky='w')

		row += 1
		self.extra_frame = ExtraFrame(self.frame)
		self.extra_frame.grid(row=row, columnspan=3, sticky='w')

		row += 1
		self.ok_btn = Button(self.frame, text="OK", command=self.confirm)
		self.ok_btn.grid(column=1, row=row)
		self.ok_btn.focus()

	def browse_models(self):
		self._browse_dir(self.models_txt)
		
	def browse_gt(self):
		self._browse_dir(self.gt_txt)

	def _browse_dir(self, target_txt):
		init_dir = target_txt.get()
		while not os.path.isdir(init_dir):
			init_dir = os.path.dirname(init_dir)

		new_entry = filedialog.askdirectory(parent=self, initialdir=init_dir)
		if new_entry:
			_set_entry_text(target_txt, new_entry)

	def _browse_file(self, target_txt, exts=None):
		init_dir = os.path.dirname(target_txt.get())
		while not os.path.isdir(init_dir):
			init_dir = os.path.dirname(init_dir)

		if exts:
			new_entry = filedialog.askopenfilename(parent=self, filetypes=exts, initialdir=init_dir)
		else:
			new_entry = filedialog.askopenfilename(parent=self, initialdir=init_dir)

		if new_entry:
			_set_entry_text(target_txt, new_entry)

	def confirm(self):
		self.args.models = Path(self.models_txt.get())
		self.args.gt = Path(self.gt_txt.get())
		self.args.resize = (int(self.width_txt.get()), int(self.height_txt.get())) if self.width_txt.get() and self.height_txt.get() else None
		self.args.k = self.k_var.get()
		self.args.extra.overwrite = self.overwrite_var.get()

		for kw in self.extra_frame.pairs:
			key, value = kw.key_txt.get(), kw.value_txt.get()
			if key:
				try:
					self.args.extra[key] = literal_eval(value)
				except ValueError:
					self.args.extra[key] = value

		self.ok = True
		self.destroy()


class ExtraFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)
		self.pairs = []

		self.key_lbl = Label(self, width=30, text="Key", anchor='w')
		self.value_lbl = Label(self, width=30, text="Value", anchor='w')

		self.add_btn = Button(self, text="+", command=self.add_pair)
		self.add_btn.grid()

	def add_pair(self):
		pair_frame = KWFrame(self, pady=2)
		self.pairs.append(pair_frame)
		pair_frame.grid(row=len(self.pairs), columnspan=3)
		self.update_labels_and_button()

	def update_labels_and_button(self):
		if self.pairs:
			self.key_lbl.grid(column=0, row=0, sticky='w')
			self.value_lbl.grid(column=1, row=0, sticky='w')
		else:
			self.key_lbl.grid_remove()
			self.value_lbl.grid_remove()
		self.add_btn.grid(row=len(self.pairs) + 1)


class KWFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

		self.key_txt = Entry(self, width=30)
		self.key_txt.grid(column=0, row=0)

		self.value_txt = Entry(self, width=30)
		self.value_txt.grid(column=1, row=0)

		self.remove_btn = Button(self, text="-", command=self.remove)
		self.remove_btn.grid(column=2, row=0)

	def remove(self):
		i = self.master.pairs.index(self)
		del self.master.pairs[i]
		for pair in self.master.pairs[i:]:
			pair.grid(row=pair.grid_info()['row'] - 1)
		self.master.update_labels_and_button()
		self.destroy()


def _set_entry_text(entry, txt):
	entry.delete(0, END)
	entry.insert(END, txt)


if __name__ == '__main__':
	main = Main()

	# If CLI arguments, read them
	if len(sys.argv) > 1:
		main.process_command_line_options()

	# Otherwise get them from a GUI
	else:
		if not main.gui():
			# If GUI was cancelled, exit
			sys.exit(0)

	main()

else:
	# Make module callable (python>=3.5)
	def _main(*args, **kw):
		Main(*args, **kw)()
	make_module_callable(__name__, _main)
