#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.abspath(__file__))
from pathlib import Path
from ast import literal_eval
from matej import make_module_callable
from matej.collections import DotDict, ensure_iterable
import argparse
from tkinter import *
import tkinter.filedialog as filedialog
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import math
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import pickle
import re
from segmentation_evaluation_compute import ATTR_EXP, Plot  # Plot is needed for pickle loading
from evaluation.segmentation import *
from evaluation import def_tick_format


# Constants
FIG_EXTS = 'png', 'pdf', 'eps'
CMAP = plt.cm.plasma
ZOOMED_LIMS = .6, .95
model_complexity = {
	'ssip': (1, 6.5e6),
	'ab sclera net': (32.7e6, 265e9),
	'sassnet': (59e6, 17.9e9),
	'sclerau-net': (2.16e6, 4.32e6),
	'multi-deeplab': (41e6, 117e9),
	'multi-fcn': (134e6, 112e9),
	'rgb-ss-eye-ms': (22.7e6, None),
	'y-ss-eye-ms': (22.6e6, None),
	'color ritnet': (250e3, None),
	's-net': (1.18e6, None),
	'unet-p': (1.94e6, 3.88e6),
	'fcn8': (138e6, 15e9),
	'mask2020cl': (64e6, None),
	'mu-net': (409e3, 180e9)
}

# Auxiliary stuff
dict_product = lambda d: (dict(zip(d, x)) for x in itertools.product(*d.values()))
FIG_EXTS = ensure_iterable(FIG_EXTS, True)


class Main:
	def __init__(self, *args, **kw):
		# Default values
		root = EYEZ/'Segmentation/Results/Sclera/2020 SSBC'
		self.models = Path(args[0] if len(args) > 0 else kw.get('models', root/'Models'))
		self.save = Path(args[1] if len(args) > 1 else kw.get('save', root))
		self.plot = kw.get('plot', False)

		# Extra keyword arguments
		self.extra = DotDict(**kw)

	def __str__(self):
		return str(vars(self))

	def __call__(self):
		if not self.models.is_dir():
			raise ValueError(f"{self.models} is not a directory.")

		self.eval_dir = self.save/'Evaluations'
		self.fig_dir = self.save/'Figures'
		self.eval_dir.mkdir(parents=True, exist_ok=True)
		self.fig_dir.mkdir(parents=True, exist_ok=True)
		self.bin_only = set(map(str.lower, ensure_iterable(self.extra.get('no_roc', ('ScleraMaskRCNN')), True)))

		plt.rcParams['font.family'] = 'Times New Roman'
		plt.rcParams['font.weight'] = 'normal'
		plt.rcParams['font.size'] = 24

		print("Sorting models by their binary F1-Score")
		self._sorted_models = sorted(os.listdir(self.models), key=lambda model: self._load(model, 'Overall')[1].f1score.mean, reverse=True)

		self._experiment1()
		for attrs in ATTR_EXP:
			self._experiment2(attrs)
		self._experiment3()

		if self.plot:
			plt.show()

	def _experiment1(self):
		print("Experiment 1: Overall performance")
		colours = CMAP(np.linspace(0, 1, len(self._sorted_models)))
		with ROC('Overall', self.fig_dir) as roc, Bar('Overall', self.fig_dir, self._sorted_models) as bar:
			for i, (model, colour) in enumerate(zip(self._sorted_models, colours)):
				pred_eval, bin_eval, plots, mean_plot, lower_std, upper_std = self._load(model, 'Overall')
				self._save_evals(pred_eval, bin_eval, f'{model} - Overall')
				roc.plot(mean_plot, lower_std, upper_std, label=model, colour=colour, bin_only=model.lower() in self.bin_only)
				bar.plot(bin_eval, i, colour=colour)

	def _experiment2(self, attrs):
		attrs = ensure_iterable(attrs, True)
		print(f"Experiment 2: Performance across different {', '.join(attr + 's' for attr in attrs)}")
		value_re = re.compile(', '.join(attr.title() + r"=([^,]*)" for attr in attrs))

		info = defaultdict(dict)
		for model in self._sorted_models:
			for f in os.listdir(self.models/model/'Pickles'):
				bname = os.path.splitext(f)[0]
				if value_re.fullmatch(bname):
					info[model][bname] = self._load(model, bname)

		bar_name = ','.join(attr.title() for attr in attrs)
		with Bar(bar_name, self.fig_dir, self._sorted_models, max(len(d) for d in info.values())) as bar:
			for i, (model, model_info) in enumerate(info.items()):
				#with ROC(f"{model} - {bar_name}", self.fig_dir) as roc:
					colours = iter(CMAP(np.linspace(0, 1, len(model_info))))
					for j, (bname, (pred_eval, bin_eval, plots, mean_plot, lower_std, upper_std)) in enumerate(model_info.items()):
						self._save_evals(pred_eval, bin_eval, f'{model} - {bname}')

						label = ",".join(value_re.fullmatch(bname).groups())
						colour = next(colours)
						#roc.plot(mean_plot, lower_std, upper_std, label=label, colour=colour, bin_only=model.lower() in self.bin_only)
						bar.plot(bin_eval, i, j, label=label, colour=colour)

	def _experiment3(self):
		print("Experiment 3: Performance across complexities")
		models_with_size = [model for model in self._sorted_models if model.lower() in model_complexity]
		models_with_both = [model for model in models_with_size if model_complexity[model.lower()][1] is not None]
		size_colours = CMAP(np.linspace(0, 1, len(models_with_size)))
		both_colours = CMAP(np.linspace(0, 1, len(models_with_both)))

		with Scatter('Size', self.fig_dir) as size, Scatter('Both', self.fig_dir) as both:
			for model, colour in zip(models_with_size, size_colours):
				bin_eval = self._load(model, 'Overall')[1]
				size.plot(model_complexity[model.lower()][0], bin_eval.f1score.mean, label=model, colour=colour)
				both.plot(model_complexity[model.lower()][0], bin_eval.f1score.mean, model_complexity[model.lower()][1], label=model, colour=colour)
		with Scatter('Full Only', self.fig_dir) as full:
			for model, colour in zip(models_with_both, both_colours):
				bin_eval = self._load(model, 'Overall')[1]
				full.plot(model_complexity[model.lower()][0], bin_eval.f1score.mean, model_complexity[model.lower()][1], label=model, colour=colour)

	def _load(self, model, name):
		name = self.models/model/f'Pickles/{name}.pkl'
		if not name.is_file():
			raise ValueError(f"{name} does not exist")
		print(f"Loading data from {name}")
		with open(name, 'rb') as f:
			return pickle.load(f) + pickle.load(f)

	def _save_evals(self, pred_eval, bin_eval, name):
		save = self.eval_dir/f'{name}.txt'
		print(f"Saving to {save}")
		with open(save, 'w', encoding='utf-8') as f:
			print("Probabilistic", file=f)
			print(pred_eval, file=f)
			print(file=f)
			print("Binarised", file=f)
			print(bin_eval, file=f)

	def process_command_line_options(self):
		ap = argparse.ArgumentParser(description="Evaluate segmentation results.")
		ap.add_argument('models', type=Path, nargs='?', default=self.models, help="directory with model information")
		ap.add_argument('save', type=Path, nargs='?', default=self.save, help="directory to save figures and evaluations to")
		ap.add_argument('-p', '--plot', action='store_true', help="show drawn plots")
		ap.parse_known_args(namespace=self)

		ap = argparse.ArgumentParser(description="Extra keyword arguments.")
		ap.add_argument('-e', '--extra', nargs=2, action='append', help="any extra keyword-value argument pairs")
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


class Figure(ABC):
	def __init__(self, name, save_dir, fontsize=24):
		self.name = name
		self.dir = save_dir
		self.fontsize = fontsize
		self.fig = None
		self.ax = None

	@abstractmethod
	def __enter__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.fig, self.ax = plt.subplots(*args, num=self.name, **kw)
		return self

	def __exit__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.close()
		plt.close(self.fig)

	@abstractmethod
	def close(self):
		pass

	@abstractmethod
	def plot(self):
		plt.rcParams['font.size'] = self.fontsize

	def save(self, name=None, fig=None):
		if fig is None:
			fig = self.fig
		if name is None:
			name = self.name
		for ext in FIG_EXTS:
			save = self.dir/f'{name}.{ext}'
			print(f"Saving to {save}")
			fig.savefig(save, bbox_inches='tight')


class ROC(Figure):
	def __init__(self, name, save_dir, fontsize=20):
		super().__init__(f'{name} ROC', save_dir, fontsize)
		self.cmb_fig = None
		self.cmb_ax = None
		self.zoom_ax = None

	def __enter__(self):
		super().__enter__()
		self.cmb_fig, self.cmb_ax = plt.subplots(num=f'{self.name} Combined')
		# This .81 has to be the diff of original ylims
		self.zoom_ax = zoomed_inset_axes(self.cmb_ax, .81 / abs(np.diff(ZOOMED_LIMS)[0]), bbox_to_anchor=(1.15, 0, 1, 1), bbox_transform=self.cmb_ax.transAxes, loc='upper left', borderpad=0)
		self.axes = self.ax, self.cmb_ax, self.zoom_ax
		for ax in self.axes:
			ax.grid(which='major', alpha=.5)
			ax.grid(which='minor', alpha=.2)
			ax.xaxis.set_major_formatter(FuncFormatter(def_tick_format))
			ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
			ax.margins(0)
			ax.set_xlabel("Recall")
			ax.set_ylabel("Precision")
		self.zoom_ax.set_xlabel(None)
		self.zoom_ax.set_ylabel(None)
		self.fig.tight_layout(pad=0)
		#self.cmb_fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		self.ax.set_xlim(.2, 1.01)
		self.ax.set_ylim(0, 1.01)
		self.ax.xaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
		self.ax.yaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
		self.save(f'{self.name} (No Legend)')

		_, labels = self.ax.get_legend_handles_labels()
		if labels:
			ncol = (len(labels) - 1) // 10 + 1
			legend = self.ax.legend(bbox_to_anchor=(1.02, .5), ncol=ncol, loc='center left', borderaxespad=0)
			self.save()

		self.ax.set_xlim(*ZOOMED_LIMS)
		self.ax.set_ylim(*ZOOMED_LIMS)
		self.ax.xaxis.set_major_locator(MultipleLocator(.1))
		self.ax.xaxis.set_minor_locator(MultipleLocator(.05))
		self.ax.yaxis.set_major_locator(MultipleLocator(.1))
		self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
		self.save(f'{self.name} (Zoomed)')

		if labels:
			legend.remove()
			self.save(f'{self.name} (Zoomed, No Legend)')

		self.cmb_ax.set_xlim(.2, 1.01)
		self.cmb_ax.set_ylim(.2, 1.01)
		self.cmb_ax.xaxis.set_major_locator(MultipleLocator(.2))
		self.cmb_ax.yaxis.set_major_locator(MultipleLocator(.2))
		mark_inset(self.cmb_ax, self.zoom_ax, loc1=2, loc2=3, ec='0.5')
		self.zoom_ax.set_xlim(*ZOOMED_LIMS)
		self.zoom_ax.set_ylim(*ZOOMED_LIMS)
		self.zoom_ax.xaxis.set_major_locator(MultipleLocator(.1))
		self.zoom_ax.xaxis.set_minor_locator(MultipleLocator(.05))
		self.zoom_ax.yaxis.set_major_locator(MultipleLocator(.1))
		self.zoom_ax.yaxis.set_minor_locator(MultipleLocator(.05))
		self.save(f'{self.name} (Combined, No Legend)', self.cmb_fig)

		if labels:
			self.cmb_ax.legend(bbox_to_anchor=(2.2, .5), loc='center left', ncol=ncol, columnspacing=.5, borderaxespad=0)
			self.save(f'{self.name} (Combined)', self.cmb_fig)

		plt.close(self.cmb_fig)

	def plot(self, mean_plot, lower_std=None, upper_std=None, *, label=None, colour=None, bin_only=False):
		super().plot()
		for ax in self.axes:
			if not bin_only:
				ax.plot(mean_plot.recall, mean_plot.precision, label=label, linewidth=2, color=colour)
				for std in lower_std, upper_std:
					if std is not None:
						ax.plot(std.recall, std.precision, ':', linewidth=1, color=colour)
				ax.plot(*mean_plot.f1_point, 'o', markersize=12, color=colour)
				ax.plot(*mean_plot.bin_point, 'o', markersize=12, color=colour, markerfacecolor='none')
			else:
				ax.plot(*mean_plot.bin_point, 'o', label=label, markersize=12, color=colour, markerfacecolor='none')

class Bar(Figure):
	def __init__(self, name, save_dir, groups, n=1, fontsize=30, margin=.2):
		super().__init__(f'{name} bar', save_dir, fontsize)
		self.groups = groups
		self.m = len(groups)
		self.n = n
		self.margin = margin
		self.width = (1 - self.margin) / self.n
		self.min = None
		self.max = None

	def __enter__(self):
		super().__enter__(figsize=(15, 5))
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_ylabel("F1-Score")
		self.fig.tight_layout(pad=0)
		self.min = float('inf')
		self.max = float('-inf')
		return self

	def close(self, *args, **kw):
		handles, labels = self.ax.get_legend_handles_labels()
		if labels:
			by_label = dict(zip(labels, handles))  # Remove duplicate labels
			for attempt in range(4, 1, -1):
				if len(by_label) % attempt == 0:
					ncol = attempt
					break
			else:
				ncol = 3
			self.ax.legend(by_label.values(), by_label.keys(), ncol=ncol, bbox_to_anchor=(.02, 1.02, .96, .1), loc='lower left', mode='expand', borderaxespad=0)
		ymin = max(self.min - .01, 0) if self.min != float('inf') else 0
		ymax = self.max + .01 if self.max != float('-inf') else 1.01
		self.ax.set_ylim(ymin, ymax)
		self.ax.set_xticks(np.arange(self.m) + (self.margin + self.n * self.width) / 2)
		self.ax.set_xticklabels(self.groups, rotation=60, ha='right', rotation_mode='anchor')
		if ymax - ymin >= .35:
			self.ax.yaxis.set_major_locator(MultipleLocator(.1))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
		else:
			self.ax.yaxis.set_major_locator(MultipleLocator(.05))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.025))
		self.save()
		self.ax.tick_params(axis='x', bottom=False, labelbottom=False)
		self.save(f'{self.name} (No Labels)')

	def plot(self, evaluation, group=0, index=0, *, label=None, colour=None):
		super().plot()
		plt.rcParams['font.size'] = 10
		err_w = np.clip(self.width * 10, 2, 5)
		self.ax.bar(
			group + self.margin / 2 + index * self.width,
			evaluation.f1score.mean,
			yerr=evaluation.f1score.std,
			error_kw=dict(lw=err_w, capsize=1.5 * err_w, capthick=.5 * err_w),
			width=self.width,
			align='edge',
			label=label,
			color=colour
		)
		self.min = min(self.min, evaluation.f1score.mean - evaluation.f1score.std)
		self.max = max(self.max, evaluation.f1score.mean + evaluation.f1score.std)


class Scatter(Figure):
	def __init__(self, name, save_dir, fontsize=28):
		super().__init__(f'{name} scatter', save_dir, fontsize)
		self.min = None
		self.max = None

	def __enter__(self):
		super().__enter__()
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.set_xscale('log')
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_xlabel("# Parameters")
		self.ax.set_ylabel("F1-Score")
		self.fig.tight_layout(pad=0)
		self.min = float('inf')
		self.max = float('-inf')
		return self

	def close(self, *args, **kw):
		ymin = max(self.min - .1, 0) if self.min != float('inf') else 0
		ymax = self.max + .1 if self.max != float('-inf') else 1.01
		self.ax.set_xlim(1, 3e8)
		self.ax.set_ylim(ymin, ymax)
		if ymax - ymin >= .35:
			self.ax.yaxis.set_major_locator(MultipleLocator(.1))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
		else:
			self.ax.yaxis.set_major_locator(MultipleLocator(.05))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.025))
		self.save(f'{self.name} (No Legend)')
		if self.ax.get_legend_handles_labels()[0]:
			self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
		self.save()

	def plot(self, x, y, size=None, *, label=None, colour=None):
		super().plot()
		markersize = 8
		flopsize = 0
		if size:
			#flopsize = 2e-4 * math.sqrt(size)
			flopsize = 2 * math.log(size)
			self.ax.plot(x, y, 'o', markersize=flopsize, color=(*colour[:3], .3))
		self.ax.plot(x, y, 'o', markersize=markersize, label=label, color=colour)
		self.min = min(self.min, y)
		self.max = max(self.max, y)


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
		self.save_lbl = Label(self.frame, text="Save to:")
		self.save_lbl.grid(column=0, row=row, sticky='w')
		self.save_txt = Entry(self.frame, width=60)
		self.save_txt.insert(END, self.args.save)
		self.save_txt.grid(column=1, columnspan=3, row=row)
		self.save_btn = Button(self.frame, text="Browse", command=self.browse_save)
		self.save_btn.grid(column=4, row=row)

		row += 1
		self.chk_frame = Frame(self.frame)
		self.chk_frame.grid(row=row, columnspan=3, sticky='w')
		self.plot_var = BooleanVar()
		self.plot_var.set(self.args.plot)
		self.plot_chk = Checkbutton(self.chk_frame, text="Show plots", variable = self.plot_var)
		self.plot_chk.grid(sticky='w')

		row += 1
		self.extra_frame = ExtraFrame(self.frame)
		self.extra_frame.grid(row=row, columnspan=3, sticky='w')

		row += 1
		self.ok_btn = Button(self.frame, text="OK", command=self.confirm)
		self.ok_btn.grid(column=1, row=row)
		self.ok_btn.focus()

	def browse_models(self):
		self._browse_dir(self.models_txt)

	def browse_save(self):
		self._browse_dir(self.save_txt)

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
		self.args.save = Path(self.save_txt.get())
		self.args.plot = self.plot_var.get()

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
