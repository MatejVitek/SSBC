#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
import os
import re


def fmt(x):
	return np.format_float_positional(x, precision=3, unique=False)


def alphanum(x):
	return re.sub(r'[^0-9a-zA-Z]', '', x)


def rec_dd():
	return defaultdict(rec_dd)


data = rec_dd()
for fname in os.listdir('Evaluations'):
	bname = os.path.splitext(fname)[0]
	model, experiment = bname.split(' - ')

	with open(os.path.join('Evaluations', fname), 'r', encoding='utf-8') as in_file:
		bin_prob = None
		for line in in_file.readlines():
			line = line.strip()
			if line in {'Probabilistic', 'Binarised'} or line == '':
				if line != '':
					bin_prob = line
				continue
			metric, value = line.split(': ')
			metric = metric.rstrip(' (μ ± σ)')
			mean, std = map(float, value.split(' ± '))
			data[experiment][bin_prob][model][metric] = mean, std

sorted_models = sorted(data['Overall']['Binarised'].keys(), key=lambda model: data['Overall']['Binarised'][model]['F1-score'], reverse=True)
with open('latexified.txt', 'w') as out_file:
	for exp in 'Overall',:
		for model in sorted_models:
			normal = model.lower() not in {'mask2020cl', 'scleramaskrcnn'}
			# DeepLab & $0.916 \pm 0.003$ & $0.905 \pm 0.005$ & $0.926 \pm 0.007$ & $0.908 \pm 0.003$ & ... \\
			print(*(
				[model.ljust(max(map(len, sorted_models))) if normal else model + r"$^\dagger$"] +
				["$" + r" \pm ".join(map(fmt, data[exp]['Binarised'][model][metric])) + "$" for metric in ('F1-score', 'Precision', 'Recall', 'IoU')] +
				(["$" + r" \pm ".join(map(fmt, data[exp]['Probabilistic'][model][metric])) + "$" for metric in ('F1-score', 'AUC')] if normal else ["n/a", "n/a"])
			), sep=" & ", end="\\\\\n", file=out_file)
