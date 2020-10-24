from joblib.parallel import Parallel, delayed
from matej.parallel import tqdm_joblib
import numpy as np
from pathlib import Path
from PIL import Image
import re
import sklearn.metrics as skmetrics
import sys
from tqdm import tqdm

from datasets import MOBIUS, SBVPI, Dataset
from evaluation import Evaluation, Metric


ALPHANUM_RE = re.compile(r'[\W_]+')


class SegmentationEvaluation(Evaluation):
	def __init__(self, *args):
		super().__init__(IoU(), *args)
	
	def evaluate(self, pred_dir, gt_dir, dataset='mobius', k=1, workers=-1, imagewise_std=False):
		pred_dir = Path(pred_dir)
		gt_dir = Path(gt_dir)
		if not pred_dir.is_dir():
			raise ValueError(f"{pred_dir} is not a directory")	
		if not gt_dir.is_dir():
			raise ValueError(f"{gt_dir} is not a directory")
		if dataset.lower() == 'mobius':
			dataset = MOBIUS
		elif dataset.lower() == 'sbvpi':
			dataset = SBVPI
		else:
			dataset = Dataset

		gt = dataset.from_dir(gt_dir, mask_dir=None)

		for _ in tqdm(range(k)):
			with tqdm_joblib(tqdm(desc="Processing", total=len(gt))) as progress_bar:
				img_metrics = dict(Parallel(n_jobs=workers)(
					delayed(self._process_image)(gt_sample, pred_dir, progress_bar)
					for gt_sample in gt
				))
			
			for metric in self.values():
				if not isinstance(metric, Metric):
					continue
				if imagewise_std:
					for metrics in img_metrics.values():
						metric.update(metrics[metric.name])
				else:
					metric.update(np.array([metrics[metric.name] for metrics in img_metrics.values()]).mean)
	
	def _process_image(self, gt_sample, pred_dir, progress_bar=None):
		gt_f = gt_sample.f
		pred_f = pred_dir/gt_f.name
		if not pred_f.is_file():
			print(f"Missing prediction file {pred_f}.", file=sys.stderr)
			return
		if progress_bar:
			progress_bar.set_postfix(file=gt_f.name)

		gt = np.array(Image.open(gt_f).convert('1')).flatten()
		pred = np.array(Image.open(pred_f).convert('L')).flatten() / 255

		return gt_f.name, self._compute_metrics(gt, pred)

	def _compute_metrics(self, gt, pred):
		return {metric.name: metric(gt, pred) for metric in self.values() if isinstance(metric, Metric)}


class BinarySegmentationEvaluation(SegmentationEvaluation):
	def __init__(self, *args):
		super(SegmentationEvaluation, self).__init__(IoU('binary'), F(), Precision(), Recall(), AUC(), *args)

	def _compute_metrics(self, gt, pred):
		print("Computing precision/recall curve")
		precisions, recalls, thresholds = skmetrics.precision_recall_curve(gt, pred)

		# TODO: Handle threshold selection
		threshold = .5

		print("Computing IoU")
		iou = self.iou(gt, pred >= threshold)
		print("Computing F1-score")
		#self.f1score.compute_and_update(precision=self.precisions[idx], recall=self.recalls[idx])
		#self.f1score.compute_and_update(prediction_scores >= self.threshold, gt)
		f1 = self.f1(gt, pred >= threshold)
		#self.precision.update(self.precisions[idx])
		self.precision.compute_and_update(prediction_scores >= self.threshold, gt)
		#self.recall.update(self.recalls[idx])
		self.recall.compute_and_update(prediction_scores >= self.threshold, gt)
		print("Computing AUC")
		self.auc.compute_and_update(precisions=self.precisions, recalls=self.recalls)


class IoU(Metric):
	def __init__(self, average='macro', *args, **kw):
		super().__init__(*args, **kw)
		self.average = average

	def __call__(self, gt, predictions):
		return skmetrics.jaccard_score(gt, predictions, average=self.average)


class F(Metric):
	def __init__(self, beta=1, *args, **kw):
		super().__init__(*args, **kw)
		self.name = f"F{beta}-score"
		self.beta = beta

	def __call__(self, gt=None, predictions=None, *, precision=None, recall=None):
		if precision is not None and recall is not None:
			# Edge case
			if precision == recall == 0:
				return 0
			return (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall)
		else:
			return skmetrics.fbeta_score(gt, predictions, beta=self.beta)


class Precision(Metric):
	def __call__(self, *args, **kw):
		return skmetrics.precision_score(*args, **kw)


class Recall(Metric):
	def __call__(self, *args, **kw):
		return skmetrics.recall_score(*args, **kw)


class AUC(Metric):
	def __call__(self, gt=None, predictions=None, *, precisions=None, recalls=None):
		if precisions is not None and recalls is not None:
			return skmetrics.auc(recalls, precisions)
		else:
			return skmetrics.roc_auc_score(gt, predictions)
