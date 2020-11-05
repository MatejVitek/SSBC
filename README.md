# About
This code was used to produce the evaluation results for the paper ["SSBC 2020: Sclera Segmentation and Benchmarking Competition in the Mobile Environment"](https://sclera.fri.uni-lj.si/publications.html#SSBC_2020).

It is designed for experiments on the [MOBIUS dataset](https://sclera.fri.uni-lj.si/datasets.html#MOBIUS), however it can be adapted to other datasets as well.

# Requirements
Our [utility libraries](https://sclera.fri.uni-lj.si/code.html#Libraries) are required to make this project work. The additional required python packages are:

	joblib
	matplotlib
	numpy
	pillow
	scikit-learn
	scipy
	tqdm

# Running the code
This project is a stripped-down version of [our Toolbox](https://sclera.fri.uni-lj.si/code.html#Toolbox). The project's functionality is divided into two parts — computation (slow and memory-intensive) and plotting (fast and efficient).

## Computation
The computation part takes as input the sclera masks that are the result from your model(s) and the ground truth information. It computes the precision/recall/... information and saves it to `.pkl` files ("pickles"). All this is handled by the script `compute.py`. To run the script, use the following syntax:

	python compute.py "/path/to/model/results" "/path/to/ground/truth"

The directory `/path/to/model/results/` should contain a subdirectory for each of your segmentation models. If you are only computing results for a single segmentation model, use only one subdirectory. Inside each of these model subdirectories should be two folders: `Predictions` (which contains greyscale probabilistic sclera masks output by your model before thresholding) and `Binarised` (which contains the binary black & white sclera masks obtained after thresholding). See below for a sample tree structure with two models called Segmentor and Segmentator:

	/path/to/model/results/
	│─── Segmentor/
	│	│─── Predictions/
	│	│   │─── 1_1i_Ll_1.png
	│	│   │─── 1_1i_Ll_2.png
	│	│   │─── 1_1i_Lr_1.png
	│	│   └─── ...
	│	└─── Binarised/
	│		│─── 1_1i_Ll_1.png
	│		│─── 1_1i_Ll_2.png
	│		│─── 1_1i_Lr_1.png
	│		└─── ...
	└─── Segmentator/
		│─── Predictions/
		│   └─── ...
		└─── Binarised/
			└─── ...

The directory `/path/to/ground/truth/` should contain the ground truth information bundled with the evaluation dataset. All the images inside should be in the base directory. All the files in `/path/to/ground/truth/` should have a corresponding entry in each of the `Predictions` and `Binarised` folders of all of your models in `/path/to/model/results/`, otherwise the evaluation cannot be executed fairly and an error will be reported.

Other arguments are not required, but can be useful. For a full list see the output of `compute.py --help`. For instance, a smaller image size (`-r`) can lead to less memory usage but will be less reliable in the calculations. The script can also be run without arguments, in which case it will launch as a simple GUI application.

When running the code on the MOBIUS and SBVPI datasets you may encounter the following warning:

	Warning: Extra info file not found. Only information parsed from file names will be available.
This is safe to ignore if you only wish to run the experiments included in the scripts. If you wish to design your own experiments based on subject attributes available in the datasets (age, gender, presence of lenses, eye conditions, etc. — see the [dataset descriptions](https://sclera.fri.uni-lj.si/datasets.html) for detailed information), you'll have to edit the `extra_info_path` argument of the `from_dir` method in the corresponding file (`datasets/mobius.py` or `datasets/sbvpi.py`) to point to the correct path of the subject information file (which you should have received with the dataset).

## Plotting and evaluation
The plotting and quantitative evaluation is handled by `plot.py`. This script takes as input the `.pkl` files produced by `compute.py` and creates and saves various different plot figures and textual quantitative evaluations. To run the script, use the following syntax:

	python plot.py "/path/to/model/results" "/path/to/save/to"

The directory `/path/to/model/results/` should contain a subdirectory for each of your segmentation model(s). This subdirectory should contain the folder `Pickles` where the `.pkl` files produced by `compute.py` are located. This tree structure will be produced automatically if you use the same `/path/to/model/results/` for both scripts, but should be respected if you decide to include pickles from other models (such as the pickles of existing submitted models available on the [benchmarking site](https://sclera.fri.uni-lj.si/benchmarking.html#Code)).

If you want your model to be included in the generated scatterplot of model complexities as well, you will also have to add your model to `model_complexity` in `plot.py`. An entry in `model_complexity` consists of the model name (which should be lowercase but otherwise identical to the name of your model folder) and a tuple of: the number of trainable parameters (you can use 1 if your solution has no trainable parameters) and floating point operations (FLOP) required for a single forward pass (inference). See the existing entries for a template of how to add your own.

The plots and quantitative evaluations will be saved to `/path/to/save/to`. We also provide an example convenience `latexify.py` script that turns the text files of the quantitative evaluations into LaTeX-style table entries, however the output of this script will likely need to be adapted for your needs depending on your LaTeX document.

# Adapting the code
The project can also be used to design your own experiments on MOBIUS/SBVPI or even adapted to a different dataset.

## Designing your own experiments
The project currently contains experiments that look at overall model performance (`_experiment1`), performance across different attributes (`_experiment2`), and how model complexity relates to model performance (`_experiment3`). To add your own, you can look at the three `_experimentX` methods in `compute.py` and `plot.py` and adapt them to your own needs.

The attribute-specific experiments in `_experiment2` are currently designed for the MOBIUS dataset, as they contain experiments related to different lighting conditions, phones, gaze directions, and even combining different lighting conditions and phones in a single experiment. You can easily change what attribute-specific experiments should be run in `ATTR_EXP` in `compute.py`. For a full list of available attributes in MOBIUS/SBVPI see the `MOBIUSSample` (or `SBVPISample`) and `_ExtraInfo` classes in `datasets/mobius.py` and `datasets/sbvpi.py`. Note that using subject-specific attributes available in MOBIUS/SBVPI will also require you to correct the `extra_info_path` argument in the `from_dir` method in the corresponding file.

## New dataset
If you are not interested in performance across different attributes (in which case you should of course make appropriate changes to `compute.py` and `plot.py` to not run `_experiment2` — see previous section for more info), the simplest way to adapt to a new dataset is to rename all your images (and corresponding ground truth and prediction masks) to purely numerical names (e.g. `1.png`, `2.png`, ...), then running `compute.py` with the argument `--dataset None`.

Otherwise you will have to define your own dataset file. For the sake of demonstration, let's assume your dataset is called COD (Cool Ocular Dataset). In the folder `datasets` you will have to create a new file called `cod.py`. You can look at `mobius.py` in the same directory to get a better idea of what this file is supposed to contain. The `cod.py` file should contain:

- `CODSample`, which is a subclass of `Sample`. This is the key component that specifies what attributes are available for each image. It should contain:
	- A regex expression that expresses how file names should be parsed into attributes (such as gaze direction or lighting conditions). For the experiments defined in our code at least the attributes `gaze`, `light`, and `phone` should be present.
	- A method called `_postprocess_attributes` where the parsed attributes are postprocessed. By default they will be parsed from the regex as strings, so you may need to convert them to the appropriate types.
- `COD`, which is a subclass of `Dataset`. This reads the samples from a directory and (optionally) reads attributes related to the subjects in the dataset (such as age, gender, eye colour, etc.).
	- At the very least this should contain the line `sample_cls = CODSample`.
	- Optionally you can override `from_dir` to define your own method for reading the samples from a directory. This is likely unnecessary unless you need extra subject-related attributes for your experiments. **If you wish to use the default method without overriding it (recommended), your `CODSample` should also contain an `id` attribute in its regex.**

Next, open `datasets/__init__.py` and add the following line:

	from .cod import COD

Finally, you'll have to update `compute.py`. First off, search for the line

	if self.dataset.lower() == 'mobius':`
In this same `if` block you should add the case for your own dataset (see how SBVPI is added right below MOBIUS to understand this better):

	elif self.dataset.lower() == 'cod':
		from datasets import COD
		dataset = COD

Next, search for the line

	ap.add_argument('-d', '--dataset', type=str.lower, choices=('mobius', 'sbvpi', 'none'), help="dataset file naming protocol used")
and add `'cod'` to the `choices`.

Optionally you can also set `COD` to be the default dataset in `compute.py` (instead of MOBIUS) by searching for the line

	self.dataset = kw.get('dataset', 'MOBIUS')
and changing `'MOBIUS'` to `'COD'`. Alternatively you can run `compute.py` with the argument `--dataset COD`.
