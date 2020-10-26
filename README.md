# About
This code was used to produce the evaluation results for the paper "SSBC 2020: Sclera Segmentation and Benchmarking Competition in the Mobile Environment".

# Requirements
For the installation process and required libraries, please see the instructions at our [Sclera website](https://sclera.fri.uni-lj.si/code.html#ssbc).

# Running the code
This project is a stripped-down version of [our Toolbox](https://sclera.fri.uni-lj.si/code.html#toolbox). The project's functionality is divided into two parts — computation (slow and memory-intensive) and plotting (fast and efficient).

## Computation
The computation part takes as input the sclera masks that are the result from your model(s) and the ground truth information. It computes the precision/recall/... information and saves it to `.pkl` files ('pickles'). All this is handled by the script `compute.py`. To run the script, use the following syntax:

	python compute.py "/path/to/model/results" "/path/to/ground/truth"

The directory `/path/to/model/results/` should contain a subdirectory for each of your segmentation model(s). If you are only computing results for a single segmentation model, use only one subdirectory. Inside each of these model subdirectories should be two folders: `Predictions` (which contains greyscale probabilistic sclera masks output by your model before thresholding) and `Binarised` (which contains the binary black & white sclera masks obtained after thresholding). See below for a sample tree structure with two models called Segmentor and Segmentator:

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
The plotting and quantative evaluation is handled by `plot.py`. This script takes as input the `.pkl` files produced by `compute.py` and creates and saves various different plot figures and textual quantative evaluations. To run the script, use the following syntax:

	python plot.py "/path/to/model/results" "/path/to/save/to"

The directory `/path/to/model/results/` should contain a subdirectory for each of your segmentation model(s). This subdirectory should contain the folder `Pickles` where the `.pkl` files produced by `compute.py` are located. This tree structure will be produced automatically if you use the same `/path/to/model/results/` for both scripts, but should be respected if you decide to include pickles from other models (such as the pickles of existing submitted models available on the [benchmarking site](https://sclera.fri.uni-lj.si/benchmarking.html#code)).

The plots and quantative evaluations will be saved to `/path/to/save/to`. We also provide an example convenience `latexify.py` script that turns the text files of the quantative evaluations into LaTeX-style table entries, however the output of this script will likely need to be adapted for your needs depending on your LaTeX document.