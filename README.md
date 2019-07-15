# Emergence of exploratory look-around behaviors through active observation completion


This repository contains the code for the paper:

[Emergence of exploratory look-around behaviors through active observation completion](http://vision.cs.utexas.edu/projects/visual-exploration/)  
Santhosh K. Ramakrishnan, Dinesh Jayaraman, Kristen Grauman   
Science Robotics 2019

## Note
This is a cleaned version of the original code used to generate the results from the paper. As a result, there may be small differences in the actual results obtained by training models using the code. Please contact me if further details are needed. 

## Setup
- First install anaconda and setup a new environment. Install anaconda from: https://www.anaconda.com/download/

```
conda create -n spl python=2.7
source activate spl
```
- Clone this repository and setup requirements through pip.

```
git clone https://github.com/srama2512/visual-exploration.git
cd visual-exploration
pip install -r requirements.txt
```

- Download preprocessed SUN360 and ModelNet data.

```
mkdir data
cd data
wget http://vision.cs.utexas.edu/projects/sidekicks/scirobo-2019-data.zip
unzip scirobo-2019-data.zip
```
- Add the repository to `PYTHONPATH`. Please add this line to `~/.bashrc`.

```
export PYTHONPATH=<path-to-repository>:$PYTHONPATH
```


The downloaded zip file will consist of the following data:

- Lookaround task: 
	- `data/SUN360/lookaround.h5` 
	- `data/ModelNet/lookaround_modelnet40.h5`
	- `data/ModelNet/lookaround_modelnet10.h5`
- Recognition task:
	- `data/SUN360/recognition.h5`
	- `data/ModelNet/recognition_modelnet10.h5`
- Light source localization task:
	- `data/ModelNet/lsl_modelnet10.h5`
	- `data/ModelNet/lsl_labels_modelnet10.h5`
- Metric tasks:
	- `data/ModelNet/metric_labels_modelnet10.h5`

The source code for individual tasks are provided in `src/`. Each task has its own `train.py` and `eval.py` scripts. 

## TODO
- Provide pre-trained models
- Instructions for evaluating pre-trained models
- Instructions for training task-specific models
