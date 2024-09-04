# GlizzyNet

GlizzyNet is a fine tuned model which utilizes Faster RCNN with ResNet 50 backbone. This project fine tunes and analyzes the results of the process.

This project is part of my primary application and desire to join the 2D perception team at aUToronto.

## Data Setup
To start, we must ensure we have the data. I downloaded the data from a google drive location.

https://drive.google.com/drive/folders/1UoM2fdY9uALEiuDtHyTwNU3ldgD1l6oB 

Once downloaded, you should have ``` Hot Dog Detection COCO.zip``` and  ``` Hot Dog Detection YOLO.zip```. Move these over to this base directory. Also unzip them.

## Environment Setup

I developed my code in VS Code jupyter notebooks. Though I did do this with a conda environment. You'll need to ensure you have conda on your system.

The main dependencies my code uses is as follows. Feel free to setup your environment however you like. Though I've included instructions on how I set mine up.

```
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - ipykernel
  - cython
  - requests
  - PIL
  - pytorch
  - torch
  - torchvision
  - pycocotools
  - torchmetrics
```

Once you have conda we can start the environment setup. I included my exact conda yml enviroment file, so all you have to do is create a new environment with that. This should ensure that you have and run with all of the things I have. I may have a few extra libraries, since I was running some of this code on my M1 MacBook. Though see the main dependencies above.

Ensure you have conda-forge set up as a channel

```bash
conda config --add channels conda-forge
```

```bash 
conda env create --file environment.yml --name my_new_env
```

Then we want to activate the environment

```bash
conda activate my_new_env
```

If you go to the hot_dog_detection.ipynb file in VS Code, select your enviornment you just created as your jupyter kernel. 

With your data in the current folder, and environment setup, you should be able to run the code.

Note, if any of the code has trouble running, it will definitely run in Google Colab. Though you must mount the dataset to the run environment.

## Hot Dog Detection

The main code will be found in the notebook file, hot_dog_detection.ipynb. This code has everything from data ingestion, training, and evaluation.

I have documented all of my findings, and future considerations in my PDF report, so please see that for all of my design decisions, troubles, etc.

For post-training analysis, I used the model

checkpoints_bs64/model_epoch_15.pth

Though I wasn't able to include it in the submission, the result validation used this model path.

As outlined in my report, this was my best performing model, so I used this for all of my analysis.


