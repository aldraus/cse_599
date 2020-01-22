# CSE 490/599 G1 Introduction to Deep Learning


## What the heck is this codebase? ##

During this class you'll be building out your own neural network framework. We'll take care of some of the more boring parts like loading data, stringing things together, etc. so you can focus on the important parts.
We will be implementing everything using [Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html) and [Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html) in a [Conda](https://docs.conda.io/en/latest/) environment.
If you are not familiar with them, take a few minutes to learn the ins and outs. PyTorch uses a Numpy-like interface, so it will be good to know for the other parts of the homework as well.

## Homework 0. Setup Conda and the codebase ##
First install Miniconda (https://docs.conda.io/en/latest/miniconda.html). Then run the following commands.

```bash
conda create -n dl-class python=3.6.9
git clone https://gitlab.com/danielgordon10/dl-class-2019a.git
cd dl-class-2019a
conda deactivate
conda env update -n dl-class -f environment.yml
conda activate dl-class
```

Although you don't technically _have to_ use Anaconda and Python 3.6, this is how we will be grading it, and if it doesn't work on our grading computer, we won't try too hard to make it work.
If you are still using Python 2.7, you can install Anaconda without messing up your current Python installation. Anaconda also makes sure that the installed libraries don't affect other projects on the same machine.

Before starting each homework, you will first need to update the repository to get the new files.
```bash
git add -A
git commit -m "before starting new homework"
git pull origin master
```

## [Homework 1. Making Your First Neural Network](hw1) - Due 10/16/19 3:29PM
## [Homework 2. Convolutional Neural Networks](hw2) - Due 11/6/19 3:29PM
## [Homework 3. Recurrent Neural Networks](hw3) - Due 11/15/19 3:29PM

