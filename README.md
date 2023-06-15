
![Competition_Logo](https://drivendata-public-assets.s3.amazonaws.com/conservision-banner.jpg)

# Tierli_ahluege ccv1

This project is a part of the [CCV1 Tierli Ahluege Group](https://gitlab.fhnw.ch/thomas.mandelz/tierli_ahluege) at [Data Science FHNW](https://www.fhnw.ch/en/degree-programmes/engineering/bsc-data-science).

This Repository is our solution to the [Competition: Conser-vision Practice Area: Image Classification](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).

## Project Status: Completed

## Project Intro/Objective

Can you classify the wildlife species that appear in camera trap images collected by conservation researchers?

Welcome to the African jungle! In recent years, automated surveillance systems called camera traps have helped conservationists study and monitor a wide range of ecologies while limiting human interference. Camera traps are triggered by motion or heat, and passively record the behavior of species in the area without significantly disturbing their natural tendencies.

However, camera traps also generate a vast amount of data that quickly exceeds the capacity of humans to sift through. That's where machine learning can help! Advances in computer vision can help automate tasks like species detection and classification, localization, depth estimation, and individual identification so humans can more effectively learn from and protect these ecologies.

In this challenge, we will take a look at object classification for wildlife species. Classifying wildlife is an important step to sort through images, quantify observations, and quickly find those with individual species.

This is a practice competition designed to be accessible to participants at all levels. That makes it a great place to dive into the world of data science competitions and computer vision. Try your hand at image classification and see what animals your model can find!

### Methods Used

* Deep Learning
* Computer Vision
* Image Classification
* CNN
* Object Detection
* Explorative Dataanalysis
* Data Visualization

### Technologies

* Python
* PyTorch
* wandb
* numpy
* Pandas
* Megadetector

## Featured Files

* To use the best Model with some demo images, use this notebook: [Demo Model Notebook for the best model](demo/demo_modell.ipynb)
* To inspect the training of the best model, use this notebook: [Training Notebook for the best model](modelling/convnext_megadetector_ensemble.ipynb)
* To inspect the Explorative Dataanalysis for the whole dataset, use this notebook: [Explorative Dataanalysis Notebook](Eda/eda.ipynb)
* If you only want the models, refer to this folder: [Folder with the best model](model_submit)

## Getting Started

* Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
* Demo files are being kept [here](demo)
* Raw Data is being kept [here](competition_data)
* Explorative Dataanalysis Scripts and Files are being kept [here](Eda)
* Megadetector Scripts and data is being kept [here](megadetector)
* Models are being kept [here](model_submit)
* Models submissions are being kept [here](data_submit)
* Source files for training are being kept [here](modelling)
* Source files for pipeline are being kept [here](src)


## Pipenv for Virtual Environment

### First install of Environment

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv install`
* Restart VS Code
* Choose the newly created "tierli_ahluege" Virtual Environment python Interpreter

### Environment already installed (Update dependecies)

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv sync`

## Contributing Members

* **[Thomas Mandelz](https://github.com/tmandelz)**
* **[Manuel Schwarz](https://gitlab.fhnw.ch/manuel.schwarz1)**
* **[Jan Zwicky](https://github.com/swiggy123)**
