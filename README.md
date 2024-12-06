# Siamese Neural Network - Fish identification project

## src

This is the main source folder for this project. It contains all code needed to test and run this project. In addition there are also i tiny dataset of images that can be used to run and test the code.

## Dataset used with this project

The dataset used for this project is sourced from the Fish4Knowledge repository, a comprehensive collection of underwater fish images acquired from live video streams. The dataset includes 27,370 manually labeled fish images organized into 23 distinct clusters, with each cluster representing a unique fish species. The grouping is based on morphological characteristics such as the presence or absence of specific fins, shapes, or other anatomical features, as defined by marine biologists.

[Homepage Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/)

Dataset used with this project can be downloaded from here: [Downloadable data from Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/)

## Web Application (Flask)

The web applicattion is still under development and are not yet integrated with the main src code. The main src-code supports testing of a single image such that the snn model can predict based on best similarity score. 

API support testing on a pretrained snn-model by its own and do some analyzing of similarity score (distance).