# Hydra: an Ensemble of Convolutional Neural Networks for Geospatial Land Classification

This repository releases models and code for [Hydra](http://arxiv.org/abs/1710.07662), our submission for the functional Map of the World (fMoW) challenge. This solution was ranked 3rd in the [final scoreboard](https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16996).

## Authors

- Rodrigo Minetto - Universidade Tecnlógica Federal do Paraná
- Mauricio Pamplona Segundo - Universidade Federal da Bahia
- Sudeep Sarkar - University of South Florida

## Functional Map of the World description

The fMoW challenge consists of creating automatic solutions to classify a specific given location as one of the 62 target classes (_e.g._ airport, flooded road, nuclear power plant and so on) or as none of them (false detections). It is sponsored by the Intelligence Advanced Research Projects Activity (IARPA), an organization within the Office of the USA Director of National Intelligence. fMoW images vary in quality and are distributed over more than 100,000 globe locations, which leads to high intraclass variations and considerable interclass confusion. This, added to traditional satellite imaging problems like viewpoint, weather, shadow and scale variations, makes this classification problem a lot harder than previous land use datasets, such as UC Merced Land Use Dataset, WHU-RS19 and NWPU-RESISC45. Finally, the fMoW challenge also limits time and computational resources for training and testing to minimize the disparity among participants' solutions.

## Hydra description

Hydra is a framework that creates ensembles of Convolutional Neural Networks (CNN) for land use classification in satellite images. The idea behind Hydra is to create an initial CNN that is coarsely optimized but provides a good starting pointing for further optimization, which will serve as the Hydra's body. Then, the obtained weights are fine tuned multiple times to form an ensemble of CNNs that represent the Hydra's heads. The Hydra framework tackles one of the most common problem in multiclass classification, which is the existence of several local minima that prioritize some classes over others and the eventual absence of a global minimum within the classifier search space. The ensemble ends up expanding this space by combining multiple classifiers that converged to local minima and reaches a better global approximation. To stimulate convergence to different end points, we exploit different strategies, such as using online data augmentation, variations in the size of the region of interest, and different image formats released by the fMoW challenge. The classifiers employed in our Hydra framework are variations of the [fMoW baseline code](https://github.com/fmow/baseline).

## Requirements

- Keras with TensorFlow backend
- nvidia-docker

## Instructions

Download the [fMoW-rgb dataset](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16996&pm=14684), uncompress it and then execute the following sequence of commands:

```
$ git clone https://github.com/maups/hydra-fmow
```

