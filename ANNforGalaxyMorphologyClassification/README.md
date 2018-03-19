# Description
This work corresponds to my master thesis project. Here, I programmed using Python an Artificial Neural Network for classifying Morphology of Galaxies. Below, I share the abstract.
---
## Abstract

Galaxy morphology classification is a central study subject for the understanding of galactic
evolution. Ever since the first classification schemes appeared, such as the Hubble sequence in
1926, the classification of galaxies have been made by humans through visual inspection. This
approach became unworkable as the data explosion arose, making available data sets containing
even billion objects. To overcome this problem, the objective of this work was to replicates via
machine learning, and in particular via feed-forward multi-layer neural networks, the classication
of galaxies made in the Galaxy Zoo project, considering three classes: ellipticals, spirals, and
starsstarnartefacts.

Two diferent neural network architectures were considered, (1) with one hidden layer and (2)
with two hidden layers. Twelve features based on colours, profile fitting and adaptive moments
of galaxies were chosen as input parameters for training the network. The networks were trained
over three diferent subsets from the Galaxy Zoo 1 catalogue. These subsets were defined applying
cuts of 0% (raw subset), 80% (clean subset) and 95% (super-clean subset) on the probabilities
obtained from the classifications made by the Galaxy Zoo project volunteers. The neural networks
were then tested over the testing partition of the Galaxy Zoo 1 catalogue and the whole Galaxy
Zoo 2 catalogue.
The results showed for the Galaxy Zoo 1 catalogue and a network architecture with one
hidden layer, accuracies of 81%, 95.8% and 97.3% for the raw, clean and super-clean subsets,
respectively. For the same catalogue, when considering a network architecture with two hidden
layers, accuracies of 81.8%, 96.1% and 97.5% were obtained. For the Galaxy Zoo 2 catalogue and
a network architecture with one hidden layer, accuracies of 74%, 80% and 83% were obtained.
Finally for the same catalogue, when considering a network architecture with two hidden layers,
accuracies of 78%, 81% and 86% were obtained.
In summary, in both Galaxy Zoo catalogues, accuracies were higher as probability cuts were
higher. This can be explained by considering that higher probabilities translate into more defined morphological properties and consequently, into better classification rules. Also, the neural
networks performed better on the Galaxy Zoo 1 catalogue. The reason is the Galaxy Zoo 2 catalogue
includes objects with finer morphological properties making the underlying distributions
of the input parameters among classes diferent than the ones in the Galaxy Zoo 1 catalogue.
In addition, using a network architecture with two hidden layers only improved marginally the
classification performance. Finally, it is possible to reproduce successfully via machine learning
the classifications made by human eye in the Galaxy Zoo project and to extend the classification
model to other galaxy morphology catalogues.
