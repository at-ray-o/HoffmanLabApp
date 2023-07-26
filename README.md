# Hoffman Lab Application
Code sample for Hoffman lab application.

## Code description:
The code trains a U-net autoencoder in a supervised fashion. Both the input and output are part of the ```.npy``` file, provided in this repository.

## Dependency:

Pytorch 1.9.0\
Numpy 1.21.2\
matplotlib 3.4.3

## How to run:

Load an anaconda pytorch environment.\
Run the code as ```python DLHoffmanLabSample.py```

## Outputs:

```Lossfile_train.txt```\
```SavedParameters.pth```\
And a pair of images showing contact maps and distance maps after every two iterations, these files are named as follows\
```Autoencoder_contact<epoch number>.png``` and ```Autoencoder_distance<epoch number>.png```

