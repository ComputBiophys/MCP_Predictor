# Membrane contact probability (MCP) prediction
## Introduction

The membrane contact probability (MCP), a characteristic and predictive quantity to describe the likelihood of the amino acids of a given sequence being in direct contact with the acyl chains of lipid molecules. And it can be predicted with a combination of deep convolutional and recurrent neural network (DCRNN) by utilizing a training dataset extracted from MemProtMD, a database generated from molecular dynamics simulations.

The MCP is complementary to solvent accessibility in characterizing the outer surface of membrane proteins, and can be used to systematically improve the precision of the ResNet-based contact map predictor.

This package provides an implementation of the MCP and MCP-incorporated contact map prediction. 

## Usage
### Requirements
1. Tensorflow 1.7.0
2. Python packages: numpy, pickle

`pip install -r requirements.txt`

### Feature generation
*Please refer to the files in the folder of example*
#### Multiple sequence alignment (MSA) generation (https://github.com/soedinglab/hh-suite) (.a3m file)
#### Input features generation
1. Input features from RaptorX-Property 

PSSM, SS3, ACC (https://github.com/realbigws/RaptorX_Property_Fast, https://github.com/lacus2009/RaptorX-Angle)  (.feat file)

2. Additional input features for contact map (CM) prediction

CCMpred (https://github.com/soedinglab/CCMpred) (.mat file)

Pairwise potential and mutual information (https://github.com/multicom-toolbox/DNCON2/) (.pai file)

### Command
#### Membrane contact probability (MCP) prediction
`python data2pkl_MCP.py 5aymA`

`python Predict_MCP.py 5aymA`

#### MCP-incorporated ResNet-based contact map (CM) prediction
`python data2pkl_CM.py 5aymA`

`python MCP_add.py 5aymA`

`python Predict_CM.py 5aymA`


## Server
Please try to use our server of MCP predictor and MCP-incorporated CM predictor at:

http://www.songlab.cn/

## References
Wang, L.; Zhang, J.; Wang, D.; Song, C.* Membrane Contact Probability: An Essential and Predictive Character for the Structural and Functional Studies of Membrane Proteins. PLoS Comput. Biol. 2022, 18, e1009972.

The implementation is based on the projects:

[1] https://github.com/soedinglab/hh-suite

[2] https://github.com/realbigws/RaptorX_Property_Fast

[3] https://github.com/lacus2009/RaptorX-Angle

[4] https://github.com/soedinglab/CCMpred

[5] https://github.com/multicom-toolbox/DNCON2/
