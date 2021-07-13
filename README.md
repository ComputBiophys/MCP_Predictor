# Membrane contact probability (MCP) prediction
## Introduction

One of the unique traits of membrane proteins is that a significant fraction of their hydrophobic amino acids is exposed to the hydrophobic core of lipid bilayers rather than being embedded in the protein interior, which is often not explicitly considered in the protein structure and function predictions. Here, we propose a characteristic and predictive quantity, the membrane contact probability (MCP), to describe the likelihood of the amino acids of a given sequence being in direct contact with the acyl chains of lipid molecules. We show that MCP is complementary to solvent accessibility in characterizing the outer surface of membrane proteins, and it can be predicted for any given sequence with a machine learning-based method by utilizing a training dataset extracted from MemProtMD, a database generated from molecular dynamics simulations for the membrane proteins with a known structure. As the first of many potential applications, we demonstrate that MCP can be used to systematically improve the prediction precision of the protein contact maps.

## Usage
### Requirements
1. Tensorflow 1.7.0
2. Python packages: numpy, pickle
`pip install -r requirements.txt`

### Command
#### Membrane contact probability (MCP) prediction
`python data2pkl_MCP.py 5aymA`
`python Predict_MCP.py 5aymA`

#### MCP-incorporated ResNet-based contact map (CM) prediction
`python data2pkl_CM.py 5aymA`
`python MCP_add.py 5aymA`
`python Predict_CM.py 5aymA`

## Server


##References
Wang, L.; Zhang, J.; Wang, D.; Song, C.* Lipid Contact Probability: An Essential and Predictive Character for the Structural and Functional Studies of Membrane Proteins. bioRxiv 2021, https://doi.org/10.1101/2021.01.17.426988
