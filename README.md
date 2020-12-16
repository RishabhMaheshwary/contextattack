### A context aware approach for generating natural language attack.

This repository contains source code for the research work described in our AAAI 2021 paper:

[A context aware approach for generating natural language attack](https://www.researchgate.net/publication/347304665_A_Context_Aware_Approach_for_Generating_Natural_Language_Attacks)

Requirements
-  Pytorch >= 0.4
-  Tensorflow >= 1.0
-  Numpy
-  Python >= 3.6
- Tensorflow 2.1.0
- TensorflowHub

Dependencies

- Download counter-fitted-vectors from [here](https://github.com/nmrksic/counter-fitting/tree/master/word_vectors), unzip it and place the txt file in the main directory.
- Download pretrained BERT for each dataset [here](https://drive.google.com/file/d/1UChkyjrSJAVBpb3DcPwDhZUE4FuL0J25/view?usp=sharing), unzip it and place all the folders it in BERT/results/ directory.
- Download top 50 synonym file from [here](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view?usp=sharing), unzip it and place the txt file in the main directory.
 
How to Run:
-   Run the following command to get classification results. 

```
bash bert.sh
```
-   Run the following command to get inference results. 

```
python run_attack_nli.py
```
The results will be available in **results** directory.

