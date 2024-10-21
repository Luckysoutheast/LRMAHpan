# LRMAHpan: *Peptide-HLA presentation Modeling*
This repository contains the codes for the paper [LRMAHpan: a novel tool for multi-allelic HLA presentation prediction using Resnet-based and LSTM-based neural network]


LRMAHpan (Peptide-HLA Presentation  Modeling) is a deep learning-based model to predict the Peptide-HLA presentation, including two submodels LRMAHpan AP (Antigen Processing) and LRMAHpan BA (Binding Affinity). 

AP Model only takes the primary sequences of  peptides as input. BA Model integrates both peptide sequences and 6 HLA types as input.

## Dependency
1. Install basic packages using:
    ```bash
    # [Optional] Create a new environment and activate it
	conda create -n LRMAHpan python=3.7.13
	conda activate LRMAHpan

    # Install Pytorch packages (for CUDA 11.3)
	conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch
 	conda install cudatoolkit=11.3.1 -c conda-forge 
    # Install other packages
	pip install torchtext==0.11.0
	pip install spacy==3.4.1
	pip install pandas==1.3.5
	pip install scikit-learn==1.0.2

    ```
    **Note**: Change the Pytorch version to be compatible with your CUDA version.


## Inference
### Predict
1. Put your input Peptide-HLA sequence pairs in the `uploaded/multiple_query.csv` file. The peptide are represented by their sequences in the following format:
    | peptide    |      A1     |      A2     |      B1     |      B2     |      C1     |      C2     | 
    | -----------| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |       
    | KLEDLERDL  | HLA-A*01:01 | HLA-A*02:01 | HLA-B*01:01 | HLA-B*37:01 | HLA-C*05:02 | HLA-A*07:01 |
    | AVLEOSGFRK | HLA-A*01:01 | HLA-A*02:01 | HLA-B*01:01 | HLA-B*37:01 | HLA-C*05:02 | HLA-A*07:01 |

**Note**:If only one HLA type（such as HLA-A*02:01） is provided, the input data format is as follows:
    | peptide    |      A1     |      A2     |      B1     |      B2     |      C1     |      C2     | 
    | -----------| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |       
    | KLEDLERDL  | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 |
    | AVLEOSGFRK | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | HLA-A*02:01 | 


2. Run 
    ```
    python predict_res.py -t uploaded/multiple_query.csv -o ./download/result_multipe.csv
    ```
3. The predicted peptide-HLA presentation scores are in the `/download/result_multipe.csv`. The `PS` column in the file represent the predicted sequence-level presentation scores (probabilities) of the peptide-HLA pair.




## Contact
If you have any questions, please contact us at 230218235@seu.edu.cn
