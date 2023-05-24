# GGNN
A Gemetric informed Graph neural network based model for Cancer survival prediction

## Requirements
* pytorch
* networkx
* numpy
* pandas
* POT
* lifelines
* matplotlib

## Use GGNN
1. Download multi-omics data from 
https://themmrf.org/finding-a-cure/our-work/the-mmrf-commpass-study/
https://www.cbioportal.org/

2. Run preprocessing: Preprocessing.R
* Input multi-omics data is filtered by available genes intersect with HPRD gene set and KEGG gene set
* Output RNA.csv, CNA.csv, Methy.csv and clinn.csv in a folder "out/" in the original data folder

3. Compute Ollivier-Ricci curvature: Compute_ORC.py
* Use csv files generated from last step
* Curvature results are saved in "out/" as RNA_curv.csv, CNA_curv.csv and Methyl_curv.csv

4. Run GGNN model: Test_multi_curv_net_surv.py