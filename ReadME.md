# GGNN
[A Gemetric informed Graph neural network based model for Cancer survival prediction](https://doi.org/10.1016/j.compbiomed.2023.107117)

## Requirements
* pytorch
* networkx
* numpy
* pandas
* POT
* lifelines
* matplotlib

## Use GGNN
1. Download multi-omics data and biological network from 
* https://themmrf.org/finding-a-cure/our-work/the-mmrf-commpass-study/
* https://www.cbioportal.org/
* https://www.hprd.org/
* https://www.genome.jp/kegg/

2. Run preprocessing: Preprocessing.R
* Input multi-omics data is filtered by available genes intersect with HPRD gene set and KEGG gene set
* Output RNA.csv, CNA.csv, Methy.csv and clinn.csv in a folder "out/" in the original data folder

3. Compute Ollivier-Ricci curvature: Compute_ORC.py
* Use csv files generated from last step
* Curvature results are saved in "out/" as RNA_curv.csv, CNA_curv.csv and Methyl_curv.csv

4. Run GGNN model: Test_multi_curv_net_surv.py
* A processed data set of TCGA study of LGG with HPRD network and KEGG pathway can be download via the link bellow: 
* [https://stonybrookmedicine.box.com/s/7t8qbrmni7n16lcfu3kmlnwje9l6qgdy](https://stonybrookmedicine.box.com/s/7t8qbrmni7n16lcfu3kmlnwje9l6qgdy)
