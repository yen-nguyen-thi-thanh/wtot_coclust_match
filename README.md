Welcome! This repository contains `python` and `R` codes to run the algorithms WTOT-matching and WTOT-coclust, as presented in the paper *Optimal transport-based machine learning to match specific expression patterns in omics data* by T. T. Y. Nguyen, A. Chambaz, O. Bouaziz, W. Harchaoui and C. Neri (2021). The paper can be found [here](https://hal.archives-ouvertes.fr/hal-03293786/). 

The aim of WTOT-matching and WTOT-coclust is to learn a pattern of correspondence between two datasets in situations where it is desirable to match elements that exhibit an affine relationship (our approach accommodates *any* relationship, not necessarily affine, as long as it can be parametrized). In the motivating case-study, the challenge is to better understand micro-RNA regulation in Huntington's disease model mice. 

The algorithms unfold in two stages. During the first stage, an optimal transport plan *P* and an optimal affine transformation are learned, using the Sinkhorn algorithm and a mini-batch gradient descent. During the second stage, *P* is exploited to derive either several co-clusters (WTOT-coclust) or several sets of matched elements (WTOT-matching).

The Jupyter notebook `WTOT_MC_demo.ipynb` presents several illustrations. 

The main files of the repository are:
- `utils.py`: defines key-functions used during the first stage of the algorithms to compute the optimal transport matrix *P*, kernel, mapping, the squared Euclidean distance and the best number of coclusters;
- `wtot.py`: it is the core code implementing the first stage of the algorithms;
- `match_coclust.py`: it is the core code of the second stage of the algorithms. 

The folder `simulations` contains the codes used to generate data for the experimantal study presented in the paper. The folder `datasets` contains the miRNA and mRNA data obtained in the striatum and cortex of the HD model mice; and the results obtained by running the WTOT-matching and WTOT-coclust. The file `sample_A4.npz` is a synthetic dataset generated in configuration A4 of the simulation study (see Section 5 of the paper). 





