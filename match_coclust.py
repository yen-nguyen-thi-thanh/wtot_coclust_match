#####========================#####
#####        WTOT-matching v1.0 #####
#####        WTOT-coclust v1.0       #####
#####======================#####
# Application: This repository contains python and R codes to run the algorithms WTOT-matching and WTOT-coclust, as presented in the paper Optimal transport-based machine learning to match specific patterns: application to the detection of molecular regulation patterns in omics data by T. T. Y. Nguyen, W. Harchaoui, L. Mégret, C. Mendoza, O. Bouaziz, C. Neri, A. Chambaz (2024). The paper can be found here.
# The aim of WTOT-matching and WTOT-coclust is to learn a pattern of correspondence between two datasets in situations where it is desirable to match elements that exhibit an affine relationship (our approach accommodates any relationship, not necessarily affine, as long as it can be parametrized). In the motivating case-study, the challenge is to better understand micro-RNA regulation in Huntington's disease model mice.
# The algorithms unfold in two stages. During the first stage, an optimal transport plan P and an optimal affine transformation are learned, using the Sinkhorn algorithm and a mini-batch gradient descent. During the second stage, P is exploited to derive either several co-clusters (WTOT-coclust) or several sets of matched elements (WTOT-matching).

# The Jupyter notebook `WTOT_MC_demo.ipynb` presents several illustrations. 

# The main files of the repository are:
# - `utils.py`: defines key-functions used during the first stage of the algorithms to compute the optimal transport matrix *P*, kernel, mapping, the squared Euclidean distance and the best number of coclusters;
# - `wtot.py`: it is the core code implementing the first stage of the algorithms;
# - `match_coclust.py`: it is the core code of the second stage of the algorithms. 

# The folder `simulations` contains the codes used to generate data for the experimantal study presented in the paper. The folder `datasets` contains the miRNA and mRNA data obtained in the striatum and cortex of the HD model mice; and the results obtained by running the WTOT-matching and WTOT-coclust. The file `sample_A4.npz` is a synthetic dataset generated in configuration A4 of the simulation study (see Section 5 of the paper). 

#
# Version: WTOT-matching v1.0 ; WTOT-coclust v1.0
# Date: 15 April 2020
#
# Contributors (alphabetic order): O. Bouaziz (1), A. Chambaz (1), W. Harchaoui (1), L. Mégret (2), C. Mendoza (2), C. Neri (2), T. T. Y. Nguyen (1) #
# Laboratory:
#   (1) MAP5, F-75006 Paris, France
#   (2) UMR CNRS 8256, Team Brain-C Lab, F-75005 Paris, France
#
# Affiliations:
#   (1) Université Paris Cité, CNRS
#   (2) Sorbonne Université, CNRS
#
#####=================================================================#####
#####       Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International license          #####
#####=============================================================#####
#####               Copyright (C) T. T. Y. Nguyen, W. Harchaoui, L. Mégret, C. Mendoza, O. Bouaziz, C. Neri, A. Chambaz (1) #####
#####                       Christian Neri(christian.neri@inserm.fr) Antoine Chambaz(antoine.chambaz@u-paris.fr) 2024                               #####
#####========================#####
#      
#      This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 
#      International License. To view a copy of this license, visit 
#      http://creativecommons.org/licenses/by-nc-nd/4.0/ or send a letter to 
#      Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#      
#####======================#####
#####=====================#####


import numpy as np
from sklearn.cluster.bicluster import SpectralCoclustering
from coclust.evaluation.external import accuracy
from utils import best_modularity_partition_update as best_nb_cluster
import warnings
warnings.filterwarnings('ignore')
def matching(P, labels_x=list(), labels_y=list(), k=20, k_prime=20, quantile=0.5):
    """
    Given a M\times N matrix, 
    outputs the collections {N_m : m \in [[M]]} and {M_n : n \in [[N]] and the rowwise and columnwise averages \tilde{k}_r, \tilde{k}_c of the cardinalities of the sets N_m and M_n.
 
    If given labels_x and labels_y, 
    also outputs the matching criterion including the average of the m-specific precision, sensitivity and specificity.
    """
    # Setting parameters
    ind_argsort_along_rows = np.argpartition(P, -k, axis = 1)[:, -k:]
    ind_argsort_along_cols = np.transpose(np.argpartition(P, -k_prime, axis=0)[-k_prime:, :])

    M, N = P.shape
    threshold = np.quantile(P, quantile)

    # Finding the k largest values in each row, yielding calN
    calN = {}
    for counter, value in enumerate(ind_argsort_along_rows):
        lt = []
        for ii in value: 
            if P[counter, ii] > threshold:
                lt.append(ii)
        if len(lt) != 0:
            calN[counter] = lt
            
    # Finding the k_prime largest values in each column, yielding calM  
    calM = {}
    for counter, value in enumerate(ind_argsort_along_cols):
        lt = []
        for ii in value: 
            if P[ii, counter] > threshold:
                lt.append(ii)
        if len(lt) != 0:
            calM[counter] = lt

    # "Inverting" calM
    inverted_calM = {}
    for key, values in calM.items():
        for value in values:
            inverted_calM.setdefault(value, []).append(key)
    len(inverted_calM)
    keys = set(np.sort(list(inverted_calM.keys())))

    # Identifying the sets N_m of the elements matched to each row
    N_m = {}
    for i in range(M):
        N_m[i] = set()
    for i in keys: 
        ii = set(inverted_calM[i]) & set(calN[i])
        N_m[i] = N_m[i].union(ii)
        
    # Identifying the sets M_n of the elements matched to each column   
    M_n = {}
    for key, values in N_m.items():
        for value in values:
            M_n.setdefault(value, []).append(key)
   
    
    # Computing the \tilde{k}_r and \tilde{k}_c numbers
    card = [len(N_m[i]) for i in range(M)]
    card_np = np.array(card)  
    k_tilde_r = sum(card_np)/sum(card_np != 0)
    
    
    card = [len(M_n[i]) for i in M_n.keys()]
    card_np = np.array(card)  
    k_tilde_c = sum(card_np)/sum(card_np != 0)
    
    # Computing the m-specific precision, sensitivity and specificity -- if labels of rows and columns are known
    if (len(labels_x) == 0 ):
        results = {'N_m': N_m, 'M_n': M_n, 'k_tilde_r':k_tilde_r, 'k_tilde_c':k_tilde_c}
    else:
        value_x = max(set(labels_x))
        value_y = max(set(labels_y))
        diff = value_x - value_y # Checking if irrelevant elements exist


        # Identifying the set of  *true* elements associated to each row     
        N_m_true = {}
        if diff != 1:
            for i in set(labels_x):
                set_x = set(np.where(labels_x == i)[0]) # the subset of x with label i
                set_y = set(np.where(labels_y == i)[0]) # the subset of x with label i
                for ii in set_x: 
                    N_m_true[ii] = set_y # Assigning the y subset with label i to elements in x subset with same labels i  
        else:             
            for i in set(labels_x):
                if i != (value_x):
                    set_x = set(np.where(labels_x == i)[0]) # the subset of x of elements with label i 
                    set_y = set(np.where(labels_y == i)[0]) # the subset of x of elements with label i
                    for ii in set_x: 
                        N_m_true[ii] = set_y  
                else:                     
                    set_x = set(np.where(labels_x == i)[0]) # the subset of x of elements with label i
                    set_y = set(np.where(labels_y == i)[0]) # the subset of x of elements with label i
                    for ii in set_x:
                        N_m_true[ii] = set() # Assigning the empty set to the irrelevant elements

        # Computing the precision, sensitivity and specificity
        precision = []
        sensitivity = []
        specificity = []
        for i in np.arange(M):

            FN = N_m_true[i] - N_m[i] # false negative set
            TP = N_m_true[i] & N_m[i] # true positive set
            FP = N_m[i] - N_m_true[i] # false positive set
            TN = set(np.arange(N)) -(FN&TP&FN) # true negative set
            
            # Computing m-specific precision
            if (len(TP)+len(FP)!= 0):
                pr = len(TP)/(len(TP)+len(FP))
            else: 
                pr = np.nan
            # Computing m-specific sensitivity
            if (len(FN)+len(TP)!=0):
                Sen = len(TP)/(len(FN)+len(TP))
            else: 
                Sen = np.nan

            # Computing m-specific specificity
            if (len(N_m_true[i]) !=0):
                SPF = len(TN)/(len(FP)+len(TN))
            else: 
                SPF = np.nan

            precision.append(pr)
            sensitivity.append(Sen)
            specificity.append(SPF)

            results = {'N_m': N_m, 'M_n': M_n, 'k_tilde_r': k_tilde_r, 'k_tilde_c': k_tilde_c, 'pre': np.nanmean(precision), 'sen': np.nanmean(sensitivity), 'spf' : np.nanmean(specificity) }
        
    return results
        


def SCC1_star(data, nb_cl, labels_x = list(), labels_y = list()):
    """
    Given a matrix and a true number of clusters, 
    outputs *bona filde* co-clusters by applying spectral co-clustering.

    If given the labels of row and columns, 
    also outputs the measure of discrepancy to assess a co-clustering error.
    """
    # Applying spectral co-clustering
    P = data
    model = SpectralCoclustering(n_clusters = nb_cl)
    model.fit(P)
    fit_P = P[np.argsort(model.row_labels_)]
    fit_P = fit_P[:, np.argsort(model.column_labels_)]

    
    column_spec = model.column_labels_  # the calculated column labels 
    row_spec = model.row_labels_        # the calculated row labels 

    if (len(labels_x) == 0 ):
        results = {'row_spec': row_spec, 'column_spec': column_spec}   
    else:
        # Computing the measure of discrepancy -- if labels of rows and columns are known
        e1 = 1 - accuracy(row_spec, labels_x)
        e2 = 1 - accuracy(column_spec, labels_y)
        SCC1_star  = e1+e2-e1*e2 

        results = {'SCC1_star': SCC1_star,'row_spec': row_spec, 'column_spec': column_spec}


    return results


def SCC1(data, labels_x=list(), labels_y=list()):
    """
    Given a matrix, 
    outputs *bona filde* co-clusters by applying spectral co-clustering required an argument of number of clusters,
    that is learnt by using a criterion involving graph modularity whose code is available on "Coclust" package.
    
    If given the labels of row and columns, 
    also outputs the measure of discrepancy to assess a co-clustering error.
    """
    # Learning a relevant number of co-clusters
    P = data
    u = best_nb_cluster(P, nbr_clusters_range = range(2, 20), n_rand_init=1)
    nb_cl = u[0].n_clusters

    # Applying spectral co-clustering
    model = SpectralCoclustering(n_clusters = nb_cl)
    model.fit(P)  
    fit_P = P[np.argsort(model.row_labels_)] 
    fit_P = fit_P[:, np.argsort(model.column_labels_)]


    column_spec = model.column_labels_ # the calculated column labels
    row_spec = model.row_labels_       # the calculated row labels
    
    if (len(labels_x) == 0 ):
        results = {'row_spec': row_spec, 'column_spec': column_spec}   
    else:
        # Computing the measure of discrepancy -- if labels of rows and columns are known
        e1 = 1 - accuracy(row_spec, labels_x)
        e2 = 1 - accuracy(column_spec, labels_y)
        SCC1  = e1+e2-e1*e2
        
        results = {'SCC1_eval': SCC1, 'row_spec': row_spec, 'column_spec': column_spec}
    
    return results

def SCC2_star(data, nb_cl, labels_x = list(), labels_y = list() ): 
    """
    Given a matrix and a true number of clusters, 
    outputs *bona filde* co-clusters by removing the irrelevant elements then applying spectral co-clustering.

    If given the labels of row and columns, 
    also outputs the measure of discrepancy to assess a co-clustering error.
    """
    # Learning a relevant number of co-clusters
    P = data
    u = best_nb_cluster(P, nbr_clusters_range = range(2, 20), n_rand_init=1)
    nb_cl_1 = u[0].n_clusters

    # Applying spectral co-clustering
    model = SpectralCoclustering(n_clusters = nb_cl_1)
    model.fit(P)
    fit_P = P[np.argsort(model.row_labels_)]
    fit_P = fit_P[:, np.argsort(model.column_labels_)]


    ind_x = np.array([], dtype = 'int32')
    ind_y = np.array([], dtype = 'int32')

    # Detecting and removing the irrelevant rows and columns
    for i in range(nb_cl_1):
        if (np.std(model.get_submatrix(i, P )) < np.std(P)):
            ind_x0 = model.get_indices(i)[0]
            ind_y0 = model.get_indices(i)[1]
            ind_x = np.concatenate((ind_x, ind_x0)) # Keeping the relevant rows
            ind_y = np.concatenate((ind_y, ind_y0)) # Keeping the relevant columns

    ind_x = np.sort(ind_x)
    ind_y = np.sort(ind_y)


    P_tilde = P[ind_x][:, ind_y] # the remaining matrix \tilde{P}
    shape = np.shape(P_tilde)
    if shape[0] < 20 or shape[1] < 20:
        SCC2_star = None
    else:
        # Applying spectral co-clustering
        model_m = SpectralCoclustering(n_clusters = nb_cl)
        model_m.fit(P_tilde)
        fit_P_tilde = P_tilde[np.argsort(model_m.row_labels_)]
        fit_P_tilde = fit_P_tilde[:, np.argsort(model_m.column_labels_)]

        row_result = np.zeros(P.shape[0])
        column_result = np.zeros(P.shape[1])

        column_result[ind_y] = model_m.column_labels_+1
        row_result[ind_x] = model_m.row_labels_ +1

        # Computing the measure of discrepancy -- if labels of rows and columns are known
        if (len(labels_x) == 0 ):
            results = {'row_spec': row_result, 'column_spec': column_result}   
        else:
            e1 = 1 - accuracy(row_result, labels_x)
            e2 = 1 - accuracy(column_result, labels_y)
            SCC2_star  = e1+e2-e1*e2
            results = {'SCC2_star_eval': SCC2_star, 'row_spec': row_result, 'column_spec': column_result}        

        return results
        
def SCC2(data, labels_x = list(), labels_y = list()): 
    """
    Given a matrix P,
    outputs *bona filde* co-clusters by applying spectral co-clustering requiring an argument of number of co-clusters,
    that is learnt by using a criterion involing graph modularity whose code is available on "Coclust" package.

    If given the labels of row and columns, 
    also outputs the measure of discrepancy to assess a co-clustering error.
    """
    # Learning a relevant number of co-clusters
    P = data
    u = best_nb_cluster(P, nbr_clusters_range = range(2, 20), n_rand_init=1)
    nb_cl_1 = u[0].n_clusters

    # Applying spectral co-clustering
    model = SpectralCoclustering(n_clusters = nb_cl_1)
    model.fit(P)
    fit_P = P[np.argsort(model.row_labels_)]
    fit_P = fit_P[:, np.argsort(model.column_labels_)]

    ind_x =np.array([], dtype = 'int32')
    ind_y = np.array([], dtype = 'int32')

    # Detecting and removing the irrelevant rows and columns
    for i in range(nb_cl_1):
        if (np.std(model.get_submatrix(i, P )) < np.std(P)):
            ind_x0 = model.get_indices(i)[0]
            ind_y0 = model.get_indices(i)[1]
            ind_x = np.concatenate((ind_x, ind_x0)) # Keeping the relevant rows
            ind_y = np.concatenate((ind_y, ind_y0)) # Keeping the relevant columns

    ind_x = np.sort(ind_x)
    ind_y = np.sort(ind_y)



    P_tilde = P[ind_x][:, ind_y] # the remaining matrix \tilde{P}
    shape = np.shape(P_tilde)
    if shape[0] < 20 or shape[1] < 20:
        SCC2 = None
    else:
        # Leaning a relevant number of co-clusters
        u_m = best_nb_cluster(P_tilde, nbr_clusters_range = range(2, 20), n_rand_init=1)
        nb_sp_2 = u_m[0].n_clusters

        # Applying spectral co-clustering   
        model_m = SpectralCoclustering(n_clusters = nb_sp_2)
        model_m.fit(P_tilde)     
        fit_P_tilde = P_tilde[np.argsort(model_m.row_labels_)]
        fit_P_tilde = fit_P_tilde[:, np.argsort(model_m.column_labels_)]

        row_result = np.zeros(P.shape[0])
        column_result = np.zeros(P.shape[1])

        column_result[ind_y] = model_m.column_labels_+1 # the calculated column labels
        row_result[ind_x] = model_m.row_labels_ +1 # the calculated row labels
        if (len(labels_x) == 0 ):
            results = {'row_spec': row_result, 'column_spec': column_result}   
        else:
            # Computing the measure of discrepancy -- if labels of rows and columns are known
            e1 = 1 - accuracy(row_result, labels_x)
            e2 = 1 - accuracy(column_result, labels_y)
            SCC2  = e1+e2-e1*e2
            results = {'SCC2_eval': SCC2, 'row_spec': row_result, 'column_spec': column_result}

    return results







