## This work is licensed under a Creative
## Commons Attribution-NonCommercial-NoDerivs 4.0 International
## (http://creativecommons.org/licenses/by-nc-nd/4.0/) 

# The functions "sinkhorn" , "sinkhorn1", "sinkhorn2" and "_squared_distances"
# build upon Aude Genevay's implementation of a Sinkhorn-GAN 
# (https://github.com/audeg/Sinkhorn-GAN/blob/master/sinkhorn.py).

# The function "best_modularity_partition_update" is a slightly modified version of the function
# "best_modularity_partition" from the "Coclust" package.
#(https://coclust.readthedocs.io/en/v0.2.1/_modules/coclust/evaluation/internal.html#best_modularity_partition).

import numpy as np
from numpy.random import multivariate_normal as mvn
import torch
from torch.autograd import Variable
from scipy.stats import multivariate_normal
from coclust.coclustering import CoclustMod

use_cuda = torch.cuda.is_available()
device = "cpu" if not(use_cuda) else 'cuda'
device = torch.device(device)

# setting how numbers are represented
torch_real_type = torch.float # decimal




def sinkhorn(x, y, epsilon, niter, device):
    """
    Given two empirical measures attached to x and y
    outputs an approximation of the OT cost with regularization parameter epsilon.
    Argument niter is the maximum number of steps in the Sinkhorn loop.
    """
    # The Sinkhorn algorithm takes as input three variables:
    C = _squared_distances(x, y) # Wasserstein cost function
    # print(C.shape)

    mu = Variable(torch.FloatTensor(x.shape[0]).fill_(1), requires_grad=False).to(device)
    mu /= mu.shape[0]
    nu = Variable(torch.FloatTensor(y.shape[0]).fill_(1), requires_grad=False).to(device)
    nu /= nu.shape[0]
        
    def M(u,v): 
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    lse = lambda A: torch.logsumexp(A, dim=1, keepdim=True)
    
    # Actual Sinkhorn loop ......................................................................
    u,v,err = 0.*mu, 0.*nu, 0.
    actual_nits = 0
    
    for i in range(niter):
        u0 = u # useful to check the update
        u =  epsilon * (torch.log(mu) - lse(M(u,v)).squeeze()) + u
        v =  epsilon * (torch.log(nu) - lse(M(u,v).t()).squeeze()) + v 
        err = (u - u0).abs().sum()
        if (err < 1e-4).data.cpu().numpy():
             break

    U, V = u, v 
    Gamma = torch.exp(M(U,V))            # eventual transport plan, g = diag(a)*K*diag(b)
    cost  = torch.sum(Gamma * C)         # simplistic cost, chosen for readability
    
    return cost, Gamma

def sinkhorn1(x, y, w, epsilon, niter, device) :
    """
    Given the empirical measure attached to x and the w-weighted empirical measure attached to y
    outputs an approximation of the OT cost with regularization parameter epsilon.
    Argument niter is the maximal number of steps in the Sinkhorn loop.
    """
    # The Sinkhorn algorithm takes as input three variables:
    C = _squared_distances(x, y) # Wasserstein cost function
    # print(C.shape)

    mu = Variable(torch.FloatTensor(x.shape[0]).fill_(1), requires_grad = False).to(device)
    mu /= mu.shape[0]
    # nu = Variable(torch.FloatTensor(w),requires_grad = False).to(device)
    nu = Variable(torch.FloatTensor(y.shape[0]).fill_(1), requires_grad = False).to(device)
    nu /= nu.shape[0]
        
    def M(u,v): 
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    lse = lambda A: torch.logsumexp(A, dim=1, keepdim=True)
    
    # Actual Sinkhorn loop ......................................................................
    u,v,err = 0.*mu, 0.*nu, 0.
    actual_nits = 0
    
    for i in range(niter) :
        u0 = u # useful to check the update
        u =  epsilon * (torch.log(mu) - lse(M(u,v)).squeeze()) + u
        v =  epsilon * (torch.log(w) - lse(M(u,v).t()).squeeze()) + v 
        err = (u - u0).abs().sum()
        if (err < 1e-4).data.cpu().numpy() :
             break

    U, V = u, v 
    Gamma = torch.exp( M(U,V) )            # eventual transport plan, g = diag(a)*K*diag(b)
    cost  = torch.sum( Gamma * C )         # simplistic cost, chosen for readability 
    
    return cost, Gamma




def sinkhorn2(x, y, w1, w2, epsilon, niter, device) :
    """
    Given w1- and w2-weighted empirical measure attached to x and y, respectively
    outputs an approximation of the OT cost with regularization parameter epsilon.
    Argument niter is the maximal number of steps in the Sinkhorn loop.
    """
    # The Sinkhorn algorithm takes as input three variables:
    C = _squared_distances(x, y) # Wasserstein cost function
    # print(C.shape)

    mu = Variable(torch.FloatTensor(x.shape[0]).fill_(1), requires_grad = False).to(device)
    mu /= mu.shape[0]
    # nu = Variable(torch.FloatTensor(w),requires_grad = False).to(device)
    nu = Variable(torch.FloatTensor(y.shape[0]).fill_(1), requires_grad = False).to(device)
    nu /= nu.shape[0]
        
    def M(u,v): 
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    lse = lambda A: torch.logsumexp(A, dim=1, keepdim=True )
    
    # Actual Sinkhorn loop ......................................................................
    u,v,err = 0.*mu, 0.*nu, 0.
    actual_nits = 0
    
    for i in range(niter) :
        u0 = u # useful to check the update
        u =  epsilon * (torch.log(w1) - lse(M(u,v)).squeeze()) + u
        v =  epsilon * (torch.log(w2) - lse(M(u,v).t()).squeeze()) + v 
        err = (u - u0).abs().sum()
        if (err < 1e-4).data.cpu().numpy() :
             break

    U, V = u, v 
    Gamma = torch.exp( M(U,V) )            # eventual transport plan, g = diag(a)*K*diag(b)
    cost  = torch.sum( Gamma * C )         # simplistic cost, chosen for readability 
    
    return cost, Gamma

def _squared_distances(x, y) :
    "Returns the matrix of $\|x_i-y_j\|^2$."
    x_col = x.unsqueeze(1) # x.dimshuffle(0, 'x', 1)
    y_lin = y.unsqueeze(0) # y.dimshuffle('x', 0, 1)
    c = torch.sum( torch.abs(x_col - y_lin) , 2)
    return c 


   












def weight(x, y, epsilon, d) :
    
    """
    Given two data sets x = {x_1,..., x_M} and y = {y_1, ..., y_N} subsets of R^d,
    outputs the weight vector w = (w_1, ..., w_N) on the data set y 
    characterized by: w_n proportional to  \sum_{m \in [[M]]} \varphi((y_n - x_m)/h)
    where \varphi is the standard normal density and h is the arithmetic
    mean of the pair distances of all elements in x.
    """
        
    x_np = x.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    
    x_np = np.expand_dims(x_np, axis=0)
    y_np = np.expand_dims(y_np, axis=1)
    
    f = (y_np - x_np)/epsilon
    rv = multivariate_normal(np.zeros(d), np.identity(d))    
    w_np = np.sum(rv.pdf(f), axis=1)
    w_np = w_np/np.sum(w_np)
    
    w = torch.from_numpy(w_np).type(torch_real_type).to(device)
    return w






def trans_simple(X, tau_1, theta_2, constr1=torch.tensor([-5., 0.])):
    """
    Given a vector x  in R^d,
    outputs \theta(x) where \theta: u -> \theta(u) = \theta_1 \odot u + \theta_2
    (\odot is the componentwise multiplication).
    """
    
    theta_1 = constr1[0] + (constr1[1] - constr1[0]) * torch.sigmoid(tau_1)
    S = X * theta_1 
    
    return S + theta2




def trans_gen(X, tau_1a, tau_1b, tau_1c, theta_2,  m , n, constr1=torch.tensor([-5., 0.]),
          constr2=torch.tensor([-0.5, 0.5]) , constr3=torch.tensor([-0.5, 0.5])):
    """
    Given a vector x in R^d, 
    outputs \theta(x) where \theta: u -> \theta_1 u + \theta_2.
    See Appendix A of 'Optimal transport-based ML to match specific expression patterns in omics data' for details...
    """    
        
    N_x = X.shape[0]
    X_tilde = X.reshape(X.shape[0], m, n)
    
    theta_1a = constr1[0] + (constr1[1] - constr1[0]) * torch.sigmoid(tau_1a)
    theta_tilde_1a = theta_1a.reshape(m, n)
    S = X_tilde * theta_tilde_1a
    
    theta_1b = constr2[0] + (constr2[1] - constr2[0]) * torch.sigmoid(tau_1b)
    theta_tilde_1b = theta_1b.reshape(m-1, n)
    S[:, 1:m, :] = S[:, 1:m, :] + theta_tilde_1b * X_tilde[:, 0:(m-1), :]
    
    theta_1c = constr3[0] + (constr3[1] - constr3[0]) * torch.sigmoid(tau_1c)
    theta_tilde_1c = theta_1c.reshape(m, n-1)
    S[:, :, 1:n] = S[:, :, 1:n] + theta_tilde_1c*X_tilde[:, :, :n-1]
    return S.reshape(N_x, m*n) + theta_2



def best_modularity_partition_update(in_data, nbr_clusters_range, n_rand_init=1):
    """
    Evaluates the best partition over a range of number of clusters
    using co-clustering by direct maximization of graph modularity.

    Parameters
    ----------
    in_data : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
        Matrix to be analyzed
    nbr_clusters_range :
        Number of clusters to be evaluated
    n_rand_init:
        Number of time the algorithm will be run with different initializations

    Returns
    -------
    tmp_best_model: :class:`coclust.coclustering.CoclustMod`
        model with highest final modularity
    tmp_max_modularities: list
        final modularities for all evaluated partitions
    """

    tmp_best_model = None
    tmp_max_modularities = [np.nan] * len(nbr_clusters_range)
    eps_best_model = 1e-4

    # Set best final modularity to -inf
    modularity_begin = float("-inf")

    for tmp_n_clusters in nbr_clusters_range:

        # Creating and fitting a model with tmp_n_clusters co-clusters
        tmp_model = CoclustMod(n_clusters=tmp_n_clusters, n_init=n_rand_init,
                               random_state=0)
        tmp_model.fit(in_data)

        modularity_end = tmp_model.modularity
        # Checking if the final modularity is better with tolerance
        if((modularity_end - modularity_begin) > eps_best_model):
            tmp_best_model = tmp_model
            modularity_begin = modularity_end

        tmp_max_modularities[(tmp_n_clusters)-min(nbr_clusters_range)] = tmp_model.modularity

    return (tmp_best_model, tmp_max_modularities)


