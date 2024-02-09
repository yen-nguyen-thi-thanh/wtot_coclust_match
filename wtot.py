## This work is licensed under a Creative
## Commons Attribution-NonCommercial-NoDerivs 4.0 International
## (http://creativecommons.org/licenses/by-nc-nd/4.0/) 

import numpy as np
from numpy import random as rd
from numpy.random import multivariate_normal as mvn
from scipy.spatial import distance_matrix
import torch
from torch.autograd import Variable
import torch.optim as optim
from coclust.evaluation.external import accuracy
from utils import weight, sinkhorn, sinkhorn1, sinkhorn2, trans_gen
from utils import _squared_distances as square_distance


max_iteration = 500 # maximum number of steps findind transformation \theta

# setting the parameters of stopping criterion
nb_compares = 20
threshold =  1e-20
factor = 2

def wtot(x, y, m, n, batch_size_x=64, batch_size_y=64, constr1=np.array([-1.5, -0.5]), constr2=np.array([-0.2, 0.2]), constr3=np.array([-0.2, 0.2])):
    """
    Given two datasets x and y,
    outputs the optimal transport(OT) matrix P along with the weight w and 
    the optimal transformation \theta      
    """
    # Setting parameters
    d = m*n 
    n_iter = 100 # fixed point iterations    
    eta = 0.95 # decay rate
    eps_0 = 3. # initial entropy regularization    
    distance_xx = distance_matrix(x, x) # the matrix of Euclidean distance
    eps_e = np.mean(distance_xx) # entropy regularization
 
    # Defineing how numbers are represented
    torch_real_type = torch.float # decimal
    torch_integer_type = torch.long # integer

    # Setting the device being GPU if available 
    use_cuda = torch.cuda.is_available() # checking if CUDA is currently available
    device = "cpu" if not(use_cuda) else 'cuda' 
    device = torch.device(device)

    # Drawing randomly the parameters of theta from uniform distribution
    tau_1a_np = np.random.rand(m*n) 
    tau_1b_np = np.random.rand((m-1)*n) 
    tau_1c_np = np.random.rand(m*(n-1))
    theta_2_np = np.random.uniform(-1, 1, d)

    # Converting numpy array to float torch tensor;
    # wrapping tensors with Variable with the argument *requires_grad=True* to allow calculate the gradients
    tau_1a = torch.nn.Parameter(Variable(torch.from_numpy(tau_1a_np).type(torch_real_type), requires_grad= True).to(device))
    tau_1b = torch.nn.Parameter(Variable(torch.from_numpy(tau_1b_np).type(torch_real_type), requires_grad= True).to(device))
    tau_1c = torch.nn.Parameter(Variable(torch.from_numpy(tau_1c_np).type(torch_real_type), requires_grad= True).to(device))
    theta_2 = torch.nn.Parameter(Variable(torch.from_numpy(theta_2_np).type(torch_real_type), requires_grad=True).to(device))
    
    # Converting numpy array to float torch tensors and using device (either GPU if available or CPU)
    constr1 = torch.from_numpy(constr1).type(torch_real_type).to(device)
    constr2 = torch.from_numpy(constr2).type(torch_real_type).to(device)
    constr3 = torch.from_numpy(constr3).type(torch_real_type).to(device)
    
    # Converting the datasets of form numpy array to float torch tensors
    x = torch.from_numpy(x).type(torch_real_type)
    y = torch.from_numpy(y).type(torch_real_type)
    
    
    params = [tau_1a, tau_1b, tau_1c, theta_2] # Declaring the variables
    optimizer = optim.RMSprop(params) # Implementing RMSprop algorithm, (or SGD, Adam algorithms)

    losses = [] 
    epsilon = eps_0 # Setting initial entropy regularization

    for iteration in range(max_iteration):
        
        
        optimizer.zero_grad() # Setting the gradients of all optimized variables to zero
        
        # Sampling a mini-batch of data set x of size batch_size_x, yield x_sub
        sub_sample_x = np.random.randint(x.size()[0], size = batch_size_x)
        x_sub = x[sub_sample_x]
        x_sub = x_sub.to(device)
        
        # Sampling a mini-batch of data set y of size batch_size_y, yield y_sub
        sub_sample_y = np.random.randint(y.size()[0], size=batch_size_y)
        y_sub = y[sub_sample_y]
        y_sub = y_sub.to(device)
        
        theta_y_sub = trans_gen(y_sub, tau_1a, tau_1b, tau_1c, theta_2, m, n, constr1=constr1,
              constr2=constr2, constr3=constr3) # Computing the image of y_sub (the subset of y)
        
        w = weight(x_sub, theta_y_sub, epsilon, d) # the weight on y_sub


        # divergence and transport
        s, pi = sinkhorn1(x_sub, theta_y_sub, w, epsilon, n_iter, device) # divergence

        sxx, _ = sinkhorn(x_sub, x_sub, epsilon, n_iter, device) # divergence

        syy, _ = sinkhorn2(theta_y_sub, theta_y_sub, w, w, epsilon, n_iter, device) # divergence

        loss = 2 * s - sxx - syy # distance (better than divergence)

        loss.backward() # Computing the gradient 
        optimizer.step() # Updating the parameters
        
        losses.append(loss.data.cpu().numpy())
        # Applying the stopping criterion
        if( iteration > 5 + nb_compares):
            test_one = all(np.abs(losses[-1-nb_compares:-1] - losses[-2-nb_compares]) <= threshold)
            test_two = np.max(losses[-1-nb_compares:-1]) - losses[-2-nb_compares] >= np.abs(losses[-2-nb_compares]) * factor - threshold
            if (test_one or test_two):
                break   
                
                
        epsilon = max(epsilon*eta, eps_e) # updating the entropy regularization
        #if iteration %10 == 0:

            #print(loss)

    x_device = x.to(device)
    y_device = y.to(device)
    
    g_y_device = trans_gen(y_device, tau_1a, tau_1b, tau_1c, theta_2, m, n, 
                 constr1=constr1, constr2=constr2, constr3=constr3) # computing the image {theta(y_1),...,theta(y_N)} of y 
    w_1 = weight(x_device, g_y_device, epsilon, d) # the weight on the data set y
    _, pi_1 = sinkhorn1(x_device, g_y_device, w_1, epsilon, n_iter, device) # computing the final OT matrix between x and y


    # Computing the parameters of optimal transformation 
    theta_1a = constr1[0] + (constr1[1] - constr1[0]) * torch.sigmoid(tau_1a)
    theta_1b = constr2[0] + (constr2[1] - constr2[0]) * torch.sigmoid(tau_1b)
    theta_1c = constr3[0] + (constr3[1] - constr3[0]) * torch.sigmoid(tau_1c)
    
    # Converting to numpy array on the CPU
    P = pi_1.cpu().detach().numpy()
    w_np = w_1.cpu().detach().numpy()
    theta  = [f_torch.data.cpu().detach().numpy() for f_torch in [theta_1a, theta_1b, theta_1c, theta_2]]


    result = {'P': P, 'w': w_np, 'theta': theta}

    return result

