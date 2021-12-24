### Simulate data in second simulation study, using determinantal gaussian distribution.


library(reticulate)
library(gtools)
np <- import("numpy")


library(spatstat)
library(MASS)
sample_data <-function(d = 2, cl = 3, N_x = 200, N_y = 200, var = 5e-4,  shuffle = TRUE, noise = FALSE){
  model <- dppGauss(lambda = cl, alpha = .05, d = d)
  nb_point = 0
  test = 1

    internal = 0
    while(  test !=0 ){
      internal = internal+1
      print('*********internal iteration******')
      print(internal)
      dat = try(simulate(model, 1))

      if (!grepl('Error', dat[[1]]))
      { 
        if(dat[["n"]] == cl){
          mean = cbind(dat$x, dat$y)
          
          test = sum(dist(mean)< 1e-3)
        } 


      }
      
    }
  
  proportion_x = rdirichlet(1, rep(7,cl))
  ind_x = sample(1:cl, size = N_x, replace = TRUE, prob = proportion_x)
  
  x = matrix(NA, ncol = d, nrow = 0)
  labels_x = c()
  for( i in (1:cl)){
    n_x = sum(ind_x == i)
    x_i = mvrnorm(n = n_x, mu = mean[i,], Sigma = var * diag(d))
    x = rbind(x, x_i)
    labels_x = c(labels_x, rep(i, n_x))
  }
  
  proportion_y = rdirichlet(1, rep(7, cl))
  ind_y = sample(1:(cl), size = N_y, replace = TRUE, prob = proportion_y)
  
  y = matrix(NA, ncol = d, nrow = 0)
  labels_y = c()
  for( i in (1:(cl))){
    n_y = sum(ind_y == i)
    y_i = mvrnorm(n = n_y, mu = - mean[i,], Sigma = var* diag(d))
    y = rbind(y, y_i)
    labels_y = c(labels_y, rep(i, n_y))
  }
  
  if (shuffle){
    id_x = sample(N_x)
    id_y = sample(N_y)
    x = x[id_x,]
    labels_x = labels_x[id_x]
    y = y[id_y,]
    labels_y = labels_y[id_y]
  }

  return(list(x = x, y = y, labels_x = labels_x, labels_y =labels_y, mean = mean, internal = internal))
}


#datas = list()
#for(i in 1:1){
  
 # data = sample_data(d = 2, cl = 6, N_x = 300, N_y = 300,  shuffle = TRUE, noise = FALSE)
  
 # datas[[i]] = data
#}

#np$savez_compressed('sample_A1', dats = datas)

