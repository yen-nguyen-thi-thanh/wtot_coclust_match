### Simulate data in third simulation study, using real dataset.


library(blockcluster)
library(reticulate)
np <- import("numpy")
gz <- import("gzip")
library(MASS)
################

server = 1

simulate_data <- function(M,
                          lambda_micro = 50, lambda_mesg = 50,
                          sd_micro = 0.1, sd_mesg = 0.1,
                          sd_r_micro = 5, sd_r_mesg = 5,
                          r_micro =  30, r_mesg =  30,
                          real_data = TRUE,
                          file = NULL
                          
) {
  if (is.null(file)) {
    
    if (server){
      file<- file.path("/home/yent/all_algorithm/real_data", 
                       "LFC_MouseAllelicSeries_mirna_Cortex.txt")
      
    }else{
      file <- file.path("//Users/thithanhyen/AResearch/POT-master/Adapted_CCOT_GW/real_data", 
                "LFC_MouseAllelicSeries_mirna_Cortex.txt")
    }
    
  }
  ## sample M real data from real data set
  
  
  if (real_data) {
    dat <- read.csv(file, header = TRUE, sep = "\t",
                    colClasses = c("character", rep("numeric", 18)))
    
    test = 1
    while(  test !=0 ){
      truth <- dat[sample(1:nrow(dat), M, replace = FALSE), ]
      nms <- truth[, 1]
      truth <- as.matrix(truth[, -1])
      rownames(truth) <- nms
      nc <- ncol(truth)
      test = sum(dist(truth)< 2)
    }
    
  } else {
    ## a silly example
    truth <- matrix(rnorm(M * 6, sd = 5), ncol = M)

    truth <- t(truth)
    nc <- ncol(truth)
  }
  
  dat_micro <- matrix(NA, ncol = nc, nrow = 0)
  dat_mesg <- matrix(NA, ncol = nc, nrow = 0)
  
  label_micro <- c( )
  label_mesg <-  c( )
  
  for (m in 1:(M+1)) {
    if (m <= M) {
      ## micro
      nb_micro <- 1 + rpois(1, lambda_micro)
      dat_micro <- rbind(dat_micro,
                         matrix(truth[m, ], ncol = nc, nrow = nb_micro, byrow = TRUE) +
                           matrix(rnorm(nb_micro * nc, sd = sd_micro), ncol = nc, nrow = nb_micro))
      label_micro = c(label_micro, rep(m-1, nb_micro))
      ## mesg
      nb_mesg <- 1 + rpois(1, lambda_mesg)
      dat_mesg <- rbind(dat_mesg,
                        matrix(-truth[m, ], ncol = nc, nrow = nb_mesg, byrow = TRUE) +
                          matrix(rnorm(nb_mesg * nc, sd = sd_mesg), ncol = nc, nrow = nb_mesg))
      label_mesg <- c(label_mesg, rep(m-1, nb_mesg ))
      ## ##
      ## ## we can use something more elaborate than -Identity!
      ## ##
      ## ## do later...
    } else {
      ## micro
      nb_micro <- 1 + rpois(1, r_micro)
      dat_micro <- rbind(dat_micro,
                         matrix(rnorm(nb_micro * nc, sd = sd_r_micro), ncol = nc, nrow = nb_micro))
      label_micro = c(label_micro, rep( m-1, nb_micro))
      ## mesg
      nb_mesg <- 1 + rpois(1, r_mesg)
      dat_mesg <- rbind(dat_mesg,
                        matrix(rnorm(nb_mesg * nc, sd = sd_r_mesg), ncol = nc, nrow = nb_mesg))
      label_mesg <- c(label_mesg, rep(m-1, nb_mesg))
    }
  }
  ind_micro = sample(nrow(dat_micro))
  ind_mesg = sample(nrow(dat_mesg))
  
  dat_micro <- dat_micro[ind_micro, ]
  dat_mesg <- dat_mesg[ind_mesg, ]
  
  label_micro = label_micro[ind_micro]
  label_mesg =label_mesg[ind_mesg]
  return(list(micro = dat_micro, mesg = dat_mesg, lb_micro = 
                label_micro, lb_mesg = label_mesg, mean = truth))
  
}

##################


datas = list()
for(i in 1:30){
  
  data = simulate_data( 15, lambda_micro = 15, lambda_mesg = 15,r_micro =  0, r_mesg =  0,
                        
                        sd_micro = 0.01, sd_mesg = 0.01)
  datas[[i]] = data
}

np$savez_compressed('sample_C', dats = datas)
