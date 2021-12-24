
start_time = Sys.time()

### I. DDEFINE FUNCTION 

#setwd("~/AResearch/POT-master/Adapted_CCOT_GW")
source('simulate_dppG.R')


### II. Parallelization 

# declare the library
library(doSNOW)
library(foreach)
library(iterators)

library(doParallel)

# Set up parallelization
myCluster <- makeCluster(10 )
registerDoParallel(myCluster)


# Implement the parallel computing
system.time({
  out.list <- foreach(i=1:30, .packages=c( "gtools", "spatstat", "MASS")
  ) %dopar% {
    d = sample_data()
    d['stt'] = i
    return(d)
  } 
  
  
})


# Stop the parallel computing
stopCluster(myCluster) 

# Save data
library(reticulate)
np = import('numpy')
np$savez_compressed('sample_B', dats= out.list)

#save(out.list, file = "simulate_parallel_update.Rdata")
print('successful')
end_time  = Sys.time()

u = end_time - start_time 
print( u)
