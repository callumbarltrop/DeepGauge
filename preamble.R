
rm(list=ls())

# Functions/packages ------------------------------------------------------

rect2polar <- function (x) 
{
  if (!is.matrix(x)) {
    x <- as.matrix(x, ncol = 1)
  }
  n <- nrow(x)
  m <- ncol(x)
  r <- rep(0, m)
  phi <- matrix(0, nrow = n - 1, ncol = m)
  for (j in 1:m) {
    y <- sqrt(cumsum(rev(x[, j]^2)))
    r[j] <- y[n]
    if (r[j] > 0) {
      if (n > 2) {
        for (k in 1:(n - 2)) {
          if (y[n - k + 1] > 0) 
            phi[k, j] <- acos(x[k, j]/y[n - k + 1])
          else {
            phi[k, j] <- ifelse(x[k, j] > 0, 0, pi)
          }
        }
      }
      if (y[2] > 0) {
        phi[n - 1, j] <- acos(x[n - 1, j]/y[2])
        if (x[n, j] < 0) {
          phi[n - 1, j] <- 2 * pi - phi[n - 1, j]
        }
      }
      else {
        phi[n - 1, j] <- ifelse(x[n, j] > 0, 0, pi)
      }
    }
  }
  return(list(r = r, phi = phi))
}

polar2rect <- function (r, phi) 
{
  m <- length(r)
  if (!is.matrix(phi)) {
    phi <- as.matrix(phi, ncol = 1)
  }
  stopifnot(m == ncol(phi))
  n <- nrow(phi) + 1
  x <- matrix(0, nrow = n, ncol = m)
  for (j in 1:m) {
    c.term <- cos(phi[, j])
    s.term <- sin(phi[, j])
    y <- c(1, cumprod(s.term))
    z <- c(c.term, 1)
    x[, j] <- r[j] * y * z
  }
  return(x)
}


Laplace_inverse = function(u){ #standard Laplace quantile function
  x = c()
  x[u<=0.5] = log(2*u[u<=0.5])
  x[u>0.5] = -log(2*(1-u[u>0.5]))
  return(x)
}

Laplace_cdf = function(x){ #Standard Laplace cumulative distribution function
  u = c()
  u[x<0] = exp(x[x<0])/2
  u[x>=0] = 1-exp(-x[x>=0])/2
  return(u)
}

# This loss is the negative log-likelihood for R| R > r.lb~truncatedGamma(alpha, g).
truncGamma_nll <- function( y_true, y_pred) {
  
  K <- backend()
  
  alpha=y_pred[all_dims(),1]
  g=y_pred[all_dims(),2]
  r.lb=y_pred[all_dims(),3]
  r <- y_true[all_dims(),1]
  # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
  # arbitrarily large (<1e10) value to r and then taking the sign ReLu
  obsInds=K$sign(K$relu(r+1e9))
  r <- K$relu(r)
  
  #For unobserved values of r only
  r=r- r*(1-obsInds)+(1-obsInds) #If missing, set r to 1
  g=g- g*(1-obsInds)+(1-obsInds) #If missing, set g to 1
  r.lb=r.lb- r.lb*(1-obsInds)+(1-obsInds) #If missing, set r.lb to 1
  
  # Normal Gamma density
  nll1 = alpha * K$log(g) + (alpha-1)*K$log(r) - g*r - tf$math$lgamma(alpha)
  nll1=nll1- nll1*(1-obsInds) #If missing, set nll1 to 0
  
  # Gamma distribution function for Gamma(alpha, g*r.lb)
  nll2 = -K$log(tf$math$igammac(alpha,g*r.lb))
  nll2=nll2- nll2*(1-obsInds) #If missing, set nll2 to 0. Hence, nll1+nll2 is 0 if input r is missing.
  
  
  return(-K$sum(nll1+nll2)/K$sum(obsInds)) #Return average loss
}

censGamma_nll <- function( y_true, y_pred) {
  
  K <- backend()
  
  alpha=y_pred[all_dims(),1]
  g=y_pred[all_dims(),2]
  r.lb=y_pred[all_dims(),3]
  r <- y_true[all_dims(),1]
  # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
  # arbitrarily large (<1e10) value to r and then taking the sign ReLu
  obsInds=K$sign(K$relu(r+1e9))
  exceedInds=K$sign(K$relu(r-r.lb))
  
  r <- K$relu(r)
  
  #For unobserved values of r only
  r=r- r*(1-obsInds)+(1-obsInds) #If missing, set r to 1
  g=g- g*(1-obsInds)+(1-obsInds) #If missing, set g to 1
  r.lb=r.lb- r.lb*(1-obsInds)+(1-obsInds) #If missing, set r.lb to 1
  
  # Normal Gamma density
  nll1 = alpha * K$log(g) + (alpha-1)*K$log(r) - g*r - tf$math$lgamma(alpha)
  nll1=nll1- nll1*(1-obsInds) #If missing, set nll1 to 0
  nll1=nll1- nll1*(1-exceedInds) #If non-exceed, set nll1 to 0
  
  
  # Gamma distribution function for Gamma(alpha, g*r.lb)
  nll2 = K$log(tf$math$igamma(alpha,g*r.lb))
  nll2=nll2- nll2*(1-obsInds) #If missing, set nll2 to 0. Hence, nll1+nll2 is 0 if input r is missing.
  nll2=nll2- nll2*(exceedInds) #If exceed, set nll2 to 0. 
  
  
  return(-K$sum(nll1+nll2)/K$sum(obsInds)) #Return average loss
}

gauge_mv_norm = function(x,P){
  return(t( sapply(x,sign)*(abs(x))^(1/2) )%*%P%*%( sapply(x,sign)*(abs(x))^(1/2) ) )
}

gauge_logistic = function(x,dep,d){
  if(sum(sign(x)) == length(x)){
    return( (1/dep)*sum(x) + (1-(d/dep))*min(x) )
  } else if(sum(sign(x)) == -length(x)){
    return( ( sum( (-x)^(1/dep)  ) )^(dep) )
  } else {
    return( (1/dep)*sum(x[sign(x)==1]) +( sum( (-x[sign(x)==-1])^(1/dep)  ) )^(dep) )
  }
  
}

gauge_mv_t = function(x,nu,d){
  return( -(1/nu)*sum(abs(x)) + (1 + (d/nu))*max(abs(x)) )
}

uniform_sample_dsphere = function(d,sampling_points,simulation_size = 1000){ #Using the technique of Marsaglia (1972) - see https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-43/issue-2/Choosing-a-Point-from-the-Surface-of-a-Sphere/10.1214/aoms/1177692644.full
  sphere_sample = matrix(nrow=0,ncol=d)
  while(dim(sphere_sample)[1]<sampling_points){
    uniform_sample = matrix(runif(d*simulation_size,-1,1),ncol=d)
    r = apply(uniform_sample,1,norm,type="2")
    sphere_sample = rbind(sphere_sample, (uniform_sample[r<=1,]/r[r<=1])) #Reject samples not inside sphere, then rescale to get points on sphere
  }
  return(sphere_sample[1:sampling_points,])
}

adjustment_func = function(x){
  n = (length(x)-2)/2
  limit_set_est = as.vector(x[1:n])
  upper = as.numeric(x[n+1])
  lower = as.numeric(x[n+2])
  indices = as.numeric(x[(n+3):(length(x))])
  adjusted_est = c()
  adjusted_est[which(indices == 1)] = limit_set_est[which(indices == 1)]/upper
  adjusted_est[which(indices == 0)] = -limit_set_est[which(indices == 0)]/lower
  return(adjusted_est)
}

inverse_angular_function = function(wstar,upper_max,lower_min){
  
  w = c()
  
  nonzero_indices = abs(wstar) > 1e-14 
  
  w[!nonzero_indices] = 0
  
  wstar_nonzero = wstar[nonzero_indices]
  upper_max_nonzero = upper_max[nonzero_indices]
  lower_min_nonzero = lower_min[nonzero_indices]
  
  if(sum(nonzero_indices) == 1){
    
    w[nonzero_indices] = sign(wstar_nonzero)
    
  } else {
    
    pos_ind = wstar_nonzero > 0
    
    b = ifelse(pos_ind,upper_max_nonzero,-lower_min_nonzero)
    
    w_nonzero = c()
    
    w_nonzero[length(wstar_nonzero)] = sign(wstar_nonzero[length(wstar_nonzero)]) * sqrt( 1/(1 + sum( (((wstar_nonzero[-length(wstar_nonzero)])*(b[-length(wstar_nonzero)]))/ ((wstar_nonzero[length(wstar_nonzero)])*(b[length(wstar_nonzero)])) )^2 ) ) )
    w_nonzero[1:(length(w_nonzero) - 1)] = (w_nonzero[length(w_nonzero)]) * ((wstar_nonzero[-length(wstar_nonzero)])*(b[-length(wstar_nonzero)]))/ ((wstar_nonzero[length(wstar_nonzero)])*(b[length(wstar_nonzero)]))
    
    w[nonzero_indices] = w_nonzero
  }
  
  return(w)
}

#Checking for required packages. This function will install any required packages if they are not already installed
packages = c("ismev","evd","randcorr","mvtnorm") #,"keras","tensorflow", reticulate
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

#Function for estimating joint tail probabilities from the estimated limit set
adf_prob_est = function(sim_data,limit_set,point_est,q){
  #sim_data is data matrix on standard Laplace margins
  #limit_set is the matrix of cartesian coordinates for the limit set 
  #point_est is the point of interest on Laplace margins
  #q is the quantile for the min-projection exceedances. Must be large in magnitude 
  
  w = point_est/norm(matrix(point_est),type="2")
  
  r = norm(matrix(point_est),type="2")
  
  min_proj = apply(sim_data,1,function(x){return(min(x/w))})
  
  q = max(q,mean(c(1-mean(min_proj>0),1))) #To ensure we take a positive value 
  
  u = quantile(min_proj,q)
  
  if(r > u){
    
    adf_point_est = 1/(max(apply(limit_set,1,function(x){return(min(x/w,na.rm = T))})))
    
    t = r - u
    
    prob_est = as.numeric(exp(-t*adf_point_est)*(1-q))
    
  } else {
    
    prob_est = mean(min_proj > r)
    
  }
  
  return(prob_est)
}

"pmvlog"<- function(q, dep, d = 2, mar = c(0,1,0), lower.tail = TRUE){
    if(lower.tail) {
      if(length(dep) != 1 || mode(dep) != "numeric" || dep <= 0 ||
         dep > 1) stop("invalid argument for `dep'")
      if(is.null(dim(q))) dim(q) <- c(1,d)
      if(ncol(q) != d) stop("`q' and `d' are not compatible")
      q <- mtransform(q, mar)
      pp <- exp(-apply(q^(1/dep),1,sum)^dep)
    } else {
      pp <- numeric(1)
      ss <- c(list(numeric(0)), subsets(d))
      ssl <- d - sapply(ss, length)
      for(i in 1:(2^d)) {
        tmpq <- q
        tmpq[ss[[i]]] <- Inf
        pp <- (-1)^ssl[i] * Recall(tmpq, dep, d, mar) + pp
      }
    }
    pp
  }

"pmvevd" <-function(q, dep, asy, model = c("log", "alog"), d = 2,
           mar = c(0,1,0), lower.tail = TRUE)
  {
    model <- match.arg(model)
    if(model == "log" && !missing(asy))
      warning("ignoring `asy' argument")
    
    switch(model,
           log = pmvlog(q = q, dep = dep, d = d, mar = mar,
                        lower.tail = lower.tail),
           alog = pmvalog(q = q, dep = dep, asy = asy, d = d, mar = mar,
                          lower.tail = lower.tail)) 
  }


block_bootstrap_function = function(data,k,n=length(as.matrix(data)[,1])){ #function for performing block bootstrapping
  #data is bivariate dataset
  #k is block length
  data = as.matrix(data)
  no_blocks = ceiling(n/k)
  n_new = no_blocks*k
  new_data = matrix(NA,nrow=n_new,ncol=dim(data)[2])
  indices = 1:(n-k+1)
  start_points = sample(x=indices,size=no_blocks,replace=TRUE)
  for(i in 1:no_blocks){
    new_data[((i-1)*k+1):(i*k),] = data[(start_points[i]:(start_points[i]+k-1)),]
  }
  return(new_data[1:n,])
}