#Load in preamble containing functions and calling packages
source("preamble.R")

#Flag for whether to fit the model. Saves time if the model weights are already computed
fit_models = T

# Arguments ---------------------------------------------

#Specify the quantile level 
quant.level = 0.90

#Specify the number of units in the hidden layers of the quantile regression neural network
quant.nunits = eval(parse(text="c(64,64,64)"))

#Specify the number of units in the hidden layers of the deepGauge neural network
gauge.nunits = eval(parse(text="c(64,64,64)")) 

# Load in data ------------------------------------------------------------

#Load in example data set 
site_num = 1

data_orig = readRDS(file=paste0("wave_data_site_",site_num,".rds"))
data_orig = cbind(data_orig$hs,data_orig$ws,data_orig$mslp)

#Transform data to Laplace coordinates using rank/empirical transform
data_lap = apply(apply(data_orig,2,function(x){return(rank(x,ties.method = "random")/(length(x)+1))}),2,Laplace_inverse)

#Compute polar coordinates - see https://en.wikipedia.org/wiki/N-sphere
polar = rect2polar(t(data_lap)) 

#Compute sample size 
n = dim(data_lap)[1]

#Compute dimension of data
d = dim(data_lap)[2]

# Quantile regression procedure -----------------------------------------------------

## Instigate Keras and Tensorflow ##

#Load in virtual environment called 'myenv'
reticulate::use_virtualenv("myenv", required = T)

#Load in keras and tensorflow
packages = c("keras","tensorflow")
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

#Call tensorflow session 
# sess = k_get_session()
sess = tf$compat$v1$keras$backend$get_session()
sess$list_devices()

#Set seed for random initial dense layer weights. 
tf$random$set_seed(1) 

#Create training data; we label the response Y and predictor variables W.
Y=polar$r
W=data_lap/polar$r #These are points on the hypersphere 

rowSums(W^2) #To illustrate that all of these points have radius 1

# Select 20% of the data as a validation set
valid.inds=sample(1:n,round(n/5))

Y.train <- Y[-valid.inds]; W.train <- W[-valid.inds,]
Y.valid <- Y[valid.inds]; W.valid <- W[valid.inds,]

## Input layer ##
#T his is so that Keras knows the shape of the input to expect.

input.pseudo.angles <- layer_input(shape = d, name = 'input.pseudo.angles')

## Densely-connected MLP ##

# We define a ReLU neural network with exponential activation in the final layer (to ensure that the quantile is strictly positive)
qBranch <- input.pseudo.angles %>%
  layer_dense(units = quant.nunits[1], activation = 'relu', name = 'q_dense1',
              kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #First hidden layer
for(i in 2:length(quant.nunits)){
  qBranch  <- qBranch %>%
    layer_dense(units =quant.nunits[i], activation = 'relu', name = paste0('q_dense',i),
                kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #Subsequent hidden layers
}
# Final layer
qBranch  <- qBranch %>% layer_dense(units =1, activation = "exponential", name = 'q_final',
                                    kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #By setting the initial weights to 0 in the last layer, the output will always be exp(log(init_q))=init_q.
 
## Model compilation ##

# Construct Keras model
model <- keras_model(
  inputs = c(input.pseudo.angles), 
  outputs = c(qBranch)
)
summary(model)

# Define the loss. Note that custom loss functions must be written in a specific way, with input (y_true, y_pred)
# All function calls must use the Keras/Tensorflow backend, e.g., K$max.

# For quantile regression, we use the check/pinball/tilted loss.
tilted_loss <- function( y_true, y_pred) {
  K <- backend()
  
  error = y_true - y_pred
  return(K$mean(K$maximum(quant.level*error, (quant.level-1)*error)))
}

# Compile the model with the tilted loss and the adam optimiser
model %>% compile(
  optimizer=optimizer_adam(learning_rate=0.001),
  loss = tilted_loss,
  run_eagerly=T
)

# After every epoch, we use a checkpoint to save the weights. 
# Only the current best version of the model is saved, i.e., the one that minimises the loss evaluated on the validation data

checkpoint <- callback_model_checkpoint(filepath=paste0("QR_est/qr_fit_",site_num), monitor = "val_loss", verbose = 0,
                                        save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                        save_freq = "epoch")

## Train Keras model ##

# Set number of epochs for training
n.epochs <- 500 

# Set mini-batch size
batch.size <- 1024 

###HEREHEHEHEHEHEHEHEH

if(fit_models == T){
# Train Keras model. Loss values will be stored in history object.
history <- model %>% fit(
  list(W.train), Y.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(checkpoint,callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 5)),
  validation_data=list(list(  input.pseudo.angles=W.valid),Y.valid)
  
)
}
# plot(history)

# Load the best fitting model from the checkpoint save
model <- load_model_weights_tf(model,filepath=paste0("QR_est/qr_fit_",site_num))

# Then save the best model. Normally we can just save the weights, but we will use the quantile model when running deepGauge_test.R
# tf$saved_model$save(model,paste0("SimulationData/QR_est/best_qr_fit_",site_num))
save_model_tf(model,paste0("QR_est/best_qr_fit_",site_num))

# Get the predicted quantiles
pred.quant<-model %>% predict(W)
#Sanity check for fit. Should be roughly equal to quant.level
mean(pred.quant>Y) 
print(quant.level)

# Load in the Keras model for the associated quant.level quantile
# quant.model <- tf$saved_model$load(paste0("SimulationData/QR_est/best_qr_fit_",site_num))
quant.model <- load_model_tf(paste0("QR_est/best_qr_fit_",site_num),custom_objects = list("tilted_loss" = tilted_loss))

SA_d = 2*pi^(d/2)/gamma(d/2) #Surface area of d-sphere - https://en.wikipedia.org/wiki/N-sphere

#Sampling points on the n sphere
set.seed(1)
#sphere_sample = uniform_sample_dsphere(d=d,sampling_points = sampling_points)
sphere_sample = readRDS(paste0("dsphere_sample_",d,"d.RDS"))
sampling_points=dim(sphere_sample)[1]
#We first evaluate the quantile set at the initial angles
#Get the predicted quantiles at observed angles 

obs.init.est_quantile_set = W*as.numeric(pred.quant)


#Get the predicted quantiles at observed angles 
sample.init.pred.quant = k_get_value(quant.model(k_constant(sphere_sample))) 

sample.init.est_quantile_set = sphere_sample*as.numeric(sample.init.pred.quant)

#Work out values for rescaling

sample.quantile_maxs = apply(rbind(sample.init.est_quantile_set,obs.init.est_quantile_set),2,max)
sample.quantile_mins = apply(rbind(sample.init.est_quantile_set,obs.init.est_quantile_set),2,min)

#This function compute the angles we need to evaluate the neural network at to assess the scaled set at observed angles
sample.W_trans = t(apply(sphere_sample,1,inverse_angular_function,upper_max = sample.quantile_maxs,lower_min = sample.quantile_mins))

#Get the predicted quantiles at desired angles
sample.pred.quant = k_get_value(quant.model(k_constant(sample.W_trans)))

#Computing the corresponding quantile set
sample.est_quantile_set = sample.W_trans*as.numeric(sample.pred.quant)

sample.positivity_indices = apply(sample.est_quantile_set,2,function(x){return(x>0)})

#Adjusting quantile set to obtain limit set estimate
sample.initial_limit_set_est = apply(rbind(sample.est_quantile_set,sample.quantile_maxs,sample.quantile_mins,sample.positivity_indices),2,adjustment_func)

#Obtaining initial estimate of gauge set
sample.initial_gauge_est = 1/apply(sample.initial_limit_set_est,1,norm,type="2" )

#We first evaluate the quantile set at the initial angles

#Get the predicted quantiles at observed angles 
init.pred.quant = k_get_value(quant.model(k_constant(W))) 

init.est_quantile_set = W*as.numeric(init.pred.quant)

#This function compute the angles we need to evaluate the neural network at to assess the scaled set at observed angles
W_trans = t(apply(W,1,inverse_angular_function,upper_max = sample.quantile_maxs,lower_min = sample.quantile_mins))

#Get the predicted quantiles at desired angles
pred.quant = k_get_value(quant.model(k_constant(W_trans)))

#Computing the corresponding quantile set
est_quantile_set = W_trans*as.numeric(pred.quant)

positivity_indices = apply(est_quantile_set,2,function(x){return(x>0)})

#Adjusting quantile set to obtain limit set estimate
initial_limit_set_est = apply(rbind(est_quantile_set,sample.quantile_maxs,sample.quantile_mins,positivity_indices),2,adjustment_func)

#Obtaining initial estimate of gauge set
initial_gauge_est = 1/apply(initial_limit_set_est,1,norm,type="2" )




# Get the theoretical lower bound for the gauge function
gauge.lb <- apply(W,1,function(x){
  return(max(max(x), -min(x)))
} 
)

#Check we satisfy lower bound. Note this wont be perfect due to the fact we only evaluate coordinate wise min/max values at a subset of angles
sum(initial_gauge_est>=gauge.lb)


#Make 20% validation data
valid.inds=sample(1:n,round(n/5))

#We set values to some arbitrarily small number, e.g., -1e10. These will then be ignored when evaluating the loss funcction
Y.train<-Y.valid<-initial_gauge_est
Y.train[valid.inds]=-1e10
Y.valid[-valid.inds]=-1e10

#Change from vectors to matrices; required by keras. 
dim(Y)=c(length(Y),1); dim(Y.train)=c(length(Y.train),1); dim(Y.valid)=c(length(Y.valid),1)
dim(gauge.lb)=c(length(gauge.lb),1)

###### Build Keras model ######

# Let g = g.x + g.lb


## Input layers ##

# This is just so that Keras knows the shape of the input to expect.
# Need two inputs; the angles and the lowerbound of the gauge 

## Define input layers for angles 
input.pseudo.angles <- layer_input(shape = d, name = 'input.pseudo.angles')

## Define input layers for the lowerbound on g and r 
input.glb <- layer_input(shape = dim(gauge.lb)[2], name = 'gauge.lb_input')

 


## Densely-connected MLP for g##

# We define a ReLU neural network with exponential activation in the final layer (to ensure that the quantile is strictly positive)

g.xBranch <- input.pseudo.angles %>%
  layer_dense(units = gauge.nunits[1], activation = 'relu', name = 'g.x_dense1',
              kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #First hidden layer
if(length(gauge.nunits) >= 2){              
  for(i in 2:length(gauge.nunits)){
    g.xBranch  <- g.xBranch  %>%
      layer_dense(units =gauge.nunits[i], activation = 'relu', name = paste0('g.x_dense',i),
                  kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4))  #Subsequent hidden layers
  }
}

g.xBranch  <- g.xBranch %>% layer_dense(units =1, activation = "relu", name = 'g.x_final',
                                        kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) 

gBranch  <- layer_add(g.xBranch,input.glb) #Add g.x to g.lb to get an estimate for g

##### Model compilation #######

#Define output of Keras model. We concatenate the three components required to evaluate the loss function.
output <- gBranch

# Construct Keras model
model <- keras_model(
  inputs = c(input.pseudo.angles,input.glb), 
  outputs = output
)
summary(model)

MSE=function(y_true,y_pred){
  K <- backend()
  
  g=y_pred[all_dims(),1]
  g.true <- y_true[all_dims(),1]
  # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
  # arbitrarily large (<1e10) value to r and then taking the sign ReLu
  obsInds=K$sign(K$relu(g.true+1e9))
  
  out =(g.true-g)^2*(obsInds)
  return(K$sum(out)/K$sum(obsInds)) #Return average loss
}

  model %>% compile(
    optimizer=optimizer_adam(learning_rate=0.001),
    loss = MSE,
    run_eagerly=T
  )
  

###### Train Keras model ######

n.epochs <- 50 # Set number of epochs for training
batch.size <- 1024 # Set mini-batch size. Needs to be roughly a multiple of the training size

if(fit_models == T){
history <- model %>% fit(
  list(W,gauge.lb), Y.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 5)),
  validation_data=list(list(  input.pseudo.angles=W, gauge.lb_input=gauge.lb),Y.valid)
  
)
}
# plot(history)

initial.gauge.weights = model$get_weights()


## Gauge function estimation



#Find and uses only exceedances of Y above pred.quant
exceed.inds=which(Y > pred.quant)
Y[-exceed.inds]=-1e10

# Get the lower bound for R, i.e., the quant.level quantile estimate
# We fit the truncated gamma above this level 
pred.quant = k_get_value(quant.model(k_constant(W))) 
r.lb <- pred.quant 

#Make 20% validation data
valid.inds=sample(1:n,round(n/5))

#We set values to some arbitrarily small number, e.g., -1e10. These will then be ignored when evaluating the loss funcction
Y.train<-Y.valid<-Y
Y.train[valid.inds]=-1e10
Y.valid[-valid.inds]=-1e10

#Change from vectors to matrices; required by keras. 
dim(Y)=c(length(Y),1); dim(Y.train)=c(length(Y.train),1); dim(Y.valid)=c(length(Y.valid),1)
dim(r.lb)=c(length(r.lb),1)

#Create supplement for evaluating maxima
#W.supplement = rbind(W,sphere_sample)
W.supplement = rbind(W)
###### Build Keras model ######

# Let g = g.x + g.lb
# We build a densely-connected MLP to estimate the distance g.x between the gauge g and the lowerbound gauge.lb.
# The activation in the final layer is ReLU, and so g.x >= 0.


# Set initial alpha estimate across all Phi. 
init_alpha=d

## Input layers ##
source("new_layers.R")					  

# This is just so that Keras knows the shape of the input to expect.
# Need three inputs; the angles, the lowerbound of the gauge, and the lowerbound of R. 

## Define input layers for angles 
input.pseudo.angles <- layer_input(shape = d, name = 'input.pseudo.angles')

## Define input layers for the lowerbound on g and r 

input.rlb <- layer_input(shape = dim(r.lb)[2], name = 'r.lb_input') #This will features in the loss function only.

# Define a layer for the alpha parameter
# Although this layer takes in the angle input, the first layer is constructed so that it always returns 1.
# The second layer is then a single trainable weight which will determine the value of alpha. 
# Exponential activation used to ensure strict positivity

alphaBranch <- input.pseudo.angles %>% layer_dense(units = 1 ,activation = 'relu', trainable=F,
                                                   weights=list(matrix(0,nrow=d,ncol=1),
                                                                array(1,dim=c(1))), 
                                                   name = 'alpha_dense') %>%
  layer_dense(units = 1 ,activation = 'exponential',
              use_bias = F,
              weights=list(matrix(log(init_alpha),nrow=1,ncol=1)),
              name = 'alpha_activation',trainable=T)


## Densely-connected MLP for g##

# We define a ReLU neural network with exponential activation in the final layer (to ensure that the quantile is strictly positive)

input.glb <- layer_g_lb(input.pseudo.angles,input_dim=d)


g.xBranch <- input.pseudo.angles %>%
  layer_dense(units = gauge.nunits[1], activation = 'relu', name = 'g.x_dense1',
              kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4),
              weights=list(initial.gauge.weights[[1]],initial.gauge.weights[[2]])) #First hidden layer
if(length(gauge.nunits) >= 2){              
  for(i in 2:length(gauge.nunits)){
    g.xBranch  <- g.xBranch  %>%
      layer_dense(units =gauge.nunits[i], activation = 'relu', name = paste0('g.x_dense',i),
                  kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4),
                  weights=list(initial.gauge.weights[[2*i-1]],initial.gauge.weights[[2*i]]))  #Subsequent hidden layers
  }
}

g.xBranch  <- g.xBranch %>% layer_dense(units =1, activation = "relu", name = 'g.x_final',
                                        kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4),
                                        weights=list(initial.gauge.weights[[2*length(gauge.nunits)+1]],initial.gauge.weights[[2*length(gauge.nunits)+2]]))



gBranch  <- layer_add(g.xBranch,input.glb) #Add g.x to g.lb to get an estimate for g


g.model <- keras_model(
  inputs = c(input.pseudo.angles), 
  outputs = gBranch
)

#gBranch2 = g.model(k_constant(W.supplement))


# hwbranch <- layer_multiply(1/gBranch2, k_constant(W.supplement))
hw_coordmax_branch <- layer_compute_max(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)
hw_coordmin_branch <- layer_compute_min(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)

Wtransbranch <- layer_inverse_angular_transform(input_dim=d)(input.pseudo.angles,hw_coordmax_branch,hw_coordmin_branch)
gtransBranch = g.model(Wtransbranch)
dGtransbranch <- layer_multiply(1/gtransBranch, Wtransbranch)


adjusted_dGtrans_branch <- layer_adjust(input_dim=d)(dGtransbranch,hw_coordmax_branch,hw_coordmin_branch)

adjusted.g.branch <- layer_limitset_to_gauge(adjusted_dGtrans_branch, input_dim=d)

##### Model compilation #######

#Define output of Keras model. We concatenate the three components required to evaluate the loss function.
output <- layer_concatenate(c(alphaBranch,adjusted.g.branch,input.rlb)) 

# Construct Keras model
model2 <- keras_model(
  inputs = c(input.pseudo.angles,input.rlb), 
  outputs = output
)
summary(model2)

# Define the loss. Note that custom loss functions must be written in a specific way, with input (y_true, y_pred)
# and all function calls using the Keras/Tensorflow backend, e.g., K$max.

# Compile the model with the adam optimiser. Use truncGamma_nll if censored.nll == 0 and censGamma_nll if censored.nll == 1

model2 %>% compile(
    optimizer=optimizer_adam(learning_rate=0.001),
    loss = truncGamma_nll,
    run_eagerly=T
  )
checkpoint <- callback_model_checkpoint(filepath=paste0("Gauge_est/gauge_fit_",site_num), monitor = "val_loss", verbose = 0,
                                        save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                        save_freq = "epoch")

###### Train Keras model ######

n.epochs <- 200 # Set number of epochs for training
batch.size <- 4096 # Set mini-batch size. Needs to be roughly a multiple of the training size

if(fit_models == T){
history <- model2 %>% fit(
  list(W,r.lb), Y.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(checkpoint,callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 5)),
  validation_data=list(list(  input.pseudo.angles=W, r.lb_input = r.lb),Y.valid)
  
)
}
# plot(history)

# Load the best fitting model from the checkpoint save
model2 <- load_model_weights_tf(model2,filepath=paste0("Gauge_est/gauge_fit_",site_num))

# Then save the best model.

W.supplement = rbind(W,sphere_sample)
n.epochs <- 100 # Set number of epochs for training
batch.size <- 4096 # Set mini-batch size. Needs to be roughly a multiple of the training size


# hwbranch <- layer_multiply(1/gBranch2, k_constant(W.supplement))
hw_coordmax_branch <- layer_compute_max(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)
hw_coordmin_branch <- layer_compute_min(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)

Wtransbranch <- layer_inverse_angular_transform(input_dim=d)(input.pseudo.angles,hw_coordmax_branch,hw_coordmin_branch)
gtransBranch = g.model(Wtransbranch)
dGtransbranch <- layer_multiply(1/gtransBranch, Wtransbranch)


adjusted_dGtrans_branch <- layer_adjust(input_dim=d)(dGtransbranch,hw_coordmax_branch,hw_coordmin_branch)

adjusted.g.branch <- layer_limitset_to_gauge(adjusted_dGtrans_branch, input_dim=d)


##### Model compilation #######


#Define output of Keras model. We concatenate the three components required to evaluate the loss function.

output <- layer_concatenate(c(alphaBranch,adjusted.g.branch,input.rlb)) 

# Construct Keras model
model2 <- keras_model(
  inputs = c(input.pseudo.angles,input.rlb), 
  outputs = output
)
summary(model2)

# Define the loss. Note that custom loss functions must be written in a specific way, with input (y_true, y_pred)
# and all function calls using the Keras/Tensorflow backend, e.g., K$max.

# Compile the model with the adam optimiser. Use truncGamma_nll if censored.nll == 0 and censGamma_nll if censored.nll == 1

model2 %>% compile(
  optimizer=optimizer_adam(learning_rate=0.001),
  loss = truncGamma_nll,
  run_eagerly=T
)

if(fit_models == T){
history <- model2 %>% fit(
  list(W,r.lb), Y.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(checkpoint,callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 6)),
  validation_data=list(list(  input.pseudo.angles=W,  r.lb_input = r.lb),Y.valid)
  
)		
}
# Load the best fitting model from the checkpoint save
model2 <- load_model_weights_tf(model2,filepath=paste0("Gauge_est/gauge_fit_",site_num))

gauge_model <- model2
				 
#Get gauge estimates
predictions = k_get_value(gauge_model(list(k_constant(W),k_constant(r.lb)))) 
print(apply(W/predictions[,2],2,max))
print(apply(W/predictions[,2],2,min))

pred.gauge.supplement =k_get_value(gauge_model(list(k_constant((W.supplement)),
                                                    k_constant(as.matrix(rep(1,nrow(W.supplement)))))))[,2]
print(apply(W.supplement/pred.gauge.supplement,2,max))
print(apply(W.supplement/pred.gauge.supplement,2,min))

#First reset everything - make sure we are just looking at exceedances of the threshold

Y=polar$r
W=data_lap/Y

quant.model <- load_model_tf(paste0("QR_est/best_qr_fit_",site_num),custom_objects = list("tilted_loss" = tilted_loss))

gauge_model <- load_model_weights_tf(model2,filepath=paste0("Gauge_est/gauge_fit_",site_num))

r.lb.sphere = k_get_value(quant.model(k_constant(sphere_sample))) #Get the predicted quantiles/ r.lb. #Get lower bounds for g

gauge.lb.sphere <- apply(sphere_sample,1,function(x){
  return(max(max(x), -min(x)))
} 
)

dim(gauge.lb.sphere)=c(length(gauge.lb.sphere),1); dim(r.lb.sphere)=c(length(r.lb.sphere),1)

#Get associated g estimate
pred.gauge.sphere <- k_get_value(gauge_model(list(k_constant(sphere_sample),k_constant(r.lb.sphere))))[,2]


pred.quant = k_get_value(quant.model(k_constant(W))) 

#Find and uses only exceedances of Y above pred.quant
exceed.inds=which(Y > pred.quant)
Y <- Y[exceed.inds]; W <- W[exceed.inds,]

# Get the lower bound for the gauge function
gauge.lb <- apply(W,1,function(x){
  return(max(max(x), -min(x)))
} 
)

# Get the lower bound for R, i.e., the quant.level quantile estimate
r.lb <- pred.quant[exceed.inds] #Only for exceedances Y > pred.quant

dim(gauge.lb)=c(length(gauge.lb),1); dim(r.lb)=c(length(r.lb),1)
#QQ plot associated with truncated Gamma distributions

#Get gauge estimates from observed data
predictions = k_get_value(gauge_model(list(k_constant(W),k_constant(r.lb)))) 

#First column of predictions gives the alpha estimate. 
pred.alpha=predictions[1,1]

#Second column of predictions gives the gauge estimate. 
pred.gauge=predictions[,2]

unif_exceedances =  exp(pgamma(Y,shape = pred.alpha,rate = pred.gauge,lower.tail = F,log.p=T)-pgamma(r.lb[,1],shape = pred.alpha,rate = pred.gauge,lower.tail = F,log.p=T))

exp_exceedances = qexp(unif_exceedances,lower.tail=F)

m = 1000

observed_quants = quantile(exp_exceedances, probs=(1:m)/(m+1))
theoretical_quants = qexp((1:m)/(m+1))

pdf(file=paste0("qqplot_",site_num,".pdf"),width=4,height=4)

#Plotting parameters
par(mfrow=c(1,1),mgp=c(2.25,0.75,0),mar=c(4,4,1,1))

plot(theoretical_quants,observed_quants,xlim=range(theoretical_quants,observed_quants),
     ylim=range(theoretical_quants,observed_quants),pch=16,col=1,ylab="Empirical",xlab="Model",
     cex.lab=1.3, cex.axis=1.2,cex.main=1.8, cex=0.5)
abline(a=0,b=1,lwd=3,col=2)
points(theoretical_quants,observed_quants,pch=16,col=1, cex=0.5)

dev.off()
