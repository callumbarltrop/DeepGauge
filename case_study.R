#Load in preamble containing functions and calling packages
source("preamble.R")

#Flag for whether to fit the model. Set this to FALSE if you have already fitted the models and just want to perform inference/compute diagnostics
fit_models = F

#Fix a random seed to ensure reproducibility 
set.seed(2311732) 

# Arguments that specify the architectures and threshold level ---------------------------------------------

#Specify the quantile level 
quant.level = 0.80

#Specify the number of units in the hidden layers of the quantile regression neural network
quant.nunits = eval(parse(text="c(64,64)"))

#Specify the number of units in the hidden layers of the deepGauge neural network
gauge.nunits = eval(parse(text="c(64,64)")) 

# Load in data ------------------------------------------------------------

#Load in example data set 
site_num = 1

data_orig = readRDS(file=paste0("Datafiles/wave_data_site_",site_num,".rds"))
data_orig = cbind(data_orig$hs,data_orig$ws,data_orig$mslp)

#Transform data to Laplace coordinates using rank/empirical transform. Note that we account for ties in the data at random. These ties come as a result of measurement accuracy
data_lap = apply(apply(data_orig,2,function(x){return(rank(x,ties.method = "random")/(length(x)+1))}),2,Laplace_inverse)

#Compute polar coordinates - see https://en.wikipedia.org/wiki/N-sphere
polar = rect2polar(t(data_lap)) 

#Compute sample size 
n = dim(data_lap)[1]

#Compute dimension of data
d = dim(data_lap)[2]

# Quantile regression procedure -----------------------------------------------------

#Instigate Keras and Tensorflow

#Load in keras and tensorflow in R
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

#Load in python virtual environment called 'deepGauge_env'
reticulate::use_virtualenv("deepGauge_env", required = T)

#Check keras is available 
keras::is_keras_available() #should return TRUE

#Call tensorflow session 
sess = tf$compat$v1$keras$backend$get_session()
sess$list_devices()

#Set seed for random initial dense layer weights. 
tf$random$set_seed(1) 

#Create training data; we label the response R and predictor variables W.
R=polar$r
W=data_lap/polar$r #These are points on the hypersphere 

rowSums(W^2)[1:10] #To illustrate that all of these points have radius 1

#Select 20% of the data as a validation set
valid.inds=sample(1:n,round(n/5))

R.train <- R[-valid.inds]; W.train <- W[-valid.inds,]
R.valid <- R[valid.inds]; W.valid <- W[valid.inds,]

#Specify input layer
#This is so that Keras knows what data shape/dimensions to expect.

input.pseudo.angles <- layer_input(shape = d, name = 'input.pseudo.angles')

#Specify a densely-connected MLP
#We define a ReLU neural network with exponential activation in the final layer (to ensure that the radial quantile is strictly positive)
qBranch <- input.pseudo.angles %>%
  layer_dense(units = quant.nunits[1], activation = 'relu', name = 'q_dense1',
              kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #First hidden layer
for(i in 2:length(quant.nunits)){
  qBranch  <- qBranch %>%
    layer_dense(units =quant.nunits[i], activation = 'relu', name = paste0('q_dense',i),
                kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #Subsequent hidden layers
}
#Final output layer
qBranch  <- qBranch %>% layer_dense(units =1, activation = "exponential", name = 'q_final',
                                    kernel_regularizer = regularizer_l1_l2(l1 = 1e-4, l2 = 1e-4)) #By setting the initial weights to 0 in the last layer, the output will always be exp(log(init_q))=init_q.
 
#Model compilation
#Construct Keras model
model <- keras_model(
  inputs = c(input.pseudo.angles), 
  outputs = c(qBranch)
)
summary(model)

#Define the loss. Note that custom loss functions in Keras must be written in a specific way, with input (y_true, y_pred)
#All function calls must use the Keras/Tensorflow backend, e.g., K$max.
#For quantile regression, we use the check/pinball/tilted loss.
tilted_loss <- function( y_true, y_pred) {
  K <- backend()
  
  error = y_true - y_pred
  return(K$mean(K$maximum(quant.level*error, (quant.level-1)*error)))
}

#Compile the model with the tilted loss and the adam optimiser. We use a learning rate of 0.001
model %>% compile(
  optimizer=optimizer_adam(learning_rate=0.001),
  loss = tilted_loss,
  run_eagerly=T
)

#After every epoch, we use a checkpoint to save the weights. 
#Only the current best version of the model is saved, i.e., the one that minimises the loss evaluated on the validation data
checkpoint <- callback_model_checkpoint(filepath=paste0("QR_est/qr_fit_",site_num), monitor = "val_loss", verbose = 0,
                                        save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                        save_freq = "epoch")

#Train the Keras model
#Set number of epochs for training
n.epochs <- 50

#Set mini-batch size
batch.size <- 1024 

#We only fit the model if the weights are not already saved
if(fit_models == T){
#Train Keras model. Loss values will be stored in history object.
history <- model %>% fit(
  list(W.train), R.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(checkpoint,callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 5)),
  validation_data=list(list(  input.pseudo.angles=W.valid),R.valid)
  
)
}

#Load the best fitting model from the checkpoint save
model <- load_model_weights_tf(model,filepath=paste0("QR_est/qr_fit_",site_num))

#Then save the best model 
save_model_tf(model,paste0("QR_est/best_qr_fit_",site_num))

#Load in the Keras model weights 
quant.model <- load_model_tf(paste0("QR_est/best_qr_fit_",site_num),custom_objects = list("tilted_loss" = tilted_loss))

#Get the predicted quantiles at the observed angles 
pred.quant = k_get_value(quant.model(k_constant(W)))

#As a sanity check for the model fit, we compute the proportion of radial observations below the estimated threshold level. 
#This should be roughly equal to quant.level
mean(pred.quant>R) 
print(quant.level)




# Pre-training the neural network for the gauge function using the quantile function ------------------

#We use the estimated quantile set as an initial estimate of the limit set/gauge function. 
#This requires us to rescale the quantile set to lie in the [-1,1]^d box.

#Load in a large sample of points on the hypersphere. These points are sampled uniformly and provide a dense coverage of the surface. 
sphere_sample = readRDS(paste0("Datafiles/dsphere_sample_",d,"d.RDS"))
sampling_points=dim(sphere_sample)[1]

#Evaluate the quantile function/set for every angle on the dense hypersphere sample 
sample.init.pred.quant = k_get_value(quant.model(k_constant(sphere_sample))) 
sample.init.est_quantile_set = sphere_sample*as.numeric(sample.init.pred.quant)

#Work out coordinate wise maxima and minima on the quantile set 
sample.quantile_maxs = apply(rbind(sample.init.est_quantile_set,sample.init.est_quantile_set),2,max)
sample.quantile_mins = apply(rbind(sample.init.est_quantile_set,sample.init.est_quantile_set),2,min)

rm(sample.init.pred.quant,sample.init.est_quantile_set) #Remove these objects to save memory 

#This function applies the inverse angular transformation to the observed angles 
#Note these points will also be on the hypersphere 
W_trans = t(apply(W,1,inverse_angular_function,upper_max = sample.quantile_maxs,lower_min = sample.quantile_mins))

#Get the predicted quantiles at the transformed angles 
pred.quant = k_get_value(quant.model(k_constant(W_trans)))

#Computing the corresponding quantile set
est_quantile_set = W_trans*as.numeric(pred.quant)

#Rescaling the quantile set to lie exactly in the unit box [-1,1]^d
initial_limit_set_est = apply(rbind(est_quantile_set,sample.quantile_maxs,sample.quantile_mins,apply(est_quantile_set,2,function(x){return(x>0)})),2,adjustment_func)

#Check angles on the rescaled set are the sample as the original observed angles. This value should be approximately 0
max(abs(initial_limit_set_est/apply(initial_limit_set_est,1,l2_norm ) - W))

#Specifying an initial estimator for the gauge function at each observed angle. This simply corresponds to the radii of the rescaled set at the transformed angles 
initial_gauge_est = 1/apply(initial_limit_set_est,1,l2_norm)

# Get the theoretical lower bound for the gauge function
gauge.lb <- apply(W,1,linf_norm)

#Check we satisfy the theoretical lower bound. This value should be approximately 1.  
#Note this wont be exactly due to the fact we only evaluate coordinate wise min/max values at a dense subset of angles. But it should be close
mean(initial_gauge_est>=gauge.lb)

#Select 20% of the data as a validation set
valid.inds=sample(1:n,round(n/5))

#For this neural network, the response variable will simply be the initial estimate of the gauge function
#When not in the validation/training set, we set the values equal to some arbitrarily small number, e.g., -1e10. These will then be ignored when evaluating the loss function
R.train<-R.valid<-initial_gauge_est
R.train[valid.inds]=-1e10
R.valid[-valid.inds]=-1e10

#Change from vectors to matrices. This is required by Keras  
dim(R)=c(length(R),1); dim(R.train)=c(length(R.train),1); dim(R.valid)=c(length(R.valid),1)
dim(gauge.lb)=c(length(gauge.lb),1)

#Build Keras model
#Note we impose conditions to ensure we are always above the theoretical lower bound for the gauge function

#Specify input layers
#Need two inputs; the angles and the lower bound of the gauge 

#Define input layers for angles 
input.pseudo.angles <- layer_input(shape = d, name = 'input.pseudo.angles')

#Define input layers for the lowerbound on g and r 
input.glb <- layer_input(shape = 1, name = 'gauge.lb_input')

#Specify a densely-connected MLP for the gauge function
#We define a ReLU neural network with exponential activation in the final layer (to ensure that the quantile is strictly positive)

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

gBranch  <- layer_add(g.xBranch,input.glb) #Add g.x to gauge lower bound to get a valid estimate for the gauge function

#Compiling the model 

#Define output of Keras model. We concatenate the three components required to evaluate the loss function.
output <- gBranch

#Construct Keras model
model <- keras_model(
  inputs = c(input.pseudo.angles,input.glb), 
  outputs = output
)
summary(model)

#We define the loss function for pre-training. This is simply the MSE between the initial gauge function estimate and the neural network output
MSE=function(y_true,y_pred){
  K <- backend()
  
  g=y_pred[all_dims(),1]
  g.true <- y_true[all_dims(),1]
  #Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
  #arbitrarily large (<1e10) value to r and then taking the sign ReLu
  obsInds=K$sign(K$relu(g.true+1e9))
  
  out =(g.true-g)^2*(obsInds)
  return(K$sum(out)/K$sum(obsInds)) #Return average loss
}

model %>% compile(
  optimizer=optimizer_adam(learning_rate=0.001),
  loss = MSE,
  run_eagerly=T
)

#We now train the Keras model

#Set number of epochs for training
n.epochs <- 50 

#Set mini-batch size. Needs to be roughly a multiple of the training size
batch.size <- 1024 

if(fit_models == T){
history <- model %>% fit(
  list(W,gauge.lb), R.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 5)),
  validation_data=list(list(  input.pseudo.angles=W, gauge.lb_input=gauge.lb),R.valid)
  
)
}

#Extract initial model weights. These will be used for initialising the main neural network 
initial.gauge.weights = model$get_weights()


# Estimate the gauge function with the pre-trained weights  ---------------

#Get the lower bound/conditioning threshold for R, i.e., the quantile estimate
#We fit the truncated gamma above this level 
pred.quant = k_get_value(quant.model(k_constant(W))) 
r.lb <- pred.quant 

#Find and use only the exceedances of R above the predicted quantile
exceed.inds=which(R > pred.quant)
#We set any non-exceeding observations to a very negative value so they are ignored by the neural network 
R[-exceed.inds]=-1e10 

#Select 20% of the data as a validation set
valid.inds=sample(1:n,round(n/5))

#Define training and validation sets in a similar manner to before
R.train<-R.valid<-R
R.train[valid.inds]=-1e10
R.valid[-valid.inds]=-1e10

#Change from vectors to matrices. This is required by Keras 
dim(R)=c(length(R),1); dim(R.train)=c(length(R.train),1); dim(R.valid)=c(length(R.valid),1)
dim(r.lb)=c(length(r.lb),1)

#We build the Keras model. Again we impose conditions to ensure the lower bound of the gauge function is respected 

#Alongside the gauge function, we also have to estimate a constant shape parameter for the truncated gamma distribution  
#Set initial alpha estimate to d - this is the theoretical asymptotic value for most cases.  
init_alpha=d

#Input additional layers. Such layers ensure the estimated limit set has componentwise max and min equal to 1 and -1 (resp.)
source("new_layers.R")					  

#Specify input layers
#We have three inputs; the angles, the lower bound of the gauge, and the lower bound/conditioning threshold of R. 

#Define input layers for angles 
input.pseudo.angles <- layer_input(shape = d, name = 'input.pseudo.angles')

#Define input layers for the lower bound on g and r 
input.rlb <- layer_input(shape = dim(r.lb)[2], name = 'r.lb_input') #This will features in the loss function only.

#Define the input later for the lower bound on the gauge function
input.glb <- layer_g_lb(input.pseudo.angles,input_dim=d)

#Define a layer for the alpha parameter
#Although this layer takes in the angle input, the first layer is constructed so that it always returns 1.
#The second layer is then a single trainable weight which will determine the value of alpha. 
#Exponential activation used to ensure strict positivity
alphaBranch <- input.pseudo.angles %>% layer_dense(units = 1 ,activation = 'relu', trainable=F,
                                                   weights=list(matrix(0,nrow=d,ncol=1),
                                                                array(1,dim=c(1))), 
                                                   name = 'alpha_dense') %>%
                                        layer_dense(units = 1 ,activation = 'exponential',
                                                    use_bias = F,
                                                    weights=list(matrix(log(init_alpha),nrow=1,ncol=1)),
                                                    name = 'alpha_activation',trainable=T)


#For the gauge function, we define a ReLU neural network with exponential activation in the final layer (to ensure that the estimated gauge function is strictly positive)
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

gBranch  <- layer_add(g.xBranch,input.glb) #Add lower bound to output to ensure validity 

g.model <- keras_model(
  inputs = c(input.pseudo.angles), 
  outputs = gBranch
)

#We first estimate the neural network with rescaling just on the observed angles.  
#This may not result in the componentwise max and min of the limit set equalling exactly 1/-1, but it is a quick (additional) pre-training step
#We then use the obtained weights to perform further optimisation using the dense angular grid for rescaling. 

#Create supplement matrix  
W.supplement = W

#The lines below detail additional layers that to ensure the limit set has the correct componentwise max and min
hw_coordmax_branch <- layer_compute_max(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)
hw_coordmin_branch <- layer_compute_min(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)

Wtransbranch <- layer_inverse_angular_transform(input_dim=d)(input.pseudo.angles,hw_coordmax_branch,hw_coordmin_branch)
gtransBranch = g.model(Wtransbranch)
dGtransbranch <- layer_multiply(1/gtransBranch, Wtransbranch)

adjusted_dGtrans_branch <- layer_adjust(input_dim=d)(dGtransbranch,hw_coordmax_branch,hw_coordmin_branch)

adjusted.g.branch <- layer_limitset_to_gauge(adjusted_dGtrans_branch, input_dim=d)

#Compiling the model

#Define output of Keras model. We concatenate the three components required to evaluate the loss function.
output <- layer_concatenate(c(alphaBranch,adjusted.g.branch,input.rlb))

#Construct the Keras model
model2 <- keras_model(
  inputs = c(input.pseudo.angles,input.rlb),
  outputs = output
)
summary(model2)

#Compile the model with the adam optimiser. 
#The loss function 'truncGamma_nll' is stored in the preamble.R file
model2 %>% compile(
    optimizer=optimizer_adam(learning_rate=0.001),
    loss = truncGamma_nll,
    run_eagerly=T
  )

checkpoint <- callback_model_checkpoint(filepath=paste0("Gauge_est/gauge_fit_",site_num), monitor = "val_loss", verbose = 0,
                                        save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                        save_freq = "epoch")

#Train the model

#Set number of epochs for training
n.epochs <- 50

#Set mini-batch size. Needs to be roughly a multiple of the training size
batch.size <- 4096

if(fit_models == T){
history <- model2 %>% fit(
  list(W,r.lb), R.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(checkpoint,callback_early_stopping(monitor = "val_loss",
                                                   min_delta = 0, patience = 5)),
  validation_data=list(list(  input.pseudo.angles=W, r.lb_input = r.lb),R.valid)

)
}

#Load the best fitting model from the checkpoint save
model2 <- load_model_weights_tf(model2,filepath=paste0("Gauge_est/gauge_fit_",site_num))

#We now alter the W.supplement to be the dense hypersphere sample. This additional step makes sure the estimated limit set has exactly 1 and -1 for componentwise max and min
W.supplement = sphere_sample

#Adjust additional layers for dense hypersphere sample 
hw_coordmax_branch <- layer_compute_max(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)
hw_coordmin_branch <- layer_compute_min(input.pseudo.angles, input_dim=d, W.supplement = W.supplement, g.model = g.model)

Wtransbranch <- layer_inverse_angular_transform(input_dim=d)(input.pseudo.angles,hw_coordmax_branch,hw_coordmin_branch)
gtransBranch = g.model(Wtransbranch)
dGtransbranch <- layer_multiply(1/gtransBranch, Wtransbranch)

adjusted_dGtrans_branch <- layer_adjust(input_dim=d)(dGtransbranch,hw_coordmax_branch,hw_coordmin_branch)
adjusted.g.branch <- layer_limitset_to_gauge(adjusted_dGtrans_branch, input_dim=d)

#Set number of epochs for training
n.epochs <- 50

#Set mini-batch size. Needs to be roughly a multiple of the training size
batch.size <- 4096 

#Define output of Keras model. We concatenate the three components required to evaluate the loss function.
output <- layer_concatenate(c(alphaBranch,adjusted.g.branch,input.rlb)) 

#Construct the Keras model
model2 <- keras_model(
  inputs = c(input.pseudo.angles,input.rlb), 
  outputs = output
)
summary(model2)

#Compile the model with the adam optimiser
model2 %>% compile(
  optimizer=optimizer_adam(learning_rate=0.001),
  loss = truncGamma_nll,
  run_eagerly=T
)

if(fit_models == T){
history <- model2 %>% fit(
  list(W,r.lb), R.train,
  epochs = n.epochs, batch_size = batch.size,
  callback=list(checkpoint,callback_early_stopping(monitor = "val_loss", 
                                                   min_delta = 0, patience = 6)),
  validation_data=list(list(  input.pseudo.angles=W,  r.lb_input = r.lb),R.valid)
  
)		
}

#Load the best fitting model from the checkpoint save
model2 <- load_model_weights_tf(model2,filepath=paste0("Gauge_est/gauge_fit_",site_num))

gauge_model <- model2
				 
#Compute the gauge function for each point on the dense hypersphere sample  
pred.gauge.hypersphere =k_get_value(gauge_model(list(k_constant((sphere_sample)),
                                                    k_constant(as.matrix(rep(1,nrow(sphere_sample)))))))[,2]
#Compute the limit set at all angles, then evaluate the componentwise maxima and minima
#These should be approx. equal to 1 and -1. Note this will not be perfect  
print(apply(sphere_sample/pred.gauge.hypersphere,2,max))
print(apply(sphere_sample/pred.gauge.hypersphere,2,min))

# Compute a range of visual diagnostics  ----------------------------------------------------

#Load in the best fitting quantile and gauge function models 
quant.model <- load_model_tf(paste0("QR_est/best_qr_fit_",site_num),custom_objects = list("tilted_loss" = tilted_loss))

gauge_model <- load_model_weights_tf(model2,filepath=paste0("Gauge_est/gauge_fit_",site_num))

#Evaluate quantile function at observed angles 
pred.quant = k_get_value(quant.model(k_constant(W))) 

#Compute threshold exceeding observations  
exceed.inds=which(R > pred.quant)
R <- R[exceed.inds]; W <- W[exceed.inds,]

#Get the lower bound for R, i.e., the quant.level quantile estimate
r.lb <- pred.quant[exceed.inds] #Only for exceedances R > pred.quant

#Set the dimesions for the lower bound vector - for Keras 
dim(r.lb)=c(length(r.lb),1)

#Get the QQ plot associated with truncated Gamma distributions

#Get gauge estimates from observed data
predictions = k_get_value(gauge_model(list(k_constant(W),k_constant(r.lb)))) 

#First column of predictions gives the alpha estimate. 
pred.alpha=predictions[1,1]

#Second column of predictions gives the gauge estimate. 
pred.gauge=predictions[,2]

#Transform the observations to the radial scale using the probability integral transform
unif_exceedances =  exp(pgamma(R,shape = pred.alpha,rate = pred.gauge,lower.tail = F,log.p=T)-pgamma(r.lb[,1],shape = pred.alpha,rate = pred.gauge,lower.tail = F,log.p=T))

#Transform the the observations to the standard exponential scale 
exp_exceedances = qexp(unif_exceedances,lower.tail=F)

#Select how many quantiles to assess 
m = 1000

#Compute empirical/observed quantiles 
observed_quants = quantile(exp_exceedances, probs=(1:m)/(m+1))

#Compute theoretical/model quantiles 
theoretical_quants = qexp((1:m)/(m+1))

#Save the plot 
pdf(file=paste0("Diagnostics/qqplot_",site_num,".pdf"),width=4,height=4)

#Set plotting parameters
par(mfrow=c(1,1),mgp=c(2.25,0.75,0),mar=c(4,4,1,1))

#Plot empirical against model quantiles 
plot(theoretical_quants,observed_quants,xlim=range(theoretical_quants,observed_quants),
     ylim=range(theoretical_quants,observed_quants),pch=16,col=1,ylab="Empirical",xlab="Model",
     cex.lab=1.3, cex.axis=1.2,cex.main=1.8, cex=0.5)
abline(a=0,b=1,lwd=3,col=2)
points(theoretical_quants,observed_quants,pch=16,col=1, cex=0.5)

dev.off()

#Compute lower dimensional limit sets for each subvector 
#This involves a minimisation scheme with respect to any indices not within the subvector

#Select points on the unit circle at which we wish to evaluate gauge function 
phi <- seq(0,2*pi,len=151)

unit_circle <- cbind(cos(phi),sin(phi))

#Functional for performing minimisation procedure 
two_dim_gauge <- function(w2,subvec,d){
  
  n_grid = 4001
  
  lap_grid = seq(-20,20,length.out=n_grid)
  
  lap_mat = matrix(NA,ncol=d,nrow=n_grid)
  
  lap_mat[,subvec] = rep(w2,each=n_grid)
  
  lap_mat[,-subvec] = lap_grid
  
  r_mat = apply(lap_mat,1,l2_norm)
  
  w_mat = lap_mat/r_mat
  
  pred_quant_temp = k_get_value(quant.model(k_constant(w_mat))) 
  
  gauge_mat = r_mat*k_get_value(gauge_model(list(k_constant(w_mat),k_constant(pred_quant_temp))))[,2]
  
  min_index = which.min(gauge_mat)
  
  min_vec = rep(NA,3)
  
  min_vec[subvec] = w2
  
  min_vec[-subvec] = lap_grid[min_index]
  
  r_min = l2_norm(min_vec)
  
  w_min = min_vec/r_min
  
  dim(w_min) = c(1,d)
  
  r_lb_min = k_constant(k_get_value(quant.model(k_constant(w_min))))
  
  g_min = k_get_value(gauge_model(list(k_constant(w_min),r_lb_min)))[,2]
  
  return(g_min)
  
}

#Save plot
pdf(file=paste0("Diagnostics/twodim_limitsets_",site_num,".pdf"),width=12,height=4)

#Plotting parameters
par(mfrow=c(1,3),mgp=c(2.25,0.75,0),mar=c(4,4,1,1))

subvec <- c(1,2)

gauge_subvec <- c()

for(i in 1:nrow(unit_circle)){
  gauge_subvec[i] <- two_dim_gauge(w2 = unit_circle[i,],subvec = subvec,d=d)
}

boundary_set_subvec <- unit_circle/gauge_subvec

plot(boundary_set_subvec ,xlab="hs (Laplace)",ylab="ws (Laplace)",
     type="l",col=2,ylim=c(-1,1),xlim=c(-1,1),lwd=4,cex.lab=1.5, cex.axis=1.2,cex.main=1.5, cex=0.5)

points((data_lap/log(n/2))[,subvec],pch=16,col="grey", cex=0.5)															   
rect(-1,-1,1,1,lwd=4,lty=2,col=NULL)

subvec <- c(1,d)

gauge_subvec <- c()

for(i in 1:nrow(unit_circle)){
  gauge_subvec[i] <- two_dim_gauge(w2 = unit_circle[i,],subvec = subvec,d=d)
}

boundary_set_subvec <- unit_circle/gauge_subvec

plot(boundary_set_subvec ,xlab="hs (Laplace)",ylab="mslp (Laplace)",
     type="l",col=2,ylim=c(-1,1),xlim=c(-1,1),lwd=4,cex.lab=1.5, cex.axis=1.2,cex.main=1.5, cex=0.5)

points((data_lap/log(n/2))[,subvec],pch=16,col="grey", cex=0.5)															   
rect(-1,-1,1,1,lwd=4,lty=2,col=NULL)

subvec <- c(d-1,d)

gauge_subvec <- c()

for(i in 1:nrow(unit_circle)){
  gauge_subvec[i] <- two_dim_gauge(w2 = unit_circle[i,],subvec = subvec,d=d)
}

boundary_set_subvec <- unit_circle/gauge_subvec

plot(boundary_set_subvec ,xlab="ws (Laplace)",ylab="mslp (Laplace)",
     type="l",col=2,ylim=c(-1,1),xlim=c(-1,1),lwd=4,cex.lab=1.5, cex.axis=1.2,cex.main=1.5, cex=0.5)

points((data_lap/log(n/2))[,subvec],pch=16,col="grey", cex=0.5)															   
rect(-1,-1,1,1,lwd=4,lty=2,col=NULL)

dev.off()

#Return level set probabilities diagnostic

#First we reset W and R to be all observations 
R=polar$r
W=data_lap/polar$r 

#Evaluate quantile function at observed angles 
pred.quant = k_get_value(quant.model(k_constant(W))) 

#Get gauge estimates from observed data
predictions = k_get_value(gauge_model(list(k_constant(W),k_constant(pred.quant)))) 

#First column of predictions gives the alpha estimate. 
pred.alpha=predictions[1,1]

#Second column of predictions gives the gauge estimate. 
pred.gauge=predictions[,2]

#Probabilities at which to evaluate return level sets
probs = exp(seq(log(0.9),log(0.999),length.out=100))

#Corresponding probabilities for the truncated gamma model 
trunc_probs = (probs - quant.level)/(1-quant.level)

empirical_probs_function = function(tp){
  radial_quants = qgamma(p = pgamma(pred.quant[,1],shape = pred.alpha,rate = pred.gauge,lower.tail = F)*tp + pgamma(pred.quant[,1],shape = pred.alpha,rate = pred.gauge,lower.tail = T) , shape = pred.alpha, rate = pred.gauge )
  
  return(mean(R<=radial_quants))
  
}

empirical_probs = sapply(trunc_probs,empirical_probs_function)

pdf(file=paste0("Diagnostics/ret_level_set_probs_",site_num,".pdf"),width=4,height=4)

#Plotting parameters
par(mfrow=c(1,1),mgp=c(2.25,0.75,0),mar=c(4,4,1,1))

plot(probs,empirical_probs,xlim=range(probs,empirical_probs),
     ylim=range(probs,empirical_probs),pch=16,col="grey",
     xlab="Model",ylab="Empirical",cex.lab=1.3, cex.axis=1.2,cex.main=1.8, cex=0.5)
abline(a=0,b=1,lwd=3,col=2)
points(probs,empirical_probs,pch=16,col=1, cex=0.8)

dev.off()

#Plotting the estimated limit set 

#First we create a dense grid of polar/spherical angles - see https://en.wikipedia.org/wiki/N-sphere
ang_num = 150

pred_angles = expand.grid(c(rep(list(seq(0,pi,length.out=ang_num)), (d-2)  ),list(seq(0,2*pi,length.out=ang_num))))

#Find the corresponding points on the hypersphere 
hypersphere = t(polar2rect(r=rep(1,(dim(pred_angles)[1])),phi = t(pred_angles)))

#Compute the corresponding quantile values 
r.lb = k_get_value(quant.model(k_constant(hypersphere))) 

dim(r.lb)=c(length(r.lb),1)

#Get gauge estimates from points on hypersphere
predictions_hypersphere = k_get_value(gauge_model(list(k_constant(hypersphere),k_constant(r.lb)))) 

#Second column of predictions gives the gauge estimate. 
pred.gauge=predictions_hypersphere[,2]

#Compiute the radii of the limit set
gauge_unit_radii = 1/pred.gauge

#Obtain the limi set in Cartsian coordinates
gauge_unit_level = t(polar2rect(r=gauge_unit_radii,phi = t(pred_angles)))

#Plot the limit set in 3D
open3d()
clear3d()
par3d(windowRect = c(20, 30, 1000, 1000))

points3d((data_lap/log(n/2)))
surface3d(x=matrix(gauge_unit_level[,1], nrow=sqrt(dim(hypersphere)[1]), ncol=sqrt(dim(hypersphere)[1])),
          y=matrix(gauge_unit_level[,2], nrow=sqrt(dim(hypersphere)[1]), ncol=sqrt(dim(hypersphere)[1])),
          z=matrix(gauge_unit_level[,3], nrow=sqrt(dim(hypersphere)[1]), ncol=sqrt(dim(hypersphere)[1])), alpha=.3, col="blue")
axes3d(labels = T,expand=1,cex.axis=2)
title3d(xlab = "Hs", ylab = "Ws", zlab = "Mslp",cex=2)

#Create an axis about which the limit set can rotate 
play3d( spin3d( axis = c(0, 0, 1), rpm = 20), duration = 10 )

#Save the rotating limit set as a gif
movie3d(
  movie= paste0("limit_set_spin_",site_num), 
  spin3d( axis = c(0, 0, 1), rpm = 7),
  duration = 10, 
  dir = "Diagnostics",
  type = "gif", 
  clean = TRUE
)

#Simulating data in the joint tail 

#Number of observations in the joint tail we wish to simulate
n.sim = 1e5 

#We sample empirically from the observed angular variable 
W.sample = W[sample(1:nrow(W),n.sim,replace=T),]

unif.sample = runif(n.sim)

#Evaluate quantile function at simulated angles 
pred.quant.sample = k_get_value(quant.model(k_constant(W.sample))) 

#Get gauge estimates from simulated data
predictions.sample = k_get_value(gauge_model(list(k_constant(W.sample),k_constant(pred.quant.sample)))) 

#First column of predictions gives the alpha estimate. 
pred.alpha.sample=predictions.sample[1,1]

#Second column of predictions gives the gauge estimate. 
pred.gauge.sample=predictions.sample[,2]

#Find corresponding radial quantiles from truncated gamma model 
R.sample = c(qgamma(p = pgamma(pred.quant.sample,shape = pred.alpha.sample,rate = pred.gauge.sample,lower.tail = F)*unif.sample + pgamma(pred.quant.sample,shape = pred.alpha.sample, rate = pred.gauge.sample,lower.tail = T) , shape = pred.alpha.sample, rate = pred.gauge.sample ))

#Transform to Cartesian coordinates
X.sample = R.sample*W.sample

#Find the observed data points in the joint tail 
data.lap.tail = data_lap[R > c(pred.quant),]

png(file=paste0("Diagnostics/simulated_data_joint_tail_",site_num,".png"),width=1000,height=1000,res = 120)

labels <- c("Hs","Ws","Mslp")

pairs(rbind(X.sample,data.lap.tail),labels = labels,main="Simulated vs observed in joint tail",pch=16,col=c(rep(rgb(0, 1, 0, alpha = 0.3),nrow(X.sample)),rep("grey",nrow(data.lap.tail))),cex.labels=1.3)

legend("center",legend=c("Observed","Simulated"),col=c("grey",rgb(0, 1, 0, alpha = 0.3)),pch=16,cex=1.2)

dev.off()