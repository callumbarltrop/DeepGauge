# deepGaugePublic -- example code for fitting the deepGauge framework of Murphy-Barltrop et al. (2024)

In order to use the deepGauge framework via R, you first need to download and install Tensorflow and Keras 2 on your workstation. This involves several steps which we detail below.

## Instructions 

First, make sure git is installed on your computer. Instructions can be found here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git 

Next, make sure you have installed the `reticulate`, `keras`, and `tensorflow` packages on R using the following commands: 

```r
packages = c("keras","tensorflow","reticulate")
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)
```

After installing, restart R. Please restart after every step in this installation process. We then use Reticulate to install the correct version of Python via following commands:
 
```r
py_version <- "3.10.17"
path_to_python <- reticulate::install_python(version=py_version)
```
 
We then create a virtual environment called 'deepGauge_env', which we will use when running Keras.

```r
reticulate::virtualenv_create(envname = 'deepGauge_env',
                              python=path_to_python,
                              version=py_version)
```
 
Restart R, then try loading in 'deepGauge_env' using the following command: 

```r
reticulate::use_virtualenv("deepGauge_env", required = T)
```

If this does not work, one can instead use a conda environment; this may be a requirement of some linux systems. In this case, replace `use_virtualenv` with `use_condaenv' throughout.

Next, we install tensorflow v2.11.0 and keras within the virtual environment. Note that your R session will automatically restart after installation.

```r
tf_version="2.11.0" 
reticulate::use_virtualenv("deepGauge_env", required = T)
tensorflow::install_tensorflow(method="virtualenv", envname="deepGauge_env",
                               version=tf_version) #Install version of tensorflow in virtual environment
```

```r
keras::install_keras(method = c("virtualenv"), envname = "deepGauge_env",version=tf_version) #Install keras
```

Again, restart R. Finally, check if R can access Keras, and that the installation has completed without any problems

```r
reticulate::use_virtualenv("deepGauge_env", required = T)
keras::is_keras_available() #Check if keras is available
```

If there are any problems, we recommend always restarting the R session and making sure the correct packages are loaded in. Moreover, ChatGPT (or any other LLM) can be a very useful tool for debugging such problems - just copy and paste the code along with any error messages.

## Descriptions of the provided R scripts and folders  

A brief description of each of the provided R scripts is given below

* **case_study.R** - this file demonstrates how to fit the deepGauge framework to one of the metocean datasets considered in the Case Study of Murphy-Barltrop et al. (2024).  
* **new_layers.R** - this file contains additional neural network layers required for imposing theoretical properties on the limit set.
* **preamble.R** - this file contains a range of functions required for fitting the model.

The repo also contains the following folders

* **Datafiles** - this folder contains all of required datafiles for running the case_study.R file. 
* **QR_est** - this folder provides a place for storing the weights associated with the quantile regression model.
* **Gauge_est** - this folder provides a place for storing the weights associated with the gauge function model.
* **Diagnostics** - this folder is where the visual diagnostics from case_study.R are stored. 

## Questions?

Please get in touch if you have any questions, or if you find a bug in the code. My email is callum.murphy-barltrop[at]tu-dresden.de 

### References

Murphy-Barltrop, C. J., Majumder, R., & Richards, J. (2024). Deep learning of multivariate extremes via a geometric representation. arXiv preprint arXiv:2406.19936.
