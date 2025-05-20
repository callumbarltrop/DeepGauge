# deepGauge
 
## Tensorflow Installation 

```r
py_version <- "3.8.10"
path_to_python <- reticulate::install_python(version=py_version)

#Create a virtual envionment 'myenv' with Python 3.8.10. Install tensorflow  within this environment.
reticulate::virtualenv_create(envname = 'myenv',
                              python=path_to_python,
                              version=py_version)

path<- paste0(reticulate::virtualenv_root(),"/myenv/bin/python")
Sys.setenv(RETICULATE_PYTHON = path) #Set Python interpreter to that installed in myenv

tf_version="2.11.0" 
reticulate::use_virtualenv("myenv", required = T)
tensorflow::install_tensorflow(method="virtualenv", envname="myenv",
                               version=tf_version) #Install version of tensorflow in virtual environment
keras::install_keras(method = c("virtualenv"), envname = "myenv",version=tf_version) #Install keras

keras::is_keras_available() #Check if keras is available

```