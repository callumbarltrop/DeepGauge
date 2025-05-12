

layer_compute_max <- Layer(
  classname = "ComputeMax",
  initialize = function(input_dim, W.supplement, g.model) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(inputs, ...) {
    gBranch2 = g.model(k_constant(W.supplement))
    hwbranch <- tf$multiply(1/gBranch2, k_constant(W.supplement))
    
    tf$reduce_max(hwbranch, axis = 0L, keepdims = T)  }
)

layer_reduce_max <- Layer(
  classname = "reduceMax",
  initialize = function(input_dim) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(inputs, ...) {
    
    tf$reduce_max(inputs, axis = 0L, keepdims = T)  }
)

layer_reduce_min <- Layer(
  classname = "reduceMin",
  initialize = function(input_dim) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(inputs, ...) {
    
    tf$reduce_min(inputs, axis = 0L, keepdims = T)  }
)


layer_g_lb <- Layer(
  classname = "Computeg_lb",
  initialize = function(input_dim) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(W) {
    
    tf$maximum(tf$reduce_max(W, 1L,keepdims = T),-tf$reduce_min(W, 1L, keepdims=T))
  }
)

layer_compute_min <- Layer(
  classname = "ComputeMin",
  initialize = function(input_dim, W.supplement, g.model) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(inputs, ...) {
    gBranch2 = g.model(k_constant(W.supplement))
    hwbranch <- tf$multiply(1/gBranch2, k_constant(W.supplement))
    
    tf$reduce_min(hwbranch, axis = 0L, keepdims = T)
  }
)

layer_inverse_angular_transform <- Layer(
  classname = "ComputeInverse_transform",
  initialize = function(input_dim) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(W,maxes,mins,...) {
    
    b=tf$sign(tf$maximum(W,0))*maxes-(1-tf$sign(tf$maximum(W,0)))*mins
    d= tf$shape(W)[2]
    W1star=tf$zeros(tf$shape(W))
    ## Need to remove small values <-1e16
    
    nonzero_inds=tf$sign(tf$maximum(tf$abs(W)-k_constant(1e-16),0))
    W=W*nonzero_inds
    ind_last_nonzero=tf$reduce_max(nonzero_inds*tf$range(1,d+1),axis=1L, keepdims = T)
    ind_last_nonzero=tf$cast(ind_last_nonzero,dtype=tf$int32)
    indices=tf$concat(
      c( 
        tf$reshape(tf$range(tf$constant(1L),tf$shape(W)[1]+tf$constant(1L)),tf$shape(ind_last_nonzero)
                   ),ind_last_nonzero), axis=1L)
    
    updates = tf$ones(tf$shape(indices)[1], dtype = tf$int32)
     ind_last_nonzero = tf$scatter_nd(indices-1, updates, tf$cast(tf$shape(W),dtype=tf$int32))

    last_nonzero = tf$reduce_sum(W * tf$cast(ind_last_nonzero, dtype=tf$float32), axis=1L, keepdims = T)
    last_nonzero_b = tf$reduce_sum(b * tf$cast(ind_last_nonzero, dtype=tf$float32), axis=1L, keepdims = T)

    aw=tf$sign(last_nonzero)/(tf$sqrt(tf$reduce_sum(
      (b*W/(last_nonzero_b*last_nonzero))^2, axis=1L, keepdims = T
    )))
    Wstar=aw*b*W/(last_nonzero_b*last_nonzero)
    Wstar
    }
)

layer_adjust <- Layer(
  classname = "ComputeAdjust",
  initialize = function(input_dim) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(dG,maxes,mins,...) {
    nonzero_inds=tf$sign(tf$abs(dG))
    
    max2 = maxes*nonzero_inds+(1-nonzero_inds)
    min2 = mins*nonzero_inds+(1-nonzero_inds)
    
    dG.pos = tf$maximum(dG,0)/max2
    dG.neg = tf$minimum(dG,0)/(-min2)
    
    
    dG.pos+dG.neg
  }
)

layer_limitset_to_gauge <- Layer(
  classname = "Computegauge",
  initialize = function(input_dim) {
    super()$`__init__`()
    self$total <- tf$Variable(
      initial_value = tf$zeros(shape(input_dim)),
      trainable = FALSE
    )
  },
  call = function(dG,...) {
    
    1/tf$sqrt(tf$reduce_sum(dG^2, axis=1L, keepdims=T))
  }
)
