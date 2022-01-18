.as_list <- function(inp, arg_name = NULL, fn_name = NULL) {
  # Ensures that inp is a tuple of Tensors
  # Returns whether or not the original inp was a tuple and the tupled version of the input
  if(is.null(arg_name) & is.null(fn_name)) {
    return(.as_list_nocheck(inp))
  }

  is_inp_list <- TRUE
  if(!is.list(inp)) {
    inp <- list(inp)
    is_inp_list <- FALSE
  }
  
  for(i in seq_along(inp)) {
    if(is_torch_tensor())
  }

  for i, el in enumerate(inp):
    if not isinstance(el, torch.Tensor):
      if is_inp_tuple:
        raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                        " value at index {} has type {}.".format(arg_name, fn_name, i, type(el)))
      else:
        raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                        " given {} has type {}.".format(arg_name, fn_name, arg_name, type(el)))

  return(list(is_inp_tuple, inp))
}

grad_preprocess <- function(inputs, create_graph, need_graph) {}
  # Preprocess the inputs to make sure they require gradient
  # inputs is a tuple of Tensors to preprocess
  # create_graph specifies if the user wants gradients to flow back to the Tensors in inputs
  # need_graph specifies if we internally want gradients to flow back to the Tensors in res
  # Note that we *always* create a new Tensor object to be able to see the difference between
  # inputs given as arguments and the same Tensors automatically captured by the user function.
  # Check this issue for more details on how that can happen: https://github.com/pytorch/pytorch/issues/32576
  res <- list()
  i <- 0
  for(inp in inputs){
    i <- i + 1
    if(create_graph & inp$requires_grad) {
      # Create at least a new Tensor object in a differentiable way
      if(!is_sparse(inp)) {
      # Use .view_as() to get a shallow copy
        res[[i]] <- inp$view_as(inp)
      } else {
        # We cannot use view for sparse Tensors so we clone
        res[[i]] <- inp$clone()
      }
    } else {
      res[[i]] <- inp$detach()$requires_grad_(need_graph)
    }
  return(res)
}
  
fill_in_zeros <- function(grads, refs, strict, create_graph, stage) {
  # Used to detect None in the grads and depending on the flags, either replace them
  # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
  # strict and create graph allow us to detect when it is appropriate to raise an error
  # stage gives us information of which backward call we consider to give good error message
  if(!stage %in% c("back", "back_trick", "double_back", "double_back_trick")) {
    value_error("Invalid stage argument '{stage}' to fill_in_zeros")
  }
  
  res <- list() 
  for(i in seq_along(grads)) {
    grads_i <- grads[[i]]
    if(is.null(grads_i)) {
      if(strict) {
        switch(stage,
          back = runtime_error("The output of the user-provided function is independent of ",
                               "input {i}. This is not allowed in strict mode."),
          back_trick = runtime_error("The gradient with respect to the input is independent of entry {i}",
                                     " in the grad_outputs when using the double backward trick to compute",
                                     " forward mode gradients. This is not allowed in strict mode."),
          double_back = runtime_error("The jacobian of the user-provided function is independent of ",
                                      "input {i}. This is not allowed in strict mode."),
          double_back_trick = runtime_error("The hessian of the user-provided function is independent of ",
                                            "entry {i} in the grad_jacobian. This is not allowed in strict ",
                                            "mode as it prevents from using the double backward trick to ",
                                            "replace forward mode AD.")
        )
      }  
      grads_i <- torch_zeros_like(refs[[i]])
    } else {
      if(strict & create_graph & !grads_i$requires_grad) {
        if(!"double" %in% stage) {
          runtime_error("The jacobian of the user-provided function is independent of ",
                        "input {i}. This is not allowed in strict mode when create_graph = TRUE.")
        } else {
          runtime_error("The hessian of the user-provided function is independent of ",
                        "input {i}. This is not allowed in strict mode when create_graph = TRUE.")
        }
      }
    }
    
    res[[i]] <- grads_i
  }
  return(res)
}
  
.autograd_grad <- function(outputs, inputs, 
                           grad_outputs = NULL, create_graph = FALSE, 
                           retain_graph = NULL) {
  # Note: removed is_grads_batched as an argument. This appears to be an (undocumented?) argument
  # to torch.autograd.grad in Python (https://github.com/pytorch/pytorch/blob/2faccc2f5d75346a50c974898ea12362672ae757/torch/autograd/__init__.py#L185), 
  # but definitely is not an argument in torch::autograd_grad
  # Version of autograd_grad that accepts `NULL` in outputs and do not compute gradients for them.
  # This has the extra constraint that inputs has to be a list
  stopifnot(is.list(outputs))
  if(is.null(grad_outputs)) {
    grad_outputs <- replicate(length(outputs), NULL)
  }
  stopifnot(is.list(grad_outputs)) 
  stopifnot(length(outputs) == length(grad_outputs))
  
  new_outputs <- list()
  new_grad_outputs <- list()
  
  index <- 0
  for(i in seq_along(outputs)) {
    if(!is.null(outputs) & outputs$requires_grad) {
      index <- index + 1
      new_outputs[[index]] <- outputs[[i]]
      new_grad_outputs[[index]] <- grad_outputs[[i]]
    }
  }
  
  if(length(new_outputs) == 0) {
    # No differentiable output, we don't need to call the autograd engine
    return(replicate(length(inputs), NULL))
  } else {
    return(autograd_grad(new_outputs, inputs, new_grad_outputs, allow_unused=TRUE,
                               create_graph=create_graph, retain_graph=retain_graph))
  }
}

#' Function that computes the dot product between a vector `v` and the
#' Jacobian of the given function at the point given by the inputs.
#'
#' @param func an R function that takes `torch_tensor` inputs and returns
#'             a list of `torch_tensor`s or a `torch_tensor`.
#' @param inputs inputs to the function `func` (a `torch_tensor` or list of `torch_tensor`s).
#' @param v The vector (a `torch_tensor` or list of `torch_tensor`s) for which the vector Jacobian 
#' product is computed. Must be the same size as the output of `func`. This argument is optional 
#' when the output of `func` contains a single element and (if it is not provided) will be set as 
#' a `torch_tensor` containing a single `1`.
#' @param create_graph If `TRUE`, both the output and result will be computed in a differentiable 
#' way. Note that when `strict` is `FALSE`, the result can not require gradients or be
#' disconnected from the inputs.  Defaults to `FALSE`.
#' @param strict If `TRUE`, an error will be raised when we detect that there exists an input 
#' such that all the outputs are independent of it. If `FALSE`, we return a `torch_tensor` of 
#' zeros as the vjp for said inputs, which is the expected mathematical value. Defaults to `FALSE`.
#'
#' @return A list with:
#' * func_output: output of `func(inputs)`
#' * vjp: result of the dot product with the same shape as the inputs.
#' @export
#'
#' @examples
#' exp_reducer <- function(x) {
#'   return(x$exp()$sum(dim = 2))
#' }
#' inputs <- torch_rand(4, 4)
#' v <- torch_ones(4)
#' vjp(exp_reducer, inputs, v)
#' vjp(exp_reducer, inputs, v, create_graph = TRUE)
#' adder <- function(x, y) {
#'   return(2 * x + 3 * y)
#' }
#' inputs <- list(torch_rand(2), torch_rand(2))
#' v <- torch.ones(2)
#' vjp(adder, inputs, v)
vjp <- function(func, inputs, v = NULL, create_graph = FALSE, strict = FALSE) {
  
  with_enable_grad({
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vjp")
    _check_requires_grad(outputs, "outputs", strict=strict)

    if v is not None:
      _, v = _as_tuple(v, "v", "vjp")
      v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
      _validate_v(v, outputs, is_outputs_tuple)
    else:
      if len(outputs) != 1 or outputs[0].nelement() != 1:
        raise RuntimeError("The vector v can only be None if the "
                           "user-provided function returns "
                           "a single Tensor with a single element.")
  })
        

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    return(list_postprocess(outputs, is_outputs_tuple), list_postprocess(vjp, is_inputs_tuple))
}