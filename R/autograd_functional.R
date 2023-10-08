.as_list <- function(inp, arg_name = NULL, fn_name = NULL) {
  # Ensures that inp is a list of Tensors
  # Returns whether or not the original inp was a list and the listed version of the input
  if(is.null(arg_name) & is.null(fn_name)) {
    return(inp)
  }

  is_inp_list <- TRUE
  if(!is.list(inp)) {
    inp <- list(inp)
    is_inp_list <- FALSE
  }
  
  for(i in seq_along(inp)) {
    el <- inp[[i]]
    if(!is_torch_tensor(el)) {
      if(is_inp_list) {
        type_error("The {arg_name} given to {fn_name} must be either a torch_tensor or a list of torch_tensors but the",
                   " value at index {i} has class {class(el)}.")
      } else {
        type_error("The {arg_name} given to {fn_name} must be either a torch_tensor or a list of torch_tensors but the",
                   " given {arg_name} has class {class(el)}.")
      }
    }
  }
  
  return(list(is_inp_list, inp))

}

list_postprocess <- function(res, to_unpack) {
    # Unpacks a potentially nested list of Tensors
    # to_unpack should be a single boolean or a vector of two booleans.
    # It is used to:
    # - invert .as_list when res should match the inp given to .as_list
    # - optionally remove nesting of two lists created by multiple calls to .as_list
    if(length(to_unpack) > 1) {
        stopifnot(length(to_unpack) == 2)
        if(!to_unpack[2]) {
            res <- lapply(res, function(x) x[[1]]) 
        }
        if(!to_unpack[1]) {
            res <- res[[1]]
        }
    } else {
        if(!to_unpack) {
            res <- res[[1]]
        }
    }
    return(res)
}

check_requires_grad <- function(inputs, input_type, strict) {
  # Used to make all the necessary checks to raise nice errors in strict mode.
  if(!strict) {
    return(invisible(NULL))
  }

  if(!input_type %in% c("outputs", "grad_inputs", "jacobian", "hessian")) {
      runtime_error("Invalid input_type to check_requires_grad")
  }
  for(i in seq_along(inputs)) {
    inp <- inputs[[i]]
    if(is.null(inp)) {
      runtime_error("The output of the user-provided function is independent of input {i}.",
                             " This is not allowed in strict mode.")
    }
    if(!inp$requires_grad) {
      switch(input_type,
             hessian = runtime_error("The hessian of the user-provided function with respect to input {i}",
                                 " is independent of the input. This is not allowed in strict mode.",
                                 " You should ensure that your function is thrice differentiable and that",
                                 " the hessian depends on the inputs."),
             jacobian = runtime_error("While computing the hessian, found that the jacobian of the user-provided",
                                 " function with respect to input {i} is independent of the input. This is not",
                                 " allowed in strict mode. You should ensure that your function is twice",
                                 " differentiable and that the jacobian depends on the inputs (this would be",
                                 " violated by a linear function for example)."),
             grad_inputs = runtime_error("The gradient with respect to input {i} is independent of the inputs of the",
                                 " user-provided function. This is not allowed in strict mode."),
             outputs = runtime_error("Output {i} of the user-provided function does not require gradients.",
                                 " The outputs must be computed in a differentiable manner from the input",
                                 " when running in strict mode."))
    }
  }
}

validate_v <- function(v, other, is_other_list) {
    # This assumes that other is the correct shape, and v should match
    # Both are assumed to be lists of Tensors
  if(length(other) != length(v)) {
    if(is_other_list) {
      runtime_error("v is a list of invalid length: should be {length(other)} but got {length(v)}.")
    } else {
      runtime_error("The given v should contain a single torch_tensor.")
    }
  }
  
  for(i in seq_along(v)) {
    el_v <- v[[i]]
    el_other <- other[[i]]
    if(el_v$size() != el_other$size()) {
      prepend = ""
      if(is_other_list) {
        prepend = "Entry {i} in "
      }
      runtime_error(prepend,
                    "v has invalid size: should be {el_other$size()} but got {el_v$size()}.")
    }
  }
}

grad_preprocess <- function(inputs, create_graph, need_graph) {
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
      if(!inp$is_sparse()) {
      # Use .view_as() to get a shallow copy
        res[[i]] <- inp$view_as(inp)
      } else {
        # We cannot use view for sparse Tensors so we clone
        res[[i]] <- inp$clone()
      }
    } else {
      res[[i]] <- inp$detach()$requires_grad_(need_graph)
    }
  }
  return(res)
}

grad_postprocess <- function(inputs, create_graph) {
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
    # request it.
    if(is_torch_tensor(inputs[[1]])) {
        if(!create_graph) {
          return(lapply(inputs, function(x) x$detach()))
        } else {
          return(inputs)
        }
    } else {
        return(lapply(inputs, grad_postprocess, create_graph = create_graph))
    }
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
                           retain_graph = create_graph) {
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
    if(!is.null(outputs[[i]]) & outputs[[i]]$requires_grad) {
      index <- index + 1
      new_outputs[[index]] <- outputs[[i]]
      new_grad_outputs[[index]] <- grad_outputs[[i]]
    }
  }
  
  if(length(new_outputs) == 0) {
    # No differentiable output, we don't need to call the autograd engine
    return(replicate(length(inputs), NULL))
  } else {
    return(autograd_grad(new_outputs, inputs, new_grad_outputs, allow_unused = TRUE,
                               create_graph = create_graph, retain_graph = retain_graph))
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
#' autograd_vjp(exp_reducer, inputs, v)
#' autograd_vjp(exp_reducer, inputs, v, create_graph = TRUE)
#' adder <- function(x, y) {
#'   return(2 * x + 3 * y)
#' }
#' inputs <- list(torch_rand(2), torch_rand(2))
#' v <- torch.ones(2)
#' autograd_vjp(adder, inputs, v)
autograd_vjp <- function(func, inputs, v = NULL, create_graph = FALSE, strict = FALSE) {
  
  with_enable_grad({
    inp_list <- .as_list(inputs, "inputs", "autograd_vjp")
    is_inputs_list <- inp_list[[1]]
    inputs <- inp_list[[2]]
    inputs <- grad_preprocess(inputs, create_graph = create_graph, need_graph = TRUE)

    outputs <- do.call(func, inputs)
    out_list <- .as_list(outputs, "outputs of the user-provided function", "autograd_vjp")
    is_outputs_list <- out_list[[1]] 
    outputs <- out_list[[2]]
    check_requires_grad(outputs, "outputs", strict = strict)

    if(!is.null(v)) {
      v <- .as_list(v, "v", "autograd_vjp")[[2]]
      v <- grad_preprocess(v, create_graph = create_graph, need_graph = FALSE)
      validate_v(v, outputs, is_outputs_list)
    } else {
      if(length(outputs) != 1 | outputs[[1]]$numel() != 1) {
        runtime_error("The vector v can only be NULL if the ",
                           "user-provided function returns ",
                           "a single torch_tensor with a single element.")
      }
    }
      
  })
        
  if(create_graph) {
    enable_grad <- TRUE
  } else {
    enable_grad <- is_grad_enabled()
  }
 
  with_set_grad_mode(enable_grad, {
    grad_res <- .autograd_grad(outputs, inputs, v, create_graph = create_graph)
    vjp <- fill_in_zeros(grad_res, inputs, strict, create_graph, "back")
  })

  # Cleanup objects and return them to the user
  outputs <- grad_postprocess(outputs, create_graph)
  vjp <- grad_postprocess(vjp, create_graph)

  return(list(func_output = list_postprocess(outputs, is_outputs_list), 
              vjp = list_postprocess(vjp, is_inputs_list)))
}