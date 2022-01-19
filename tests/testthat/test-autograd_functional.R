test_that("autograd_vjp errors correctly", {
  foo <- function(a){
    return(3 * a$narrow(1, 1, 4))
  }
  bar <- function(a) {
    return(list(3 * a$narrow(1, 1, 4), "bar"))
  }
  inp <- torch_rand(4)
  v <- torch$ones(3)
  expect_error(res <- autograd_vjp(foo, list(inp, 2), v), 
               "The inputs given to autograd_vjp must be either a torch_tensor")

  expect_error(res <- autograd_vjp(bar, inp, v), 
               "The outputs of the user-provided function given to autograd_vjp must")

  expect_error(res <- autograd_vjp(foo, inp), 
               "The vector v can only be NULL if the user-provided function returns")

  expect_error(res <- autograd_vjp(foo, inp, list(torch_ones_like(inp), torch_ones_like(inp))), 
               "The given v should contain a single torch_tensor")


  expect_error(res <- autograd_vjp(foo, inp, v[1:2]), "v has invalid size: should be")

  res <- autograd_vjp(foo, inp, v)[[2]]
  expect_identical(res$size(), inp$size())
  
})
