#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchTensor cpp_contrib_torch_sparsemax (Rcpp::XPtr<XPtrTorchTensor> input, int dim)
{
  XPtrTorchTensor out = lantern_contrib_torch_sparsemax(input->get(), dim);
  return out;
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_contrib_torch_sort_vertices (XPtrTorchTensor vertices, XPtrTorchTensor mask, XPtrTorchTensor num_valid)
{
  return XPtrTorchTensor(lantern_contrib_sort_vertices(vertices.get(), mask.get(), num_valid.get()));
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_contrib_torch_bias_act (XPtrTorchTensor x, XPtrTorchTensor b, XPtrTorchTensor xref, XPtrTorchTensor yref, XPtrTorchTensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
  return XPtrTorchTensor(lantern_contrib_bias_act(
    x.get(), 
    b.get(), 
    xref.get(), 
    yref.get(), 
    dy.get(), 
    grad, 
    dim,
    act, 
    alpha, 
    gain, 
    clamp)
  );
}