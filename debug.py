import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching




def test_weighted_sum_gradcheck(fn):
    print("WEIGHTED SUM CHECK")
    input_data = torch.randn(5, 10, 3, requires_grad=True)
    weights_data = torch.randn(5, 10, 1, requires_grad=True)

    weighted_sum_custom = fn(weights_data, input_data)
    weighted_sum_custom.backward(torch.ones_like(weighted_sum_custom))

    # Compute the weighted sum using PyTorch built-in functions
    weighted_sum_torch = torch.sum(input_data * weights_data, dim=1)


    # Compute gradients with respect to the weighted sum
    gradient_weighted_sum_torch = torch.autograd.grad(weighted_sum_torch, (weights_data, input_data),
                                                      grad_outputs=torch.ones_like(weighted_sum_torch))

    #print("Analytical gradient with respect to input:", gradient_weighted_sum_torch[0])
    #print("Analytical gradient with respect to weights:", gradient_weighted_sum_torch[1])

    # Compare gradients
    weights_grad_match = torch.allclose(gradient_weighted_sum_torch[0], weights_data.grad)
    input_grad_match = torch.allclose(gradient_weighted_sum_torch[1], input_data.grad)

    print("Gradients match (input):", input_grad_match)
    print("Gradients match (weights):", weights_grad_match)



def test_get_weights_gradcheck(fn):
    print("WEIGHTS FROM SIGMAS CHECK")
    sigmas = torch.rand(32, 64, requires_grad=True)
    deltas = torch.rand(32, 64, requires_grad=True)

    # Check gradients using gradcheck with increased eps and disabled exception raising


    weights_custom = fn(sigmas, deltas)
    weights_custom.backward(torch.ones_like(weights_custom))
    # Compute the weighted sum using PyTorch built-in functions
    alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T+t]
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
    # [N, T+t+1]
    weights_torch = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
    # Compute gradients with respect to the weighted sum
    gradient_weights_torch = torch.autograd.grad(weights_torch, (sigmas, deltas),
                                                      grad_outputs=torch.ones_like(weights_torch))

    #print("Analytical gradient with respect to sigmas:", gradient_weights_torch[0])

    # Compare gradients
    sigmas_grad_match = torch.allclose(gradient_weights_torch[0], sigmas.grad, atol=1e-05)

    print("Gradients match (sigmas):", sigmas_grad_match)

def test_get_image_gradcheck(fn):
    print("IMAGES RENDERING CHECK")
    sigmas = torch.rand(32, 16, requires_grad=True)
    rgbs = torch.rand(32, 16, 3, requires_grad=True)
    deltas = torch.rand(32, 16, requires_grad=True)

    # Check gradients using gradcheck with increased eps and disabled exception raising
    images_custom = fn(sigmas, rgbs, deltas)
    images_custom.backward(torch.ones_like(images_custom))
    # Compute the weighted sum using PyTorch built-in functions
    alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T+t]
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
    # [N, T+t+1]
    weights_torch = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
    image_torch = torch.sum(weights_torch.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]
    # Compute gradients with respect to the weighted sum
    gradient_image_torch = torch.autograd.grad(image_torch, (sigmas, rgbs, deltas),
                                                      grad_outputs=torch.ones_like(image_torch))

    #print("Analytical gradient with respect to sigmas:", gradient_weights_torch[0])

    # Compare gradients
    sigmas_grad_match = torch.allclose(gradient_image_torch[0], sigmas.grad, atol=1e-05)
    rgbs_grad_match = torch.allclose(gradient_image_torch[1], rgbs.grad, atol=1e-05)

    print("Gradients match (sigmas):", sigmas_grad_match)
    print("Gradients match (rgbs):", rgbs_grad_match)
def grad_image_to_sigma(deltas, sigmas, rgbs):
    alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T]
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
    Ts = torch.cumprod(alphas_shifted, dim=-1)
    T = torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
    Tf = torch.cumprod(alphas_shifted, dim=-1)[..., 1:]
    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
    wrgbs = weights[..., None] * rgbs
    color = wrgbs.sum(dim=-1)
    rgbs_partial = torch.cumsum(weights[..., None] * rgbs, dim=-2)
    rgbs_partial_debug = rgbs_partial.detach().cpu().numpy()
    rgbs_final = rgbs_partial[:, -1, :][:, None, :]
    grad = deltas[..., None] * (Tf[..., None] * rgbs - (rgbs_partial- rgbs_final))
    print()

    return

if __name__=='__main__':
    test_weighted_sum_gradcheck(raymarching.weighted_sum)
    test_get_weights_gradcheck(raymarching.get_weights)
    test_get_image_gradcheck(raymarching.get_image)
    sigmas = torch.rand(32, 64)
    deltas = torch.rand(32, 64)
    rgbs = torch.rand(32, 64, 3)
    grad_image_to_sigma(deltas, sigmas, rgbs)

