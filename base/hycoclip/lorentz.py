from __future__ import annotations

import math

import torch
from torch import Tensor
from loguru import logger


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def return_time(x: Tensor, curv: float | Tensor = 1.0):

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    return x_time



def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def oxy_angle_eval(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    logger.info(f"x_time shape: {x_time.size()}")
    logger.info(f"y_time shape: {y_time.size()}")

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).

    # c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    c_xyl = curv * (y @ x.T - y_time @ x_time.T)
    logger.info(f"c_xyl shape: {c_xyl.size()}")

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time.T
    logger.info(f"acos_numer shape: {acos_numer.size()}")
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    logger.info(f"acos_denom shape: {acos_denom.size()}")

    acos_input = acos_numer / (torch.norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = - torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output




def mobius_add(x, y, curv):
    """
    Möbius addition on the Poincaré ball with curvature c.
    
    Args:
        x: Tensor of shape (..., dim)
        y: Tensor of shape (..., dim)
        c: Positive float (curvature)

    Returns:
        Tensor of shape (..., dim) representing x ⊕_c y
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)  # ||x||^2
    y2 = torch.sum(y * y, dim=-1, keepdim=True)  # ||y||^2
    xy = torch.sum(x * y, dim=-1, keepdim=True)  # <x, y>

    num = (1 + 2 * curv * xy + curv * y2) * x + (1 - curv * x2) * y
    denom = 1 + 2 * curv * xy + curv**2 * x2 * y2

    return num / denom

def mobius_scalar_mul(x,r, c):
    """
    Möbius scalar multiplication of x by scalar r in the Poincaré ball model.

    Args:
        r: scalar (float or tensor)
        x: torch tensor of shape (..., dim)
        c: positive curvature constant (float)

    Returns:
        Tensor of same shape as x after Möbius scalar multiplication.
    """
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x.dtype, device=x.device))
    
    scaled = (1.0 / sqrt_c) * torch.tanh(r * torch.atanh(sqrt_c * norm_x)) * (x / norm_x)
    scaled = torch.where(norm_x > 0, scaled, torch.zeros_like(x))  # Handle x == 0 case safely

    return scaled


def interpolation(x,y,r,c):
    add_x_y = mobius_add(-x, y, c)
    mul_c = mobius_scalar_mul(add_x_y, r,c)
    interpolated = mobius_add(x, mul_c, c)
    return interpolated



def poincare_to_lorentz(x_b, c):
    """
    Maps point from Poincaré ball to Lorentz hyperboloid.
    
    Args:
        x_b: Tensor of shape (..., dim), point in Poincaré ball
        c: Positive float (curvature)

    Returns:
        Tensor of shape (..., dim), point in Lorentz model
    """
    norm_sq = torch.sum(x_b**2, dim=-1, keepdim=True)
    scale = 2.0 / (1 - c * norm_sq)
    x_h = scale * x_b
    return x_h


def lorentz_to_poincare(x_h, c):
    """
    Maps point from Lorentz hyperboloid to Poincaré ball.

    Args:
        x_h: Tensor of shape (..., dim), point in Lorentz model
        c: Positive float (curvature)

    Returns:
        Tensor of shape (..., dim), point in Poincaré ball
    """
    norm_sq = torch.sum(x_h**2, dim=-1, keepdim=True)
    denom = 1 + torch.sqrt(1 + c * norm_sq)
    x_b = x_h / denom
    return x_b



### interpolation 

def lorentz_inner(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Lorentzian inner product: -p0*q0 + <p_spatial, q_spatial>"""
    return -p[..., 0:1] * q[..., 0:1] + torch.sum(p[..., 1:] * q[..., 1:], dim=-1, keepdim=True)

def lorentz_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Lorentzian norm: sqrt(<v, v>_L), returns shape (B, 1)"""
    norm2 = -v[..., 0:1] ** 2 + torch.sum(v[..., 1:] ** 2, dim=-1, keepdim=True)
    return torch.sqrt(torch.clamp(norm2, min=eps))


def log_map_pq(p: torch.Tensor, q: torch.Tensor, curv: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Logarithmic map on Lorentz model at point p, mapping q -> tangent vector at p.

    Args:
        p, q: tensors of shape (B, D+1), points on Lorentz manifold
        curv: curvature constant (positive scalar)
    
    Returns:
        v: tensor of shape (B, D+1), vector in tangent space at p
    """
    inner_l = lorentz_inner(p, q)               # <p, q>_L
    k_inner = curv * inner_l                    # κ<p, q>

    numer = torch.acosh(torch.clamp(-k_inner, min=1.0 + eps))  # arccosh(-κ<p,q>)
    denom = torch.sqrt(torch.clamp(k_inner ** 2 - 1, min=eps)) # sqrt((κ<p,q>)² - 1)

    proj = q + k_inner * p                      # q + κ<p,q>p
    return (numer / denom) * proj


def exp_map_pv(p: torch.Tensor, v: torch.Tensor, curv: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map on Lorentz model at point p, mapping v (in T_pL^n) -> manifold point.

    Args:
        p: base point on Lorentz manifold, shape (B, D+1)
        v: tangent vector at p, shape (B, D+1)
        curv: curvature constant (positive scalar)

    Returns:
        Tensor of shape (B, D+1): point on Lorentz manifold
    """
    sqrt_c = curv ** 0.5
    v_norm = lorentz_norm(v, eps=eps)  # ||v||_L

    coef1 = torch.cosh(sqrt_c * v_norm)
    coef2 = torch.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm + eps)  # add eps for stability

    return coef1 * p + coef2 * v

def geodesic_interploation(p_spatial,q_spatial,t,curv):
    p_time = torch.sqrt(1 / curv + torch.sum(p_spatial**2, dim=-1, keepdim=True))
    q_time = torch.sqrt(1 / curv + torch.sum(q_spatial**2, dim=-1, keepdim=True))
    p = torch.cat([p_time, p_spatial], dim=-1)  # shape: (B, D+1)
    q = torch.cat([q_time, q_spatial], dim=-1)

    v = log_map_pq(p,q,curv)
    
    interpolated_z = exp_map_pv(p,t*v,curv)
    interpolated_z = interpolated_z[...,1:]
    return interpolated_z

# def geodesic_interpolation(p: Tensor, q: Tensor, t: float, curv: float = 1.0, eps: float = 1e-8) -> Tensor:
#     """
#     Perform geodesic interpolation between two points on the hyperboloid.

#     Args:
#         p, q: Tensors of shape (B, D) on the hyperboloid.
#         t: Interpolation coefficient in [0, 1]
#         curv: Positive scalar curvature.
#         eps: Small number for numerical stability.

#     Returns:
#         Interpolated points of shape (B, D)
#     """
#     # Efficient diagonal distance only
#     inner = -curv * lorentz_inner(p, q)  # shape: (B,)
#     alpha = torch.acosh(torch.clamp(inner, min=1 + eps))  # shape: (B,)
#     alpha = torch.clamp(alpha, min=eps)  # avoid zero division
#     c = torch.sinh((1 - t) * alpha) / torch.sinh(alpha)
#     d = torch.sinh(t * alpha) / torch.sinh(alpha)
#     return c.unsqueeze(-1) * p + d.unsqueeze(-1) * q