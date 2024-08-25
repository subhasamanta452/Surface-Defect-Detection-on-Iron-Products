import torch
import torchvision
import scipy
import numpy as np

def denormalized(image):
    denormalized_fake = image.clone()
    std = [0.5,0.5,0.5]
    mean = [0.5,0.5,0.5]
    for i in range(3):  # Assuming image has 3 channels (RGB)
        denormalized_fake[:, i, :, :] = (image[:, i, :, :] * std[i]) + mean[i]
    return denormalized_fake


def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

def matrix_sqrt(x):
    '''
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    '''
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features) 
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2*torch.trace(matrix_sqrt(sigma_x @ sigma_y))