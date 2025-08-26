import torch


def dist2gp(x, c):
    """ Calculates squared distance between two sets of points.
    Args:
        x: A 'n_d x ndims' tensor
        c: A 'n_c x ndims' tensor
    Returns:
        A 'n_d x n_c' tensor
    """
    dim_x = x.shape[-1]
    dim_c = c.shape[-1]
    if dim_x != dim_c:
        raise Exception('Data dimension does not match dimension of centers.')
    xx = torch.sum(x ** 2, dim=-1)  # (n_d, 1)
    cc = torch.sum(c ** 2, dim=-1)  # (n_c, 1)
    xc = x @ c.transpose(-1, -2)  # (n_data, n_centres)
    return xx.unsqueeze(-1) + cc.unsqueeze(-2) - 2. * xc


if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
    print(dist2gp(x, x))
