import paddle


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (paddle.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
 

def euclidean_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = paddle.pow(x, 2).sum(1, keepdim=True).expand([m, n])
    yy = paddle.pow(y, 2).sum(1, keepdim=True).expand([n, m]).t()
    dist = xx + yy - 2 * paddle.matmul(x, y.t())
    dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_sim(x, y):
    bs1, bs2 = x.shape[0], y.shape[0]
    frac_up = paddle.matmul(x, y.transpose((1, 0)))
    frac_down = (paddle.sqrt(paddle.sum(paddle.pow(x, 2), 1))).unsqueeze(-1) * \
                (paddle.sqrt(paddle.sum(paddle.pow(y, 2), 1))).unsqueeze(0)
    cosine = frac_up / frac_down
    return cosine

def cosine_dist(x, y):
    return 1 - cosine_sim(x, y)