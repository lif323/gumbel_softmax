# code from https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())

    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y_soft = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y_soft
    else:
        _, ind = y_soft.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(1, ind, 1)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = y_hard - y_soft.detach() + y_soft
    return y_hard

x_prob = torch.tensor([[0.3, 0.6, 0.1],[0.7, 0.2, 0.1]]).float()
x = torch.log(x_prob)
cum_y = 0
cum_y_offical = 0
print("original prob.", x_prob)
total = 100000
temperature = 1
is_hard = True
for i in range(total):
    y = gumbel_softmax(x, temperature=temperature, hard=is_hard)
    y_o = F.gumbel_softmax(x, tau=temperature, hard=is_hard)
    cum_y = cum_y + y
    cum_y_offical = cum_y_offical + y_o
print("self   :\n", cum_y / total)
print("offical:\n", cum_y_offical / total)