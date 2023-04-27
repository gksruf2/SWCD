import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import wandb

from typing import Optional, Sequence, Tuple
def _check_same_shape(pred: torch.Tensor, target: torch.Tensor):
    """ Check that predictions and target have the same shape, else raise error """
    if pred.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")
    
def reduce(to_reduce: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduces a given tensor by a given reduction method
    Args:
        to_reduce : the tensor, which shall be reduced
       reduction :  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')
    Return:
        reduced Tensor
    Raise:
        ValueError if an invalid reduction parameter was given
    """
    if reduction == "elementwise_mean":
        return torch.mean(to_reduce)
    if reduction == "none":
        return to_reduce
    if reduction == "sum":
        return torch.sum(to_reduce)
    raise ValueError("Reduction parameter unknown.")

def _gaussian(kernel_size: int, sigma: int, dtype: torch.dtype, device: torch.device):
    dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=dtype, device=device)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)


def _gaussian_kernel(channel: int, kernel_size: Sequence[int], sigma: Sequence[float],
                     dtype: torch.dtype, device: torch.device):
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])

def _ssim_update(
    preds: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got pred: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW shape."
            f" Got pred: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _ssim_compute(
    preds: torch.Tensor,
    target: torch.Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
):
    if len(kernel_size) != 2 or len(sigma) != 2:
        raise ValueError(
            "Expected `kernel_size` and `sigma` to have the length of two."
            f" Got kernel_size: {len(kernel_size)} and sigma: {len(sigma)}."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())

    c1 = pow(k1 * data_range, 2)
    c2 = pow(k2 * data_range, 2)
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    kernel = _gaussian_kernel(channel, kernel_size, sigma, dtype, device)
    pad_w = (kernel_size[0] - 1) // 2
    pad_h = (kernel_size[1] - 1) // 2

    preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)
    outputs = F.conv2d(input_list, kernel, groups=channel)
    output_list = [outputs[x * preds.size(0): (x + 1) * preds.size(0)] for x in range(len(outputs))]

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target + c2
    lower = sigma_pred_sq + sigma_target_sq + c2

    ssim_idx = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    ssim_idx = ssim_idx[..., pad_h:-pad_h, pad_w:-pad_w]

    return reduce(ssim_idx, reduction)

def ssim(
    preds: torch.Tensor,
    target: torch.Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Computes Structual Similarity Index Measure
    Args:
        pred: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel (default: (11, 11))
        sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied
        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03
    Return:
        Tensor with SSIM score
    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim(preds, target)
        tensor(0.9219)
    """
    preds, target = _ssim_update(preds, target)
    return _ssim_compute(preds, target, kernel_size, sigma, reduction, data_range, k1, k2)

def _psnr_compute(
    sum_squared_error: torch.Tensor,
    n_obs: int,
    data_range: float,
    base: float = 10.0,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    psnr_base_e = 2 * torch.log(data_range) - torch.log(sum_squared_error / n_obs)
    psnr = psnr_base_e * (10 / torch.log(torch.tensor(base)))
    return psnr

def _psnr_update(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
    sum_squared_error = torch.sum(torch.pow(preds - target, 2))
    n_obs = target.numel()
    return sum_squared_error, n_obs

def psnr(
    preds: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    base: float = 10.0,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    Computes the peak signal-to-noise ratio

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range: the range of the data. If None, it is determined from the data (max - min)
        base: a base of a logarithm to use (default: 10)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied
        return_state: returns a internal state that can be ddp reduced
            before doing the final calculation

    Return:
        Tensor with PSNR score

    Example:

        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(pred, target)
        tensor(2.5527)

    """
    if data_range is None:
        data_range = target.max() - target.min()
    else:
        data_range = torch.tensor(float(data_range))
    sum_squared_error, n_obs = _psnr_update(preds, target)
    return _psnr_compute(sum_squared_error, n_obs, data_range, base, reduction)

#from pytorch_lightning.metrics.functional import ssim, psnr
from stage2_cINN.AE.modules.LPIPS import LPIPS

#https://gaussian37.github.io/vision-concept-ssim/

def KL(mu, logvar):
    ## computes KL-divergence loss between NormalGaussian and parametrized learned distribution
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))

def fmap_loss(fmap1, fmap2, metric):
    recp_loss = 0
    for idx in range(len(fmap1)):
        if metric == 'L1':
            recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
        if metric == 'L2':
            recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
    return recp_loss / len(fmap1)


def hinge_loss(fake_data, orig_data, update):
    ## hinge loss implementation
    ## update determines if loss should be computed for generator or discrimnator
    if update == 'disc':
        L_disc1 = torch.mean(torch.nn.ReLU()(1.0 - orig_data))
        L_disc2 = torch.mean(torch.nn.ReLU()(1.0 + fake_data))
        return (L_disc1 + L_disc2) / 2
    elif update == 'gen':
        return -torch.mean(fake_data)


def gradient_penalty(pred, x):
    batch_size = x.size(0)
    grad_dout = torch.autograd.grad(
                    outputs=pred.mean(), inputs=x, allow_unused=True,
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg.mean()

def optical_flow(seq):
    import cv2
    #print(seq.shape)
    optical_flow = []
    for frame in seq:
        frame1 = frame[0].cpu().numpy().astype(np.uint8)
        #print(frame1.shape)
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        bgr_list = []
        for i, frame2 in enumerate(frame[1:]):
            frame2 = frame[i].cpu().numpy().astype(np.uint8)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            temp = np.zeros((flow.shape[0],flow.shape[1],1))
            bgr_list.append(np.concatenate((flow,temp),axis=-1))
        optical_flow.append(bgr_list)
    return torch.from_numpy(np.array(optical_flow)).float()

## Backward incl Loss
class Backward(nn.Module):
    def __init__(self, opt):
        super(Backward, self).__init__()
        self.dic = opt
        self.w_kl = opt.Training['w_kl']
        self.lpips = LPIPS().cuda()
        self.gan_loss = opt.Training['GAN_Loss']
        self.w_coup_t = opt.Training['w_coup_t']
        self.w_fmap_t = opt.Training['w_fmap_t']
        self.w_coup_s = opt.Training['w_coup_s']
        self.subsample_length = opt.Training['subsample_length']
        self.w_mse = opt.Training['w_recon']
        self.w_GP = opt.Training['w_GP']
        self.w_percep = opt.Training['w_percep']
        self.seq_length = opt.Data['sequence_length']
        self.pretrain = opt.Training['pretrain']

    def forward(self, decoder, encoder, disc_t, disc_s, disc_to, seq_o, optimizers, epoch, logger):

        opt_all, opt_d_t, opt_d_s, opt_d_to = optimizers      # [optimizer_AE, optimizer_3Dnet, optimizer_2Dnet]

        ## Perform forward pass through network
        seq_orig = seq_o[:, 1:]
        motion, mu, covar = encoder(seq_orig.transpose(1, 2))
        seq_gen = decoder(seq_o[:, 0], motion)

        ## PSNR
        PSNR = psnr(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))

        ## SSIM
        SSIM = ssim(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))

        # Subsample 16 frames if sequence length is bigger than or equal to 16
        if seq_gen.size(1) >= 16:
            length = self.subsample_length
            rand_start = np.random.randint(0, seq_gen.size(1) - length + 1)
            seq_fake = seq_gen[:, rand_start:rand_start+length]
            seq_real = seq_orig[:, rand_start:rand_start+length]
        else:
            seq_fake = seq_gen
            seq_real = seq_orig

        

        ## Sample subset of images from sequence for spatial discriminator
        rand_k = np.random.randint(0, seq_orig.size(0) * seq_orig.size(1), 20)
        data_fake = torch.cat([seq_gen.reshape(-1, *seq_gen.shape[2:])[i].unsqueeze(0) for i in rand_k])
        data_real = torch.cat([seq_orig.reshape(-1, *seq_orig.shape[2:])[i].unsqueeze(0) for i in rand_k])

        ## Update (temporal) 3D discriminator
        if self.w_GP:
            seq_real.requires_grad_() ## needs to be set to true due to GP

        data_fake_op = optical_flow(seq_fake.transpose(2, 4).detach())
        data_real_op = optical_flow(seq_real.transpose(2, 4).detach())
        data_fake_op, data_real_op = data_fake_op.cuda(), data_real_op.cuda()
        pred_gen_to, _ = disc_to(data_fake_op.transpose(2, 4).transpose(1, 2).detach())
        pred_orig_to, _ = disc_to(data_real_op.transpose(2, 4).transpose(1, 2).detach())
        L_d_to = hinge_loss(pred_gen_to, pred_orig_to, update='disc')

        if epoch >= self.pretrain:
            opt_d_to.zero_grad()
            L_d_to.backward()
            opt_d_to.step()

        pred_gen_t, _ = disc_t(seq_fake.transpose(1, 2).detach())
        pred_orig_t, _ = disc_t(seq_real.transpose(1, 2))
        L_d_t = hinge_loss(pred_gen_t, pred_orig_t, update='disc')
        
        if self.w_GP:
            L_GP = gradient_penalty(pred_orig_t, seq_real)
        else:
            L_GP = torch.zeros(1)

        if epoch >= self.pretrain:
            opt_d_t.zero_grad()
            (L_d_t + self.w_GP * L_GP).backward()
            opt_d_t.step()

        ## Update spatial discriminator (patch disc)
        pred_gen_s = disc_s(data_fake.detach())
        pred_orig_s = disc_s(data_real)
        L_d_s = hinge_loss(pred_gen_s, pred_orig_s, update='disc')
        if epoch >= self.pretrain:
            opt_d_s.zero_grad()
            L_d_s.backward()
            opt_d_s.step()

        ## Update VAE
        Loss_VAE = 0
        pred_gen_s = disc_s(data_fake)
        loss_gen_s = hinge_loss(pred_gen_s, pred_orig_s, update='gen')
        if epoch >= self.pretrain:
            Loss_VAE += loss_gen_s

        pred_gen_t, fmap_gen_t  = disc_t(seq_fake.transpose(1, 2))
        pred_orig_t, fmap_orig_t = disc_t(seq_real.transpose(1, 2))
        coup_t = hinge_loss(pred_gen_t, pred_orig_t, update='gen')

        ## Feature Map Loss
        L_fmap_t = fmap_loss(fmap_gen_t, fmap_orig_t, metric='L1')

        ## Generator loss
        L_temp = self.w_coup_t * coup_t + self.w_fmap_t * L_fmap_t
        if epoch >= self.pretrain:
            Loss_VAE += L_temp

        LPIPS = self.lpips(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:])).mean()

        ## L1 Error
        L_recon = torch.mean(torch.abs((seq_gen - seq_orig)))

        ## KL Loss
        L_kl = KL(mu, covar)

        Loss_VAE += self.w_percep * LPIPS + self.w_kl * L_kl + self.w_mse * L_recon

        opt_all.zero_grad()
        Loss_VAE.backward()
        opt_all.step()

        loss_dic = {
            ## Losses for VAE
            "Loss_VAE": Loss_VAE.item(),
            "Loss_L1": L_recon.item(),
            "LPIPS": LPIPS.item(),
            "Loss_KL": L_kl.item(),
            "Loss_GEN_S": loss_gen_s.item(),
            "Loss_GEN_T": coup_t.item(),
            ## Losses for temporal discriminator
            "Loss_Disc_T": L_d_t.item(),
            "Loss_Fmap_T": L_fmap_t.item(),
            "L_GP": L_GP.item(),
            "Logits_Real_T": pred_orig_t.mean().item(),
            "Logits_Fake_T": pred_gen_t.mean().item(),
            ## Losses for spatial discriminator
            "Loss_Disc_S": L_d_s.item(),
            "Logits_Real_S": pred_orig_s.mean().item(),
            "Logits_Fake_S": pred_gen_s.mean().item(),
            ## Losses for optical discriminator
            "Loss_Disc_TO": L_d_to.item(),
            "Logits_Real_TO": pred_orig_to.mean().item(),
            "Logits_Fake_TO": pred_gen_to.mean().item(),
            ## Additional
            "PSNR": PSNR.item(),
            "SSIM": SSIM.item(),
        }

        ## Log dic online and offline
        wandb.log(loss_dic)
        logger.append(loss_dic)

        return [seq_gen.detach().cpu(), seq_orig.cpu()]


    def eval(self, decoder, encoder, seq_o, logger):

        ## Perform forward pass through network
        seq_orig = seq_o[:, 1:]
        motion, mu, covar = encoder(seq_orig.transpose(1, 2))
        seq_gen = decoder(seq_o[:, 0], motion)

        ## PSNR
        PSNR = psnr(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))

        ## SSIM
        SSIM = ssim(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))

        LPIPS = self.lpips(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))

        ## L1 Error
        L_recon = torch.mean(torch.abs((seq_gen - seq_orig)))

        ## KL Loss
        L_kl = KL(mu, covar)

        loss_dic = {"Loss_L1": L_recon.item(),
                    "LPIPS": LPIPS.mean().item(),
                    "L_KL": L_kl.item(),
                    "PSNR": PSNR.item(),
                    "SSIM": SSIM.item()
        }

        ## Log dic online and offline
        logger.append(loss_dic)
        loss_dic = {'eval_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)

        return [seq_gen.detach().cpu(), seq_orig.cpu()]



