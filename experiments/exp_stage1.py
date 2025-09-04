import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

from ..encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from ..vector_quantization import VectorQuantize
from ..utils import compute_downsample_rate, timefreq_to_time, time_to_timefreq, zero_pad_low_freq, zero_pad_high_freq, quantize, linear_warmup_cosine_annealingLR


class ExpStage1(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 input_length: int,
                 config: dict):
        """
        :param input_length: length of input time series
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """
        super().__init__()
        self.input_length = input_length
        self.config = config

        self.n_fft = config['VQ-VAE']['n_fft']
        init_dim = config['encoder']['init_dim']
        hid_dim = config['encoder']['hid_dim']
        downsampled_width_l = config['encoder']['downsampled_width']['lf']
        downsampled_width_h = config['encoder']['downsampled_width']['hf']
        downsample_rate_l = compute_downsample_rate(
            input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(
            input_length, self.n_fft, downsampled_width_h)

        # encoder
        self.encoder_l = VQVAEEncoder(init_dim, hid_dim, 2*in_channels, downsample_rate_l,
                                      config['encoder']['n_resnet_blocks'], 'lf', self.n_fft, frequency_indepence=True)
        self.encoder_h = VQVAEEncoder(init_dim, hid_dim, 2*in_channels, downsample_rate_h,
                                      config['encoder']['n_resnet_blocks'], 'hf', self.n_fft, frequency_indepence=False)

        # quantizer
        self.vq_model_l = VectorQuantize(
            hid_dim, config['VQ-VAE']['codebook_sizes']['lf'], **config['VQ-VAE'])
        self.vq_model_h = VectorQuantize(
            hid_dim, config['VQ-VAE']['codebook_sizes']['hf'], **config['VQ-VAE'])

        # decoder
        self.decoder_l = VQVAEDecoder(init_dim, hid_dim, 2*in_channels, downsample_rate_l,
                                      config['decoder']['n_resnet_blocks'], input_length, 'lf', self.n_fft, in_channels, frequency_indepence=True)
        self.decoder_h = VQVAEDecoder(init_dim, hid_dim, 2*in_channels, downsample_rate_h,
                                      config['decoder']['n_resnet_blocks'], input_length, 'hf', self.n_fft, in_channels, frequency_indepence=False)

    def forward(self, batch, batch_idx, return_x_rec: bool = False):
        """
        :param x: input time series (b c l)
        """
        x, y = batch

        recons_loss = {'LF.time': 0., 'HF.time': 0.}
        vq_losses = {'LF': None, 'HF': None}
        perplexities = {'LF': 0., 'HF': 0.}

        # STFT
        in_channels = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)
        u_l = zero_pad_high_freq(xf)  # (b c h w)
        x_l = F.interpolate(timefreq_to_time(
            u_l, self.n_fft, in_channels), self.input_length, mode='linear')  # (b c l)
        u_h = zero_pad_low_freq(xf)  # (b c h w)
        x_h = F.interpolate(timefreq_to_time(
            u_h, self.n_fft, in_channels), self.input_length, mode='linear')  # (b c l)

        # LF
        z_l = self.encoder_l(x)
        z_q_l, s_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        xhat_l = self.decoder_l(z_q_l)  # (b c l)

        # HF
        z_h = self.encoder_h(x)
        z_q_h, s_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        xhat_h = self.decoder_h(z_q_h)  # (b c l)

        if return_x_rec:
            x_rec = xhat_l + xhat_h  # (b c l)
            return x_rec  # (b c l)

        recons_loss['LF.time'] = F.mse_loss(x_l, xhat_l)
        perplexities['LF'] = perplexity_l
        vq_losses['LF'] = vq_loss_l

        recons_loss['HF.time'] = F.l1_loss(x_h, xhat_h)
        perplexities['HF'] = perplexity_h
        vq_losses['HF'] = vq_loss_h

        # plot `x` and `xhat`
        if not self.training and batch_idx == 0:
            b = np.random.randint(0, x_h.shape[0])
            c = np.random.randint(0, x_h.shape[1])

            alpha = 0.7
            n_rows = 3
            fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2*n_rows))

            # Extract data for plotting and metrics
            x_l_data = x_l[b, c].cpu()
            xhat_l_data = xhat_l[b, c].detach().cpu()
            x_h_data = x_h[b, c].cpu()
            xhat_h_data = xhat_h[b, c].detach().cpu()
            x_combined = x_l_data + x_h_data
            xhat_combined = xhat_l_data + xhat_h_data

            # Calculate reconstruction metrics for this sample
            mse_l = F.mse_loss(xhat_l_data, x_l_data).item()
            mse_h = F.mse_loss(xhat_h_data, x_h_data).item()
            mse_combined = F.mse_loss(xhat_combined, x_combined).item()

            plt.suptitle(
                f'Step {self.global_step} | Channel {c} | MSE: LF={mse_l:.4f}, HF={mse_h:.4f}, Total={mse_combined:.4f}')

            # Plot LF components with adaptive y-limits
            axes[0].plot(x_l_data, alpha=alpha, label='Ground Truth')
            axes[0].plot(xhat_l_data, alpha=alpha, label='Reconstructed')
            axes[0].set_title(r'$x_l$ (LF)')
            # Adaptive y-limits based on actual data range
            y_min_l = min(x_l_data.min().item(), xhat_l_data.min().item())
            y_max_l = max(x_l_data.max().item(), xhat_l_data.max().item())
            y_margin_l = (y_max_l - y_min_l) * 0.1  # 10% margin
            axes[0].set_ylim(y_min_l - y_margin_l, y_max_l + y_margin_l)
            axes[0].legend(fontsize=8)

            # Plot HF components with adaptive y-limits
            axes[1].plot(x_h_data, alpha=alpha, label='Ground Truth')
            axes[1].plot(xhat_h_data, alpha=alpha, label='Reconstructed')
            axes[1].set_title(r'$x_h$ (HF)')
            # Adaptive y-limits based on actual data range
            y_min_h = min(x_h_data.min().item(), xhat_h_data.min().item())
            y_max_h = max(x_h_data.max().item(), xhat_h_data.max().item())
            y_margin_h = (y_max_h - y_min_h) * 0.1  # 10% margin
            axes[1].set_ylim(y_min_h - y_margin_h, y_max_h + y_margin_h)
            axes[1].legend(fontsize=8)

            # Plot combined signal with adaptive y-limits
            axes[2].plot(x_combined, alpha=alpha, label='Ground Truth')
            axes[2].plot(xhat_combined, alpha=alpha, label='Reconstructed')
            axes[2].set_title(r'$x$ (LF+HF)')
            # Adaptive y-limits based on actual data range
            y_min_combined = min(x_combined.min().item(),
                                 xhat_combined.min().item())
            y_max_combined = max(x_combined.max().item(),
                                 xhat_combined.max().item())
            y_margin_combined = (
                y_max_combined - y_min_combined) * 0.1  # 10% margin
            axes[2].set_ylim(y_min_combined - y_margin_combined,
                             y_max_combined + y_margin_combined)
            axes[2].legend(fontsize=8)

            plt.tight_layout()
            wandb.log({"x vs x_rec (val)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexities

    def training_step(self, batch, batch_idx):
        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time']) + \
            vq_losses['LF']['loss'] + vq_losses['HF']['loss']

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],
                     }

        # log
        self.log('global_step', self.global_step)
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time']) + \
            vq_losses['LF']['loss'] + vq_losses['HF']['loss']

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],
                     }

        # log
        self.log('global_step', self.global_step)
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.config['exp_params']['lr'])
        scheduler = linear_warmup_cosine_annealingLR(
            opt, self.config['trainer_params']['max_steps']['stage1'], self.config['exp_params']['linear_warmup_rate'], min_lr=self.config['exp_params']['min_lr'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}
