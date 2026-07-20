"""PyTorch improved MeanFlow training loss for video latents.

Faithful port of iMeanFlow.forward() in imf.py (JAX), adapted from images
(B, H, W, C) to video latents (B, C, T, H, W): per-sample scalars broadcast
as (B, 1, 1, 1, 1) and losses sum over the (C, T, H, W) dims.

Model calls that do not need jvp (guidance) run under torch.no_grad();
the mean-field u prediction and its time derivative come from a single
torch.func.jvp (forward-mode AD) call through the network.
"""

import torch
import torch.nn as nn


class IMFVideoLoss(nn.Module):
    """improved MeanFlow loss for video latents."""

    def __init__(
        self,
        net,
        num_classes,
        # Noise distribution
        P_mean=-0.4,
        P_std=1.0,
        # Loss
        data_proportion=0.5,
        cfg_beta=1.0,
        class_dropout_prob=0.1,
        s_max=7.0,
        # Training dynamics
        norm_p=1.0,
        norm_eps=0.01,
    ):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.s_max = s_max
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz, device, dtype):
        rnd_normal = torch.randn(bz, 1, 1, 1, 1, device=device, dtype=dtype)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz, device, dtype):
        """Sample t and r from logit-normal distribution."""
        t = self.logit_normal_dist(bz, device, dtype)
        r = self.logit_normal_dist(bz, device, dtype)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = torch.arange(bz, device=device) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1, 1)
        r = torch.where(fm_mask, t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, device):
        """Sample CFG scale omega from power distribution."""
        u = torch.rand(bz, 1, 1, 1, 1, device=device, dtype=torch.float32)

        if self.cfg_beta == 1.0:
            s = torch.exp(
                u * torch.log1p(torch.tensor(self.s_max, dtype=torch.float32))
            )
        else:
            smax = torch.tensor(self.s_max, dtype=torch.float32)
            b = torch.tensor(self.cfg_beta, dtype=torch.float32)

            log_base = (1.0 - b) * torch.log1p(smax)
            log_inner = torch.log1p(u * torch.expm1(log_base))

            s = torch.exp(log_inner / (1.0 - b))

        return s.float()

    def sample_cfg_interval(self, bz, device, dtype, fm_mask=None):
        """Sample CFG interval [t_min, t_max] from uniform distribution."""
        t_min = 0.5 * torch.rand(bz, 1, 1, 1, 1, device=device, dtype=dtype)
        t_max = 0.5 + 0.5 * torch.rand(bz, 1, 1, 1, 1, device=device, dtype=dtype)

        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        By default, we use auxiliary v-head to predict v component as well.

        Args:
            x: Noisy video latent at time t, (B, C, T, H, W).
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v)
            u: Predicted u (average velocity field).
            v: Predicted v (instantaneous velocity field).
        """
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        """Compute the predicted v component conditioned on class labels."""

        # Set h, t_min, t_max to dummy values for v prediction
        h = torch.zeros_like(t)
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)

        v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]

        return v

    def v_fn(self, x, t, omega, y):
        """
        Compute both conditioned and unconditioned predicted v components.

        Returns:
            v_c: Predicted v component conditioned on class labels.
            v_u: Predicted v component without class labels.
        """
        bz = x.shape[0]

        # Create duplicated batch for conditioned and unconditioned predictions
        x = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, device=y.device, dtype=y.dtype)
        y = torch.cat([y, y_null], dim=0)
        t = torch.cat([t, t], dim=0)
        w = torch.cat([omega, torch.ones_like(omega)], dim=0)

        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = torch.chunk(out, 2, dim=0)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        For samples with dropped labels, v_g = v_t.
        """
        bz = v_t.shape[0]

        rand_mask = (
            torch.rand(bz, device=v_t.device) < self.class_dropout_prob
        )
        num_drop = rand_mask.sum().to(torch.int32)
        drop_mask = (
            torch.arange(bz, device=v_t.device)[:, None, None, None, None] < num_drop
        )

        labels = torch.where(
            drop_mask.reshape(bz),
            torch.full_like(labels, self.num_classes),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy video latent at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            w: CFG scale.
            t_min, t_max: Guidance interval.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, for jvp computation.
        """

        # compute CFG target
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        w = torch.where((t >= t_min) & (t <= t_max), w, torch.ones_like(w))

        v_c = self.v_cond_fn(z_t, t, w, y=y)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)

        # For flow matching samples, there is no CFG interval
        v_g = torch.where(fm_mask, v_g_fm, v_g)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, latents, labels):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            latents: A batch of video latents, shape (B, C, T, H, W).
            labels: Corresponding class labels, shape (B,).

        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = latents
        bz = x.shape[0]
        device, dtype = x.device, x.dtype

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz, device, dtype)

        e = torch.randn_like(x)
        z_t = (1 - t) * x + t * e
        v_t = e - x

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, device, dtype, fm_mask)
        omega = self.sample_cfg_scale(bz, device)

        # Compute guided velocity v_g and conditioned velocity v_c
        # (no jvp / gradient needed through the guidance targets)
        with torch.no_grad():
            v_g, v_c = self.guidance_fn(
                v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
            )

        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)

        # Warped u-function for jvp computation.
        # torch.func.jvp has_aux semantics: u_fn returns (output, aux),
        # jvp returns (output, output_tangent, aux). The v-head prediction
        # is the aux — its tangent is never formed.
        def u_fn(z_t, t, r):
            u, v = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)
            return u, v

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)

        # Different from original MeanFlow, we use predicted v in the jvp
        u, du_dt, v = torch.func.jvp(
            u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True
        )

        # Our compound function V = u + (t - r) * du/dt
        V = u + (t - r) * du_dt.detach()

        v_g = v_g.detach()

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / adp_wt.detach()

        # improved MeanFlow objective is conceptually v-loss
        loss_u = torch.sum((V - v_g) ** 2, dim=(1, 2, 3, 4))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss
        loss_v = torch.sum((v - v_g) ** 2, dim=(1, 2, 3, 4))
        loss_v = adp_wt_fn(loss_v)

        loss = loss_u + loss_v
        loss = loss.mean()  # mean over batch

        dict_losses = {
            "loss": loss.detach(),
            "loss_u": torch.mean((V - v_g) ** 2).detach(),
            "loss_v": torch.mean((v - v_g) ** 2).detach(),
        }

        return loss, dict_losses
