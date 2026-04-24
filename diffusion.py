import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from network import UniTransformer, compose_context, ShiftedSoftplus


# ──────────────────────────────────────────────
# Schedule Utilities
# ──────────────────────────────────────────────

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'sigmoid':
        x = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = (1 / (1 + np.exp(-x))) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas


def to_torch_const(x):
    return nn.Parameter(torch.from_numpy(x).float(), requires_grad=False)


def extract(coef, t, batch):
    return coef[t][batch].unsqueeze(-1)


# ──────────────────────────────────────────────
# Position Utilities
# ──────────────────────────────────────────────

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# ──────────────────────────────────────────────
# Categorical Diffusion Utilities (log-space)
# ──────────────────────────────────────────────

def index_to_log_onehot(x, num_classes): # 인덱스를 log-onehot 형태로 변환
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    return torch.log(F.one_hot(x, num_classes).float().clamp(min=1e-30)) # one-hot encoding 후 log 취함
# log를 씌우는 이유?: discrete diffusion 모델에서 확률을 log-space로 다루기때문

def log_sample_categorical(logits): # 로그 공간에서 카테고리 샘플링
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return (gumbel_noise + logits).argmax(dim=-1) # 최대값 인덱스 반환


def log_1_min_a(a): # 1 - exp(a) 계산
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b): # log-space에서 두 값을 더하는 함수
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def categorical_kl(log_prob1, log_prob2):
    return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


# ──────────────────────────────────────────────
# Time Embedding
# ──────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ──────────────────────────────────────────────
# ScorePosNet3D
# ──────────────────────────────────────────────

class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        self.model_mean_type = config.model_mean_type
        self.loss_v_weight = config.loss_v_weight
        self.sample_time_method = config.sample_time_method
        self.center_pos_mode = config.center_pos_mode

        # ── Position diffusion schedule ──
        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)

        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(
            np.log(np.append(posterior_variance[1], posterior_variance[1:])))

        # ── Atom type diffusion schedule (log space) ──
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
        else:
            raise NotImplementedError

        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # ── Model architecture ──
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        emb_dim = self.hidden_dim - 1 if config.node_indicator else self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = getattr(config, 'time_emb_mode', 'simple')

        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net = UniTransformer(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup,
        )

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

    # ──────────────────────────────────────────────
    # Node Embedding
    # ──────────────────────────────────────────────

    def get_node_embedding(self, protein_v, init_ligand_v, time_step, batch_ligand):
        """Protein/ligand 노드 hidden embedding 계산."""
        init_ligand_v_onehot = F.one_hot(init_ligand_v, self.num_classes).float()

        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                t_norm = (time_step.float() / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                input_ligand_feat = torch.cat([init_ligand_v_onehot, t_norm], dim=-1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v_onehot, time_feat[batch_ligand]], dim=-1)
        else:
            input_ligand_feat = init_ligand_v_onehot

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([
                h_protein, torch.zeros(len(h_protein), 1, device=h_protein.device)
            ], dim=-1)
            init_ligand_h = torch.cat([
                init_ligand_h, torch.ones(len(init_ligand_h), 1, device=init_ligand_h.device)
            ], dim=-1)

        return h_protein, init_ligand_h

    # ──────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────

    def forward(self, protein_pos, protein_v, batch_protein,
                init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        h_protein, init_ligand_h = self.get_node_embedding(
            protein_v, init_ligand_v, time_step, batch_ligand)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein, h_ligand=init_ligand_h,
            pos_protein=protein_pos, pos_ligand=init_ligand_pos,
            batch_protein=batch_protein, batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos = final_pos[mask_ligand]
        final_ligand_h = final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
        }
        if return_all:
            preds['layer_pred_ligand_pos'] = [p[mask_ligand] for p in outputs['all_x']]
            preds['layer_pred_ligand_v'] = [
                self.v_inference(h[mask_ligand]) for h in outputs['all_h']
            ]
        return preds

    # ──────────────────────────────────────────────
    # Categorical diffusion
    # ──────────────────────────────────────────────

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):  # 한 단계의 변화 (q(v_t|v_t-1)); eq2
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)
        return log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

    def q_v_pred(self, log_v0, t, batch):  # v_0에서 t단계 후의 상태 (q(v_t|v_0)); eq3
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)
        return log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

    def q_v_sample(self, log_v0, t, batch):  # q(v_t|v_0)에서 실제 샘플 추출(샘플링)
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_posterior(self, log_v0, log_vt, t, batch): # 카테고리 사후분포 (q(v_t-1|v_t, v_0)); eq4
        t_minus_1 = torch.where(t - 1 < 0, torch.zeros_like(t), t - 1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        return unnormed - torch.logsumexp(unnormed, dim=-1, keepdim=True)

    # ──────────────────────────────────────────────
    # Position diffusion
    # ──────────────────────────────────────────────

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        return (extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt
                - extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps)

    def q_pos_posterior(self, x0, xt, t, batch):
        return (extract(self.posterior_mean_c0_coef, t, batch) * x0
                + extract(self.posterior_mean_ct_coef, t, batch) * xt)

    # ──────────────────────────────────────────────
    # Time sampling
    # ──────────────────────────────────────────────

    def sample_time(self, num_graphs, device, method):  # loss가 큰 timestep을 더 자주 샘플링하도록
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001  # Lt_history: 각 timestep 별로 과거 loss 기록
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError(f'Unknown time sampling method: {method}')

    # ──────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)
        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        mask = (t == 0).float()[batch]
        return scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)

    def get_diffusion_loss(self, protein_pos, protein_v, batch_protein,
                           ligand_pos, ligand_v, batch_ligand, time_step=None):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample timestep
        if time_step is None:
            time_step, pt = self.sample_time(
                num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        # 2. perturb position   # 노이즈 추가 (forward process)
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.randn_like(ligand_pos)
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        # 3. perturb atom type   # 카테고리 샘플링 (forward process)
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand)

        # 4. forward pass
        preds = self(
            protein_pos=protein_pos, protein_v=protein_v, batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed, init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand, time_step=time_step,
        )
        pred_ligand_pos = preds['pred_ligand_pos']
        pred_ligand_v = preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # 5. position loss
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError(f'Unknown model_mean_type: {self.model_mean_type}')

        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError(f'Unknown model_mean_type: {self.model_mean_type}')
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0).mean()

        # 6. atom type loss (KL)
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(
            log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(
            log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(
            log_v_model_prob, log_ligand_v0, log_v_true_prob, time_step, batch_ligand)
        loss_v = kl_v.mean()

        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss': loss,
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
        }

    # ──────────────────────────────────────────────
    # Sampling (reverse process)
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(reversed(range(
            self.num_timesteps - num_steps, self.num_timesteps)))

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(
                size=(num_graphs,), fill_value=i,
                dtype=torch.long, device=protein_pos.device)

            preds = self(
                protein_pos=protein_pos, protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos, init_ligand_v=ligand_v,
                batch_ligand=batch_ligand, time_step=t,
            )

            # position: predict x0 then posterior sample
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(
                    xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos = (pos_model_mean
                          + nonzero_mask * (0.5 * pos_log_variance).exp()
                          * torch.randn_like(ligand_pos))

            # atom type: predict v0 then posterior sample
            if not pos_only:
                log_v_recon = F.log_softmax(preds['pred_ligand_v'], dim=-1)
                log_v_cur = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(
                    log_v_recon, log_v_cur, t, batch_ligand)
                ligand_v = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }
