import argparse
import os
import logging
import random
from types import SimpleNamespace

import yaml
import numpy as np
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Subset

from dataset import LMDBDataset, get_transforms, FOLLOW_BATCH
from diffusion import ScorePosNet3D


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inf_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def resolve_lmdb_path(data_cfg):
    candidate = (
        data_cfg.get('lmdb_path')
        or data_cfg.get('lmdb')
        or data_cfg.get('path')
    )
    if candidate is None:
        return './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
    if os.path.exists(candidate):
        return candidate
    lmdb_candidate = f'{candidate}.lmdb'
    if os.path.exists(lmdb_candidate):
        return lmdb_candidate
    return candidate


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--lmdb_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=None)
    args = parser.parse_args()

    # ── Config ──
    with open(args.config, 'r') as f:
        yml = yaml.safe_load(f)
    model_dict = yml['model']
    train_dict = yml['train']

    data_dict = yml.get('data', {})
    lmdb_path = resolve_lmdb_path(data_dict)
    split_path = data_dict.get('split', None)

    model_cfg = SimpleNamespace(**model_dict)
    train_cfg = SimpleNamespace(**train_dict)

    # optimizer/scheduler는 config.yml에서 중첩 dict로 저장됨 → 별도로 꺼냄
    opt_cfg = train_dict.get('optimizer', {})
    sch_cfg = train_dict.get('scheduler', {})

    # CLI override
    if args.lmdb_path is not None:
        lmdb_path = args.lmdb_path
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        opt_cfg['lr'] = args.lr
    if args.max_iters is not None:
        train_cfg.max_iters = args.max_iters

    # train_report_iter: config에 없으면 val_freq와 동일하게 설정
    train_report_iter = train_dict.get('train_report_iter', train_cfg.val_freq)

    seed_all(train_cfg.seed)

    # ── Logging ──
    tag = f'_{args.tag}' if args.tag else ''
    log_dir = os.path.join(args.logdir, f'targetdiff{tag}')
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(f'Model config: {model_dict}')
    logger.info(f'Train config: {train_dict}')
    logger.info(f'LMDB path: {lmdb_path}')

    # ── Dataset ──
    logger.info('Loading dataset...')
    transform, protein_featurizer, ligand_featurizer = get_transforms()
    full_dataset = LMDBDataset(lmdb_path, transform=transform)
    n_total = len(full_dataset)

    if split_path is not None:
        split = torch.load(split_path)
        train_idx = split['train']
        # Prefer explicit val split when it is non-empty; otherwise fall back to test.
        val_idx = split.get('val')
        if val_idx is None or len(val_idx) == 0:
            val_idx = split.get('test')
        if val_idx is None or len(val_idx) == 0:
            raise ValueError("split file must contain non-empty 'val' or 'test' key.")
        train_set = Subset(full_dataset, train_idx)
        val_set   = Subset(full_dataset, val_idx)
        n_train = len(train_idx)
        n_val   = len(val_idx)
    else:
        # split 파일 없으면 9:1 random split
        n_val   = max(1, int(n_total * 0.1))
        n_train = n_total - n_val
        train_set, val_set = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(train_cfg.seed)
        )

    logger.info(f'Dataset: {n_total} total | {n_train} train | {n_val} val')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = inf_iterator(DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    ))
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )

    # ── Model ──
    logger.info('Building model...')
    model = ScorePosNet3D(
        model_cfg,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    logger.info(f'Trainable parameters: {count_parameters(model) / 1e6:.4f} M')
    logger.info(f'Protein feature dim: {protein_featurizer.feature_dim} | '
                f'Ligand feature dim: {ligand_featurizer.feature_dim}')

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.get('lr', 5e-4),
        weight_decay=opt_cfg.get('weight_decay', 0),
        betas=(opt_cfg.get('beta1', 0.95), opt_cfg.get('beta2', 0.999)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=sch_cfg.get('factor', 0.6),
        patience=sch_cfg.get('patience', 10),
        min_lr=sch_cfg.get('min_lr', 1e-6),
    )

    # ── Train ──
    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(train_cfg.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * train_cfg.pos_noise_std   # 노이즈 생성하고, 0.1 곱해서 노이즈 크기 조절
            gt_protein_pos = batch.protein_pos + protein_noise   # 노이즈 추가

            results = model.get_diffusion_loss(   # diffusion.py로 진입 -> get_diffusion_loss 함수 호출
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
            )
            loss = results['loss'] / train_cfg.n_acc_batch
            loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)   # clip_grad_norm_ : gradient 폭주 막는 torch 내장함수
        optimizer.step()

        if it % train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | '
                'Lr: %.6f | Grad Norm: %.6f' % (
                    it, results['loss'], results['loss_pos'], results['loss_v'],
                    optimizer.param_groups[0]['lr'], orig_grad_norm,
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

    # ── Validate ──
    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,
                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,
                    )
                    sum_loss     += float(results['loss'])     * batch_size
                    sum_loss_pos += float(results['loss_pos']) * batch_size
                    sum_loss_v   += float(results['loss_v'])   * batch_size
                    sum_n        += batch_size

        avg_loss     = sum_loss     / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v   = sum_loss_v   / sum_n

        scheduler.step(avg_loss)

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000,
            )
        )
        writer.add_scalar('val/loss',     avg_loss,     it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v',   avg_loss_v,   it)
        writer.flush()
        return avg_loss

    # ── Main loop ──
    try:
        best_loss, best_iter = None, None

        for it in range(1, train_cfg.max_iters + 1):
            train(it)
            if it % train_cfg.val_freq == 0 or it == train_cfg.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                    torch.save({
                        'config':    yml,
                        'model':     model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(
                        f'[Validate] Val loss not improved. '
                        f'Best: {best_loss:.6f} at iter {best_iter}'
                    )
    except KeyboardInterrupt:
        logger.info('Terminating...')
