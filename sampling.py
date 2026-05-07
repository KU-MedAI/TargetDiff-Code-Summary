import argparse
import os
import time
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_sum, scatter_mean
from rdkit import Chem
from tqdm.auto import tqdm

from dataset import (
    LMDBDataset, get_transforms, FOLLOW_BATCH,
    MAP_INDEX_TO_ATOM_TYPE_AROMATIC, ProteinLigandData,
)
from diffusion import ScorePosNet3D, log_sample_categorical
from reconstruct import reconstruct_from_generated, MolReconsError


# ──────────────────────────────────────────────
# Atom Number Sampling (pocket size → num atoms)
# ──────────────────────────────────────────────

ATOM_NUM_BOUNDS = [
    26.522, 27.875, 28.827, 29.653, 30.440, 31.234, 32.015, 33.035, 34.576
]

ATOM_NUM_BINS = [
    ([10,6,9,22,11,13,12,28,8,7,16,18,5,19,17,20,24,27,14,26,21,15,33,29,3,4,30,31,23,34,36,25,32,39,35,37,38,40,43,2,41],
     [.1196,.0839,.1152,.0065,.1131,.0645,.0807,.0086,.0807,.0626,.0151,.019,.0092,.024,.0205,.0164,.011,.0122,.03,.0141,.0105,.0164,.0033,.0061,.0031,.0085,.0052,.0057,.0146,.002,.0019,.0084,.0024,.0007,.0017,.0012,.0005,.0003,.0003,.0002,.0001]),
    ([26,13,28,22,20,15,11,10,25,14,9,12,17,34,24,21,19,16,23,43,32,18,51,8,30,33,29,31,27,7,38,6,37,35,36,44,40,39,5,50,41,53],
     [.0175,.0747,.0185,.0169,.0444,.0437,.1204,.0819,.0177,.0731,.0456,.1039,.0374,.0048,.0171,.0297,.0358,.0407,.0247,.0008,.0131,.0334,.0002,.0203,.0113,.0078,.0116,.0098,.0188,.0078,.0004,.0038,.0012,.0057,.002,.0009,.0003,.0019,.0002,.0001,.0001,.0001]),
    ([16,32,24,18,23,20,21,14,27,12,11,19,30,22,29,15,9,33,25,13,28,42,26,17,10,37,31,35,44,40,38,34,8,54,36,39,6,7,45,51,46,43,5,41],
     [.0604,.0105,.0329,.0547,.0513,.0737,.0666,.0613,.0368,.0438,.0317,.0574,.0176,.042,.0243,.0474,.0102,.0041,.0379,.0347,.027,.0001,.0329,.0614,.037,.0006,.0117,.0113,.0002,.0018,.0014,.0038,.0038,.0001,.0032,.0007,.0011,.0014,.0004,.0002,.0001,.0001,.0001,.0001]),
    ([24,18,10,28,25,20,17,19,22,21,26,23,13,11,15,27,29,31,16,30,14,12,35,37,9,39,34,33,40,36,38,32,44,51,42,43,8,41,48,47,57,50,6],
     [.0436,.0472,.0306,.0318,.0489,.0881,.0441,.0643,.0513,.0697,.0438,.0591,.0213,.0097,.0414,.0474,.0316,.0226,.0644,.0244,.0337,.0224,.0124,.0026,.0024,.0006,.0066,.0095,.0027,.0064,.0021,.0099,.0006,.0001,.0011,.0005,.0003,.0006,.0002,.0002,.0001,.0001,.0001]),
    ([17,14,25,24,32,29,18,20,28,19,41,23,31,15,35,27,34,44,30,13,9,16,26,21,38,33,22,36,10,37,12,57,11,39,50,47,40,42,55,67,48,43,45],
     [.042,.0247,.0608,.0469,.0208,.0378,.042,.1022,.0458,.0554,.003,.063,.0307,.0325,.0117,.0588,.011,.0008,.0304,.0098,.0008,.0407,.0504,.0858,.0026,.0131,.0439,.0073,.0026,.0039,.0082,.0001,.0034,.001,.0005,.0003,.0035,.0006,.0001,.0001,.0004,.0001,.0001]),
    ([26,38,17,24,32,25,34,28,31,22,21,29,30,20,23,19,42,27,65,35,18,16,41,14,33,37,43,13,10,15,36,40,48,11,39,12,44,9,49,66,45,67,69,51,57,55,50],
     [.0512,.0046,.0297,.0525,.0356,.0624,.0206,.0541,.0379,.0739,.0829,.0397,.0424,.0801,.0643,.0421,.0011,.0656,.001,.018,.0326,.0236,.0024,.0109,.0238,.0047,.0009,.0034,.001,.0134,.0092,.0029,.001,.0017,.0026,.002,.001,.0012,.0003,.0003,.0002,.0002,.0001,.0001,.0006,.0001,.0001]),
    ([26,28,29,22,19,23,24,30,21,15,33,36,35,25,32,34,31,27,20,14,17,16,40,58,13,41,39,38,18,48,37,44,43,11,42,50,57,45,54,46,52,65,49,10,69,12,66,47],
     [.0613,.0816,.0496,.0506,.0245,.0567,.0565,.0562,.0537,.0079,.035,.0186,.0218,.0664,.0488,.0215,.059,.0912,.0416,.0021,.0155,.0124,.0045,.0004,.0014,.0042,.0025,.0089,.0225,.0012,.0088,.001,.0027,.0001,.002,.0004,.0014,.001,.0003,.001,.0002,.0012,.0004,.0004,.0002,.0005,.0002,.0001]),
    ([34,32,27,22,20,35,33,23,28,29,26,24,21,31,25,16,40,17,19,37,44,30,36,41,43,38,10,42,15,18,57,39,48,51,66,52,54,46,53,45,50,12,49,67,13,14,65,11,47,58,69],
     [.0293,.0612,.0847,.0332,.0387,.0219,.0424,.0557,.075,.0449,.0627,.0486,.0342,.0693,.0656,.0111,.0144,.0124,.0236,.0122,.003,.0585,.0234,.0093,.0048,.0155,.0002,.0087,.0041,.0101,.0007,.007,.0007,.0004,.0003,.0003,.0007,.0029,.0001,.0023,.0004,.0004,.0009,.0001,.0004,.0025,.0003,.0004,.0003,.0001,.0001]),
    ([32,37,26,46,28,33,27,31,30,29,34,38,21,25,24,22,36,35,41,17,23,40,39,48,42,63,45,20,15,43,54,50,12,19,44,13,58,47,51,67,57,18,56,16,49,61,55,68,52,62,11,14,53,10,59],
     [.0626,.023,.0854,.0026,.0624,.0632,.0835,.0733,.0678,.0509,.0506,.0197,.022,.0486,.04,.024,.0332,.0416,.0097,.0034,.0368,.0127,.0126,.0031,.012,.0002,.0038,.0105,.0026,.0062,.0052,.0005,.0002,.0087,.0045,.0002,.0008,.0021,.0009,.0005,.001,.0027,.0003,.0012,.001,.0003,.0002,.0001,.0006,.0001,.0001,.0004,.0002,.0001,.0001]),
    ([35,44,49,26,39,32,23,31,29,27,45,33,38,28,40,42,36,24,67,30,22,43,34,50,19,37,51,41,17,48,25,21,66,13,16,46,20,55,52,47,15,18,54,63,65,56,57,60,58,61,70,59,64,53,14,12,69,86],
     [.0521,.0663,.0027,.0381,.0427,.0696,.0094,.0604,.0383,.0423,.0087,.0815,.0379,.0491,.0163,.0175,.0474,.0367,.0022,.0413,.0095,.022,.0513,.0016,.0026,.0399,.0017,.0285,.0011,.0101,.0218,.0077,.0019,.0002,.0014,.0096,.007,.0025,.0025,.0066,.0005,.0022,.0017,.0001,.001,.0009,.0015,.0001,.0002,.0002,.0002,.0003,.0002,.0004,.0001,.0001,.0002,.0001]),
]


def get_space_size(pocket_pos):
    from scipy.spatial.distance import pdist
    dists = np.sort(pdist(pocket_pos, metric='euclidean'))[::-1]
    return np.median(dists[:10])


def _get_bin_idx(space_size):
    for i, bound in enumerate(ATOM_NUM_BOUNDS):
        if bound > space_size:
            return i
    return len(ATOM_NUM_BOUNDS)


def sample_atom_num(space_size):
    bin_idx = _get_bin_idx(space_size)
    num_atoms, probs = ATOM_NUM_BINS[bin_idx]
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(num_atoms, p=probs))


# ──────────────────────────────────────────────
# Atom Type Conversion
# ──────────────────────────────────────────────

def get_atomic_number_from_index(index):
    return [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]


def is_aromatic_from_index(index):
    return [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]


# ──────────────────────────────────────────────
# Core Sampling
# ──────────────────────────────────────────────

def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))

    for i in tqdm(range(num_batch), desc='Sampling batches'):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list(
            [data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH
        ).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch

            if sample_num_atoms == 'prior':
                pocket_size = get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [sample_atom_num(pocket_size) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(
                    torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(
                    torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError(f'Unknown sample_num_atoms: {sample_num_atoms}')

            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
            )
            ligand_pos = r['pos']
            ligand_v = r['v']

            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            v_array = ligand_v.cpu().numpy()

            for k in range(n_data):
                s, e = ligand_cum_atoms[k], ligand_cum_atoms[k + 1]
                all_pred_pos.append(pos_array[s:e])
                all_pred_v.append(v_array[s:e])

            # unbatch pos traj
            all_step_pos = [[] for _ in range(n_data)]
            for p in r['pos_traj']:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
            all_pred_pos_traj += all_step_pos

            # unbatch v traj
            all_step_v = [[] for _ in range(n_data)]
            for v in r['v_traj']:
                v_arr = v.cpu().numpy()
                for k in range(n_data):
                    all_step_v[k].append(v_arr[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_v = [np.stack(sv) for sv in all_step_v]
            all_pred_v_traj += all_step_v

            if not pos_only:
                # unbatch v0/vt traj
                all_step_v0 = [[] for _ in range(n_data)]
                for v0 in r['v0_traj']:
                    v0_arr = v0.cpu().numpy()
                    for k in range(n_data):
                        all_step_v0[k].append(v0_arr[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
                all_pred_v0_traj += [np.stack(sv) for sv in all_step_v0]

                all_step_vt = [[] for _ in range(n_data)]
                for vt in r['vt_traj']:
                    vt_arr = vt.cpu().numpy()
                    for k in range(n_data):
                        all_step_vt[k].append(vt_arr[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
                all_pred_vt_traj += [np.stack(sv) for sv in all_step_vt]

        time_list.append(time.time() - t1)

    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


# ──────────────────────────────────────────────
# Pocket PDB → Data object (for --pdb_path mode)
# ──────────────────────────────────────────────

def pdb_to_data(pdb_path):
    from build_lmdb import parse_pdb
    protein_dict = parse_pdb(pdb_path)
    data = ProteinLigandData()
    data.protein_element = torch.tensor(protein_dict['element'])
    data.protein_pos = torch.tensor(protein_dict['pos'])
    data.protein_is_backbone = torch.tensor(protein_dict['is_backbone'])
    data.protein_atom_to_aa_type = torch.tensor(protein_dict['atom_to_aa_type'])
    data.ligand_element = torch.empty([0], dtype=torch.long)
    data.ligand_pos = torch.empty([0, 3], dtype=torch.float)
    data.ligand_atom_feature = torch.empty([0, 8], dtype=torch.float)
    data.ligand_bond_index = torch.empty([2, 0], dtype=torch.long)
    data.ligand_bond_type = torch.empty([0], dtype=torch.long)
    return data


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir is not None:
        fh = logging.FileHandler(os.path.join(log_dir, 'sampling_log.txt'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def resolve_lmdb_path(data_cfg):
    candidate = (
        data_cfg.get('lmdb_path')
        or data_cfg.get('lmdb')
        or data_cfg.get('path')
    )
    if candidate is None:
        return './best_affinity_complex_processed.lmdb'
    if os.path.exists(candidate):
        return candidate
    lmdb_candidate = f'{candidate}.lmdb'
    if os.path.exists(lmdb_candidate):
        return lmdb_candidate
    return candidate


def get_model_config(ckpt_config):
    if isinstance(ckpt_config, dict) and 'model' in ckpt_config:
        return SimpleNamespace(**ckpt_config['model'])
    return SimpleNamespace(**ckpt_config)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='sampling.yml')
    parser.add_argument('--pdb_path', type=str, default=None,
                        help='Pocket PDB path (single pocket mode)')
    parser.add_argument('--data_id', type=int, default=None,
                        help='Test set index (0~99)')
    parser.add_argument('--split_path', type=str, default=None,
                        help='Path to split .pt file (CLI override)')
    parser.add_argument('--lmdb_path', type=str, default=None,
                        help='Path to LMDB file (CLI override)')
    parser.add_argument('--result_path', type=str, default='./sampling_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--sample_num_atoms', type=str, default=None,
                        choices=['prior', 'ref'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sample_cfg = yaml.safe_load(f)

    ckpt_path = sample_cfg['model']['checkpoint']

    # ── data config (yml → CLI override 순서) ──
    data_section = sample_cfg.get('data', {})
    split_path = data_section.get('split', None)
    lmdb_path = resolve_lmdb_path(data_section)

    sample_section = sample_cfg.get('sample', {})
    sample_num_samples = sample_section.get('num_samples', 100)
    sample_num_steps   = sample_section.get('num_steps', None)
    sample_pos_only    = sample_section.get('pos_only', False)
    sample_center_mode = sample_section.get('center_pos_mode', None)
    sample_num_atoms   = sample_section.get('sample_num_atoms', 'prior')
    sample_seed        = sample_section.get('seed', 2021)

    # CLI override
    if args.split_path is not None:
        split_path = args.split_path
    if args.lmdb_path is not None:
        lmdb_path = args.lmdb_path
    if args.num_samples is not None:
        sample_num_samples = args.num_samples
    if args.num_steps is not None:
        sample_num_steps = args.num_steps
    if args.sample_num_atoms is not None:
        sample_num_atoms = args.sample_num_atoms
    if args.seed is not None:
        sample_seed = args.seed

    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)

    os.makedirs(args.result_path, exist_ok=True)
    logger = get_logger('sampling', args.result_path)

    # ── Load checkpoint ──
    logger.info(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model_cfg = get_model_config(ckpt['config'])

    # ── Build model ──
    transform, protein_featurizer, ligand_featurizer = get_transforms()
    model = ScorePosNet3D(
        model_cfg,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info('Model loaded.')

    # ── Load data ──
    if args.pdb_path is not None:
        logger.info(f'Loading pocket from PDB: {args.pdb_path}')
        data = pdb_to_data(args.pdb_path)
        from torch_geometric.transforms import Compose
        pocket_transform = Compose([protein_featurizer])
        data = pocket_transform(data)

    elif args.data_id is not None:
        if split_path is None:
            raise ValueError('split_path must be provided via sampling.yml (data.split) or --split_path')
        logger.info(f'Loading test data_id={args.data_id} | split: {split_path} | lmdb: {lmdb_path}')
        split = torch.load(split_path)
        real_idx = split['test'][args.data_id]
        dataset = LMDBDataset(lmdb_path, transform=transform)
        data = dataset[real_idx]

    else:
        raise ValueError('Either --pdb_path or --data_id must be provided.')

    # ── Sample ──
    logger.info(f'Sampling {sample_num_samples} ligands (batch_size={args.batch_size})...')
    pred_pos, pred_v, pos_traj, v_traj, v0_traj, vt_traj, time_list = sample_diffusion_ligand(
        model, data, sample_num_samples,
        batch_size=args.batch_size,
        device=args.device,
        num_steps=sample_num_steps,
        pos_only=sample_pos_only,
        center_pos_mode=sample_center_mode or model_cfg.center_pos_mode,
        sample_num_atoms=sample_num_atoms,
    )
    logger.info(f'Sampling done. Time per batch: {np.mean(time_list):.2f}s')

    # ── Save raw results ──
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pos_traj,
        'pred_ligand_v_traj': v_traj,
        'time': time_list,
    }
    torch.save(result, os.path.join(args.result_path, 'result.pt'))

    # ── Reconstruct molecules ──
    logger.info('Reconstructing molecules...')
    sdf_dir = os.path.join(args.result_path, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    n_recon, n_complete = 0, 0
    gen_mols = []
    for idx, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        atomic_nums = get_atomic_number_from_index(v)
        aromatic = is_aromatic_from_index(v)
        try:
            mol = reconstruct_from_generated(pos, atomic_nums, aromatic)
            smiles = Chem.MolToSmiles(mol)
        except MolReconsError:
            gen_mols.append(None)
            continue
        n_recon += 1

        if '.' in smiles:
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)

        writer = Chem.SDWriter(os.path.join(sdf_dir, f'{idx:03d}.sdf'))
        writer.write(mol)
        writer.close()

    result['mols'] = gen_mols
    torch.save(result, os.path.join(args.result_path, 'result.pt'))

    logger.info(f'Reconstruction: {n_recon}/{len(pred_pos)} success, '
                f'{n_complete}/{len(pred_pos)} complete (no fragments)')
    logger.info(f'Results saved to {args.result_path}')
