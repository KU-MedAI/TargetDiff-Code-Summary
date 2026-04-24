import os
import pickle
import lmdb
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_scatter import scatter_add

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

AROMATIC_FEAT_MAP_IDX = 2  # index of 'Aromatic' in ATOM_FAMILIES

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12,
}

MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type')


# ──────────────────────────────────────────────
# Data Container
# ──────────────────────────────────────────────

class ProteinLigandData(Data): 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        return super().__inc__(key, value)


# ──────────────────────────────────────────────
# LMDB Dataset
# ──────────────────────────────────────────────

class LMDBDataset(Dataset):
    """LMDB 파일에서 (protein, ligand) pair를 읽어옴."""

    def __init__(self, lmdb_path, transform=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.db = None
        self.keys = None

    def _connect(self):
        self.db = lmdb.open(   # LMDB 파일 열기
            self.lmdb_path,
            map_size=10 * (1024 ** 3),   # 메모리 크기 설정
            create=False, subdir=False,   # 데이터베이스 생성 여부 설정
            readonly=True, lock=False,   # 읽기 전용 여부 설정
            readahead=False, meminit=False,   # 메모리 초기화 여부 설정
        )
        with self.db.begin() as txn:   # 데이터베이스 시작
            self.keys = list(txn.cursor().iternext(values=False))   # 데이터베이스 키 리스트 가져오기

    def __len__(self):
        if self.db is None:
            self._connect()
        return len(self.keys)   # 데이터베이스 크기

    def __getitem__(self, idx):
        if self.db is None:
            self._connect()

        raw = pickle.loads(self.db.begin().get(self.keys[idx]))   # 데이터베이스에서 데이터 가져오기
        data = self._to_data(raw)   # 데이터 변환
        data.id = idx   # 데이터 인덱스 추가

        if self.transform is not None:
            data = self.transform(data)   # 데이터 변환
        return data

    @staticmethod
    def _to_data(raw: dict) -> ProteinLigandData:
        data = ProteinLigandData()
        for k, v in raw.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
            else:
                data[k] = v

        # neighbour lookup (used in molecular reconstruction)
        bond_idx = data.ligand_bond_index   # 리간드 결합 인덱스
        nbh = {}   # 이웃 리스트
        for i in range(bond_idx.size(1)):
            src = bond_idx[0, i].item()   # 소스 노드
            dst = bond_idx[1, i].item()   # 목적지 노드
            nbh.setdefault(src, []).append(dst)
        data.ligand_nbh_list = nbh   # 이웃 리스트 추가
        return data


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

class FeaturizeProteinAtom:
    """One-hot element (H,C,N,O,S,Se) + one-hot AA type (20) + is_backbone (1)."""

    def __init__(self):
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        data.protein_atom_feature = torch.cat([element, amino_acid, is_backbone], dim=-1)
        return data


class FeaturizeLigandAtom:
    """(atomic_number, is_aromatic) → integer class index로 변환."""

    def __init__(self):
        pass

    @property
    def feature_dim(self):
        return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)

    def __call__(self, data):
        element_list = data.ligand_element
        hybridization_list = getattr(data, 'ligand_hybridization', [None] * len(element_list))
        aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]

        x = []
        for e, h, a in zip(element_list, hybridization_list, aromatic_list):
            key = (int(e), bool(a))
            if key in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
                x.append(MAP_ATOM_TYPE_AROMATIC_TO_INDEX[key])
            else:
                x.append(MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)])
        data.ligand_atom_feature_full = torch.tensor(x, dtype=torch.long)
        return data


class LigandCountNeighbors:
    """리간드 원자의 결합 수와 DEGREE를 계산"""

    def __call__(self, data):
        edge_index = data.ligand_bond_index   # 리간드 결합 인덱스
        num_nodes = data.ligand_element.size(0)   # 리간드 원자 수

        ones = torch.ones(edge_index.size(1), dtype=torch.long) # 결합 개수만큼 1 만듬
        data.ligand_num_neighbors = scatter_add(ones, edge_index[0], dim=0, dim_size=num_nodes) # 결합 수 계산

        valence = data.ligand_bond_type.long()
        data.ligand_atom_valence = scatter_add(valence, edge_index[0], dim=0, dim_size=num_nodes) # DEGREE 계산
        return data


class NormalizePosition:
    """protein centroid를 중심으로 Pocket, Ligand Center화. """

    def __call__(self, data):
        center = data.protein_pos.mean(dim=0, keepdim=True)
        data.protein_pos = data.protein_pos - center
        data.ligand_pos = data.ligand_pos - center
        data.center_of_mass_offset = center.squeeze(0)
        return data


# ──────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────

def get_transforms():
    protein_featurizer = FeaturizeProteinAtom()   # protein 원자 특성 추출
    ligand_featurizer = FeaturizeLigandAtom()   # ligand 원자 특성 추출
    from torch_geometric.transforms import Compose
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        LigandCountNeighbors(),   # 리간드 원자의 결합 수와 DEGREE 계산
    ])
    return transform, protein_featurizer, ligand_featurizer


if __name__ == '__main__':
    transform, pf, lf = get_transforms()
    lmdb_path = '/scratch/x3317a09/scripts/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
    ds = LMDBDataset(lmdb_path, transform=transform)

    print(f'Dataset size: {len(ds)}')
    sample = ds[0]
    print(f'\n--- Sample 0: {sample.complex_name} ---')
    print(f'  protein_pos:              {sample.protein_pos.shape}')
    print(f'  protein_atom_feature:     {sample.protein_atom_feature.shape}  (dim={pf.feature_dim})')
    print(f'  ligand_pos:               {sample.ligand_pos.shape}')
    print(f'  ligand_atom_feature_full: {sample.ligand_atom_feature_full.shape}  (dim={lf.feature_dim})')
    print(f'  ligand_bond_index:        {sample.ligand_bond_index.shape}')
    print(f'  ligand_num_neighbors:     {sample.ligand_num_neighbors.shape}')
    print(f'  ligand_atom_valence:      {sample.ligand_atom_valence.shape}')
    if hasattr(sample, 'center_of_mass_offset'):
        print(f'  center_of_mass_offset:    {sample.center_of_mass_offset.shape}')
