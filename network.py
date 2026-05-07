import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax, scatter_sum


# ──────────────────────────────────────────────
# Common 부분에 있는 Utility들
# ──────────────────────────────────────────────

NONLINEARITIES = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'softplus': nn.Softplus(),
    'elu': nn.ELU(),
    'silu': nn.SiLU(),
}


class GaussianSmearing(nn.Module):  # 거리를 Gaussian 함수로 변환
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ShiftedSoftplus(nn.Module):   # Softplus 함수의 출력을 0을 중심(bias=0)으로 이동시키는 역할을 하는 함수. 일반 Softplus는 항상 0보다 크지만, ShiftedSoftplus는 출력을 쉬프트하여 입력이 0일 때 0이 되도록 만듬.
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand): # protein-ligand 그래프 합치기 (노드병합) (diffusion.py 에서 호출할 함수임)
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]
    return h_ctx, pos_ctx, batch_ctx, mask_ligand


# ──────────────────────────────────────────────
# Hybrid Edge Construction (실제론 이거 안쓰고, 밑에있는 knn 사용함)
# ──────────────────────────────────────────────

def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index): # ligand-ligand, ligand-protein KNN 그래프 생성 (엣지생성)
    # ligand-ligand: fully connected
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    ll_edge_index = torch.stack([src[mask], dst[mask]])

    # ligand-protein: kNN
    dist = torch.norm(
        ligand_pos.unsqueeze(1) - protein_pos.unsqueeze(0), p=2, dim=-1
    )
    knn_p_idx = protein_index[torch.topk(dist, k=k, largest=False, dim=1).indices]
    knn_l_idx = ligand_index.unsqueeze(1).repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0).view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=True): # 배치의 그래프별 hybrid edge 생성
    batch_size = batch.max().item() + 1
    all_edges = []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & mask_ligand).nonzero()[:, 0]
            protein_index = ((batch == i) & ~mask_ligand).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]

            ll_edge, pl_edge = hybrid_edge_connection( # ligand-ligand, ligand-protein KNN 그래프 생성 (엣지생성)
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            edges = [ll_edge, pl_edge]

            if add_p_index: # protein-protein KNN 그래프 생성 (엣지생성)
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge = p_edge[:, p_edge[1] < len(protein_pos)]
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge = torch.stack([all_index[p_edge[0]], all_index[p_edge[1]]], 0)
                edges.append(p_edge)

            all_edges.append(torch.cat(edges, dim=-1))
    return torch.cat(all_edges, dim=-1)


# ──────────────────────────────────────────────
# Attention Layers
# ──────────────────────────────────────────────

class BaseX2HAttLayer(nn.Module): # X2H 레이어 (Geometry → hidden state update via multi-head attention.)

    def __init__(self, input_dim, hidden_dim, output_dim, n_heads,
                 edge_feat_dim, r_feat_dim, act_fn='relu', norm=True,
                 ew_net_type='r', out_fc=True):
        super().__init__()
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        head_dim = self.output_dim // self.n_heads

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.hk_func(kv_input).view(-1, self.n_heads, head_dim)
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.output_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.0
        v = (v * e_w).view(-1, self.n_heads, head_dim)

        q = self.hq_func(h).view(-1, self.n_heads, head_dim)
        alpha = scatter_softmax(
            (q[dst] * k / np.sqrt(head_dim)).sum(-1), dst, dim=0, dim_size=N
        )

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N).view(-1, self.output_dim)

        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))
        return output + h


class BaseH2XAttLayer(nn.Module): # H2X 레이어 (hidden state → coordinate update via equivariant attention.)

    def __init__(self, input_dim, hidden_dim, output_dim, n_heads,
                 edge_feat_dim, r_feat_dim, act_fn='relu', norm=True,
                 ew_net_type='r'):
        super().__init__()
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        head_dim = self.output_dim // self.n_heads

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, head_dim)
        v = self.xv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.0
        v = (v * e_w).unsqueeze(-1) * rel_x.unsqueeze(1)  # [E, heads, 3]

        q = self.xq_func(h).view(-1, self.n_heads, head_dim)
        alpha = scatter_softmax(
            (q[dst] * k / np.sqrt(head_dim)).sum(-1), dst, dim=0, dim_size=N
        )

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # [N, heads, 3]
        return output.mean(dim=1)


# ──────────────────────────────────────────────
# Attention Block (X2H + H2X combined)
# ──────────────────────────────────────────────

class AttentionBlock(nn.Module): # X2H + H2X 레이어 (One block: num_x2h × X2H layers then num_h2x × H2X layers.)

    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim,
                 act_fn='relu', norm=True, num_x2h=1, num_h2x=1,
                 r_min=0., r_max=10., num_node_types=8, ew_net_type='r',
                 x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.sync_twoup = sync_twoup
        self.num_node_types = num_node_types
        r_feat_dim = num_r_gaussian * 4  # 4 edge types × gaussian features

        self.distance_expansion = GaussianSmearing(r_min, r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList([
            BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads,
                            edge_feat_dim, r_feat_dim, act_fn=act_fn, norm=norm,
                            ew_net_type=ew_net_type, out_fc=x2h_out_fc)
            for _ in range(num_x2h)
        ])
        self.h2x_layers = nn.ModuleList([
            BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads,
                            edge_feat_dim, r_feat_dim, act_fn=act_fn, norm=norm,
                            ew_net_type=ew_net_type)
            for _ in range(num_h2x)
        ])

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        edge_feat = edge_attr if edge_attr.size(-1) > 0 else None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        # X2H: update hidden states
        h_in = h
        for layer in self.x2h_layers:
            dist_feat = outer_product(edge_attr, self.distance_expansion(dist))
            h_in = layer(h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
        x2h_out = h_in

        # H2X: update ligand coordinates
        new_h = h if self.sync_twoup else x2h_out
        for layer in self.h2x_layers:
            dist_feat = outer_product(edge_attr, self.distance_expansion(dist))
            delta_x = layer(new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


# ──────────────────────────────────────────────
# UniTransformer (full backbone)
# ──────────────────────────────────────────────

class UniTransformer(nn.Module): # SE(3)-equivariant transformer backbone. 메인 backbone 모델

    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='hybrid', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0,
                 num_x2h=1, num_h2x=1, r_max=10.,
                 x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.k = k
        self.cutoff_mode = cutoff_mode
        self.ew_net_type = ew_net_type
        self.r_max = r_max
        self.num_node_types = num_node_types
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)

        if ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        block_kwargs = dict(
            hidden_dim=hidden_dim, n_heads=n_heads,
            num_r_gaussian=num_r_gaussian, edge_feat_dim=edge_feat_dim,
            act_fn=act_fn, norm=norm, r_max=r_max,
            num_node_types=num_node_types,
            ew_net_type=ew_net_type, x2h_out_fc=x2h_out_fc,
            sync_twoup=sync_twoup,
        )

        self.init_block = AttentionBlock(
            num_x2h=num_init_x2h, num_h2x=num_init_h2x, **block_kwargs)

        self.blocks = nn.ModuleList([
            AttentionBlock(num_x2h=num_x2h, num_h2x=num_h2x, **block_kwargs)
            for _ in range(num_layers)
        ])

    # ── Edge construction ──

    def _connect_edge(self, x, mask_ligand, batch): # 엣지 생성 (KNN, Hybrid)
        if self.cutoff_mode == 'knn':
            return knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            return batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Unsupported cutoff_mode: {self.cutoff_mode}')

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand): # 엣지 타입 생성 (0: ligand-ligand, 1: ligand-protein, 2: protein-ligand, 3: protein-protein)
        src, dst = edge_index
        n_src = mask_ligand[src]
        n_dst = mask_ligand[dst]
        edge_type = torch.zeros(len(src), device=edge_index.device, dtype=torch.long)
        edge_type[n_src & n_dst] = 0    # ligand-ligand
        edge_type[n_src & ~n_dst] = 1   # ligand-protein
        edge_type[~n_src & n_dst] = 2   # protein-ligand
        edge_type[~n_src & ~n_dst] = 3  # protein-protein
        return F.one_hot(edge_type, num_classes=4).float()

    # ── Forward ──

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        all_x, all_h = [x], [h]

        for _ in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            edge_type = self._build_edge_type(edge_index, mask_ligand)

            if self.ew_net_type == 'global':
                src, dst = edge_index
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                e_w = torch.sigmoid(self.edge_pred_layer(self.distance_expansion(dist)))
            else:
                e_w = None

            for layer in self.blocks:
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)

            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
