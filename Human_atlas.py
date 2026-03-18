# %% [markdown]
# # Human Immune Multiomics Atlas
# ## Multimodal Deep Learning for Joint scRNA-seq + scATAC-seq Integration
# 
# **Obiettivo**: Costruire un Multimodal Variational Autoencoder (MVAE) che apprende una rappresentazione latente congiunta da dati trascritomici e epigenomici single-cell, permettendo:
# - Integrazione non-lineare delle due modalità
# - Predizione cross-modale (RNA ↔ ATAC)
# - Clustering nel manifold latente congiunto
# - Classificazione dei tipi cellulari immunitari

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("qasimhu/human-immune-multiomics-atlas")

print("Path to dataset files:", path)

# %%
import warnings
warnings.filterwarnings("ignore")

# Blocca TensorFlow (incompatibile con NumPy 2.x, importato da umap)
import sys, types
from importlib.machinery import ModuleSpec

_fake_tf = types.ModuleType("tensorflow")
_fake_tf.__spec__ = ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _fake_tf

_fake_tfk = types.ModuleType("tensorflow.keras")
_fake_tfk.__spec__ = ModuleSpec("tensorflow.keras", None)
sys.modules["tensorflow.keras"] = _fake_tfk

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import scanpy as sc
import muon as mu
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

import umap
from tqdm.auto import tqdm

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 5))
plt.rcParams["font.family"] = "sans-serif"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## Data loading & Exploration
# Carichiamo il MuData object contenente entrambe le modalità (RNA + ATAC) già allineate per barcode cellulare.

# %%
DATA_PATH = Path.home() / ".cache/kagglehub/datasets/qasimhu/human-immune-multiomics-atlas/versions/1"
mdata = mu.read(DATA_PATH / "data/interim/pbmc_multiome.h5mu")
print(mdata)
print(f"\nRNA: {mdata['rna'].shape}  |  ATAC: {mdata['atac'].shape}")
print(f"Shared cells: {mdata.n_obs}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

rna_counts = np.array(mdata["rna"].X.sum(axis=1)).flatten()
axes[0].hist(rna_counts, bins=80, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].set_title("RNA — Total UMI per Cell")
axes[0].set_xlabel("Total counts")
axes[0].axvline(np.median(rna_counts), color="red", ls="--", label=f"Median: {np.median(rna_counts):.0f}")
axes[0].legend()

atac_counts = np.array(mdata["atac"].X.sum(axis=1)).flatten()
axes[1].hist(atac_counts, bins=80, color="#FF9800", edgecolor="white", alpha=0.85)
axes[1].set_title("ATAC — Total Fragments per Cell")
axes[1].set_xlabel("Total counts")
axes[1].axvline(np.median(atac_counts), color="red", ls="--", label=f"Median: {np.median(atac_counts):.0f}")
axes[1].legend()

rna_sparsity = 1 - mdata["rna"].X.nnz / np.prod(mdata["rna"].X.shape)
atac_sparsity = 1 - mdata["atac"].X.nnz / np.prod(mdata["atac"].X.shape)
bars = axes[2].bar(["RNA", "ATAC"], [rna_sparsity * 100, atac_sparsity * 100],
                   color=["#2196F3", "#FF9800"], edgecolor="white")
axes[2].set_ylabel("% Zero Entries")
axes[2].set_title("Matrix Sparsity")
for bar, val in zip(bars, [rna_sparsity, atac_sparsity]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val*100:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Quality control & Preprocessing
# ### RNA: Filtro geni, normalizzazione log, HVG selection
# ### ATAC: TF-IDF normalization, feature selection per varianza

# %%
rna = mdata["rna"].copy()
rna.var_names_make_unique()

sc.pp.calculate_qc_metrics(rna, percent_top=None, log1p=True, inplace=True)
rna.var["mt"] = rna.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
sc.pl.violin(rna, "n_genes_by_counts", ax=axes[0], show=False)
sc.pl.violin(rna, "total_counts", ax=axes[1], show=False)
sc.pl.violin(rna, "pct_counts_mt", ax=axes[2], show=False)
plt.tight_layout()
plt.show()

print(f"Pre-filter: {rna.shape}")
sc.pp.filter_cells(rna, min_genes=200)
sc.pp.filter_genes(rna, min_cells=10)
rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
print(f"Post-filter: {rna.shape}")

rna.layers["raw_counts"] = rna.X.copy()
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

sc.pp.highly_variable_genes(rna, n_top_genes=3000, flavor="seurat_v3", layer="raw_counts")
print(f"HVGs selected: {rna.var['highly_variable'].sum()}")

mdata.mod["rna"] = rna

# %%
atac = mdata["atac"].copy()

sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=True, inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(np.array(atac.X.sum(1)).flatten(), bins=80, color="#FF9800", edgecolor="white")
axes[0].set_title("ATAC — Fragments per Cell")
axes[1].hist(np.log10(np.array(atac.X.sum(0)).flatten() + 1), bins=80, color="#4CAF50", edgecolor="white")
axes[1].set_title("ATAC — log10(Cells per Peak)")
plt.tight_layout()
plt.show()

print(f"Pre-filter peaks: {atac.shape[1]}")
sc.pp.filter_genes(atac, min_cells=50)  # picchi presenti in almeno 50 cellule
print(f"Post-filter peaks: {atac.shape[1]}")

def tfidf_normalize(adata):
    """Term Frequency — Inverse Document Frequency per scATAC-seq."""
    X = adata.X.copy()
    if sp.issparse(X):
        X = X.astype(np.float64)
    tf = X.multiply(1.0 / X.sum(axis=1))
    n_cells = X.shape[0]
    idf = np.log1p(n_cells / (np.array(X.sum(axis=0)).flatten() + 1))
    tfidf = tf.multiply(idf)
    adata.X = sp.csr_matrix(tfidf)
    return adata

atac.layers["raw_counts"] = atac.X.copy()
atac = tfidf_normalize(atac)

peak_var = np.array(atac.X.toarray().var(axis=0)) if sp.issparse(atac.X) else atac.X.var(axis=0)
n_top_peaks = 15000
top_idx = np.argsort(peak_var)[::-1][:n_top_peaks]
atac.var["highly_variable"] = False
atac.var.iloc[top_idx, atac.var.columns.get_loc("highly_variable")] = True
print(f"Top variable peaks selected: {n_top_peaks}")

mdata.mod["atac"] = atac

# %% [markdown]
# ## Unimodal analysis
# Riduzione dimensionale e clustering indipendente per ciascuna modalità, come baseline da confrontare con l'integrazione multimodale.

# %%
rna_hvg = rna[:, rna.var["highly_variable"]].copy()
sc.pp.scale(rna_hvg, max_value=10)
sc.tl.pca(rna_hvg, n_comps=50, svd_solver="arpack")

sc.pp.neighbors(rna_hvg, n_neighbors=30, n_pcs=30)
sc.tl.leiden(rna_hvg, resolution=0.8, key_added="leiden_rna")
sc.tl.umap(rna_hvg)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(rna_hvg, color="leiden_rna", ax=axes[0], show=False, 
           title="RNA — Leiden Clusters", legend_loc="on data", frameon=False)
sc.pl.umap(rna_hvg, color="log1p_total_counts", ax=axes[1], show=False,
           title="RNA — Total Counts (log)", frameon=False, color_map="viridis")
plt.tight_layout()
plt.show()

print(f"RNA clusters: {rna_hvg.obs['leiden_rna'].nunique()}")
rna.obs["leiden_rna"] = rna_hvg.obs["leiden_rna"]
rna.obsm["X_umap_rna"] = rna_hvg.obsm["X_umap"]
rna.obsm["X_pca"] = rna_hvg.obsm["X_pca"]

# %%
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

atac_hvp = atac[:, atac.var["highly_variable"]].copy()

n_components = 50
svd = TruncatedSVD(n_components=n_components + 1, random_state=SEED)
X_lsi = svd.fit_transform(atac_hvp.X)
X_lsi = X_lsi[:, 1:] # rimuovi prima componente (depth)

atac.obsm["X_lsi"] = X_lsi
print(f"LSI explained variance (comp 2-51): {svd.explained_variance_ratio_[1:].sum():.3f}")

atac_hvp.obsm["X_lsi"] = X_lsi
sc.pp.neighbors(atac_hvp, use_rep="X_lsi", n_neighbors=30)
sc.tl.leiden(atac_hvp, resolution=0.6, key_added="leiden_atac")
sc.tl.umap(atac_hvp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(atac_hvp, color="leiden_atac", ax=axes[0], show=False,
           title="ATAC — Leiden Clusters (LSI)", legend_loc="on data", frameon=False)
sc.pl.umap(atac_hvp, color="log1p_total_counts", ax=axes[1], show=False,
           title="ATAC — Total Fragments (log)", frameon=False, color_map="magma")
plt.tight_layout()
plt.show()

atac.obs["leiden_atac"] = atac_hvp.obs["leiden_atac"]
atac.obsm["X_umap_atac"] = atac_hvp.obsm["X_umap"]

# %%
MARKERS = {
    "CD14+ Mono": ["CD14", "LYZ", "S100A9", "S100A8"],
    "CD16+ Mono": ["FCGR3A", "MS4A7", "CDKN1C"],
    "CD4+ T naive": ["IL7R", "CCR7", "LEF1", "TCF7"],
    "CD4+ T memory": ["IL7R", "S100A4", "IL32", "ANXA1"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "CCL5"],
    "NK": ["GNLY", "NKG7", "KLRD1", "PRF1"],
    "B naive": ["MS4A1", "CD79A", "TCL1A", "IGHD"],
    "B memory": ["MS4A1", "CD79A", "AIM2", "TNFRSF13B"],
    "Plasmablast": ["MZB1", "JCHAIN", "XBP1", "IRF4"],
    "DC": ["FCER1A", "CLEC10A", "CD1C"],
    "pDC": ["LILRA4", "IRF7", "CLEC4C"],
    "Platelet": ["PPBP", "PF4", "GP9"],
}

rna_score = rna.copy()
sc.pp.scale(rna_score, max_value=10)

for ct, genes in MARKERS.items():
    genes_found = [g for g in genes if g in rna_score.var_names]
    if genes_found:
        sc.tl.score_genes(rna_score, gene_list=genes_found, score_name=ct)

score_cols = [ct for ct in MARKERS.keys() if ct in rna_score.obs.columns]

# --- Cluster-based annotation (robust vs per-cell idxmax) ---
# Per ogni Leiden cluster, calcola il punteggio medio per ciascun tipo cellulare
# e assegna il tipo con score medio più alto all'intero cluster.
# Questo elimina il rumore delle annotazioni per-cellula.
rna_score.obs["leiden_rna"] = rna.obs["leiden_rna"]
cluster_annotations = {}
for cluster_id in sorted(rna_score.obs["leiden_rna"].unique(), key=int):
    mask = rna_score.obs["leiden_rna"] == cluster_id
    cluster_means = rna_score.obs.loc[mask, score_cols].mean()
    best_ct = cluster_means.idxmax()
    cluster_annotations[cluster_id] = best_ct

print("Cluster → Cell type mapping:")
for k, v in sorted(cluster_annotations.items(), key=lambda x: int(x[0])):
    n_cells = (rna_score.obs["leiden_rna"] == k).sum()
    print(f"  Cluster {k:>2s}: {v:<20s} ({n_cells} cells)")

cell_types = rna_score.obs["leiden_rna"].map(cluster_annotations)
cell_types.name = "cell_type"
rna.obs["cell_type"] = cell_types.values

shared_cells = rna.obs_names.intersection(atac.obs_names)
atac = atac[shared_cells].copy()
rna = rna[shared_cells].copy()
atac.obs["cell_type"] = rna.obs.loc[shared_cells, "cell_type"].values
mdata.mod["rna"] = rna
mdata.mod["atac"] = atac

print(f"\nAligned cells: {len(shared_cells)}")

rna_hvg = rna_hvg[rna_hvg.obs_names.isin(shared_cells)].copy()
rna_hvg.obs["cell_type"] = rna.obs.loc[rna_hvg.obs_names, "cell_type"].values
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(rna_hvg, color="cell_type", ax=ax, show=False,
           title="RNA UMAP — Cluster-based Cell Types", frameon=False,
           palette="tab20")
plt.tight_layout()
plt.show()

print(rna.obs["cell_type"].value_counts())

# %% [markdown]
# ## Multimodal Variational Autoencoder (MVAE v2)
# 
# Architettura **Product-of-Experts (PoE)** con miglioramenti significativi rispetto a v1:
# 
# - **Encoder RNA** $q_\phi(z|x_{rna})$: Residual MLP (3000 HVG → 512 hidden → $z_{20}$)
# - **Encoder ATAC** $q_\psi(z|x_{atac})$: Residual MLP (50 LSI → 256 hidden → $z_{20}$) — **LSI input** per signal denso
# - **PoE fusion**: $q(z|x_{rna}, x_{atac}) = \text{PoE}(q_\phi, q_\psi, p(z))$
# - **Decoder RNA** $p_\theta(x_{rna}|z)$: Residual MLP → Negative Binomial con **per-cell library size**
# - **Decoder ATAC** $p_\omega(x_{atac}|z)$: Residual MLP → Bernoulli (15k peaks)
# 
# **Key improvements vs v1**: LSI input per ATAC encoder, per-cell library size (non più scalare globale), residual blocks, latent dim 20 (da 64), cyclical KL annealing (Fu et al. 2019).

# %%
class MultiomeDataset(Dataset):
    """Dataset per dati multiomici RNA + ATAC con LSI features e per-cell library size."""
    
    def __init__(self, rna_adata, atac_adata, rna_hvg_mask, atac_hvp_mask):
        shared = rna_adata.obs_names.intersection(atac_adata.obs_names)
        rna_sub = rna_adata[shared]
        atac_sub = atac_adata[shared]
        
        # RNA: log-normalized HVGs + raw counts per NB loss
        self.rna_raw = self._to_dense(rna_sub[:, rna_hvg_mask].layers["raw_counts"])
        self.rna_norm = self._to_dense(rna_sub[:, rna_hvg_mask].X)
        
        # ATAC encoder input: LSI components (50-dim, dense)
        # Standardize per-component per convergenza stabile
        atac_lsi = atac_sub.obsm["X_lsi"].astype(np.float32).copy()
        lsi_mean = atac_lsi.mean(axis=0, keepdims=True)
        lsi_std = atac_lsi.std(axis=0, keepdims=True) + 1e-8
        self.atac_lsi = ((atac_lsi - lsi_mean) / lsi_std).astype(np.float32)
        
        # ATAC decoder target: binary peak accessibility (15k peaks)
        atac_raw = self._to_dense(atac_sub[:, atac_hvp_mask].layers["raw_counts"])
        self.atac_binary = (atac_raw > 0).astype(np.float32)
        
        # Per-cell RNA library size (log scale) per Negative Binomial
        raw_sums = self.rna_raw.sum(axis=1, keepdims=True)
        self.log_library = np.log(raw_sums + 1).astype(np.float32)
        
        self.n_cells = len(shared)
        self.n_genes = self.rna_norm.shape[1]
        self.n_peaks = self.atac_binary.shape[1]
        self.n_lsi = self.atac_lsi.shape[1]
        
        print(f"Dataset: {self.n_cells} cells | {self.n_genes} genes | "
              f"{self.n_peaks} peaks | {self.n_lsi} LSI components")
        print(f"Library size range: [{np.exp(self.log_library.min()):.0f}, "
              f"{np.exp(self.log_library.max()):.0f}] "
              f"(median: {np.exp(np.median(self.log_library)):.0f})")
    
    @staticmethod
    def _to_dense(X):
        if sp.issparse(X):
            return np.array(X.todense(), dtype=np.float32)
        return np.array(X, dtype=np.float32)
    
    def __len__(self):
        return self.n_cells
    
    def __getitem__(self, idx):
        return {
            "rna_norm": torch.tensor(self.rna_norm[idx]),
            "rna_raw": torch.tensor(self.rna_raw[idx]),
            "atac_lsi": torch.tensor(self.atac_lsi[idx]),
            "atac_binary": torch.tensor(self.atac_binary[idx]),
            "log_library": torch.tensor(self.log_library[idx]),
        }

dataset = MultiomeDataset(
    rna, atac,
    rna_hvg_mask=rna.var["highly_variable"].values,
    atac_hvp_mask=atac.var["highly_variable"].values,
)

n_val = int(0.15 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], 
                                generator=torch.Generator().manual_seed(SEED))

BATCH_SIZE = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

print(f"Train: {n_train} | Val: {n_val}")

# %%
class ResidualEncoder(nn.Module):
    """Encoder con residual blocks per gradient flow stabile e maggiore capacità."""
    
    def __init__(self, n_input, n_latent, hidden_dim=256, n_blocks=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.mu = nn.Linear(hidden_dim, n_latent)
        self.logvar = nn.Linear(hidden_dim, n_latent)
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.mu(h), self.logvar(h)


class ResidualDecoder(nn.Module):
    """Decoder con residual blocks."""
    
    def __init__(self, n_latent, n_output, hidden_dim=256, n_blocks=2, output_activation=None):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.output = nn.Linear(hidden_dim, n_output)
        self.output_activation = output_activation
    
    def forward(self, z):
        h = self.input_proj(z)
        for block in self.blocks:
            h = h + block(h)
        out = self.output(h)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class MVAE(nn.Module):
    """
    Multimodal VAE v2 con Product-of-Experts.
    
    Miglioramenti rispetto a v1:
    - ATAC encoder usa LSI (50-dim dense) invece di 15k TF-IDF sparse
    - Residual blocks per gradient flow stabile
    - Per-cell library size per Negative Binomial (non più scalare globale)
    - Latent dim 20 (più appropriato per ~11k cellule e ~11 tipi cellulari)
    """
    
    def __init__(self, n_genes, n_peaks, n_lsi, n_latent=20,
                 rna_enc_dim=512, atac_enc_dim=256, dec_dim=512,
                 n_enc_blocks=3, n_dec_blocks=2, dropout=0.1):
        super().__init__()
        self.n_latent = n_latent
        
        # Encoders (RNA: 3000→512, ATAC: 50 LSI→256)
        self.enc_rna = ResidualEncoder(n_genes, n_latent, rna_enc_dim, n_enc_blocks, dropout)
        self.enc_atac = ResidualEncoder(n_lsi, n_latent, atac_enc_dim, n_enc_blocks, dropout)
        
        # Decoders (z→512→output)
        self.dec_rna = ResidualDecoder(n_latent, n_genes, dec_dim, n_dec_blocks,
                                       output_activation=nn.Softmax(dim=-1))
        self.dec_atac = ResidualDecoder(n_latent, n_peaks, dec_dim, n_dec_blocks)
        
        # NB dispersion (per-gene, learned)
        self.log_theta = nn.Parameter(torch.randn(n_genes) * 0.1)
    
    @staticmethod
    def product_of_experts(mu_list, logvar_list):
        """PoE: combina posterior Gaussiane indipendenti con il prior."""
        mu_prior = torch.zeros_like(mu_list[0])
        logvar_prior = torch.zeros_like(logvar_list[0])
        
        precision = torch.exp(-logvar_prior)
        weighted_mu = mu_prior * precision
        
        for mu, logvar in zip(mu_list, logvar_list):
            prec_i = torch.exp(-logvar)
            precision = precision + prec_i
            weighted_mu = weighted_mu + mu * prec_i
        
        logvar_joint = -torch.log(precision)
        mu_joint = weighted_mu / precision
        return mu_joint, logvar_joint
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    
    def forward(self, rna_norm, atac_lsi):
        mu_rna, logvar_rna = self.enc_rna(rna_norm)
        mu_atac, logvar_atac = self.enc_atac(atac_lsi)
        
        mu_joint, logvar_joint = self.product_of_experts(
            [mu_rna, mu_atac], [logvar_rna, logvar_atac]
        )
        z = self.reparameterize(mu_joint, logvar_joint)
        
        return {
            "z": z,
            "mu_joint": mu_joint, "logvar_joint": logvar_joint,
            "mu_rna": mu_rna, "logvar_rna": logvar_rna,
            "mu_atac": mu_atac, "logvar_atac": logvar_atac,
            "rna_rate": self.dec_rna(z),
            "atac_logits": self.dec_atac(z),
        }
    
    def get_latent(self, rna_norm, atac_lsi):
        with torch.no_grad():
            mu_rna, logvar_rna = self.enc_rna(rna_norm)
            mu_atac, logvar_atac = self.enc_atac(atac_lsi)
            mu_joint, _ = self.product_of_experts(
                [mu_rna, mu_atac], [logvar_rna, logvar_atac]
            )
        return mu_joint

print("MVAE v2 architecture defined")

# %%
def nb_loss(x_raw, rate, log_theta, log_library):
    """Negative Binomial reconstruction loss con per-cell library size."""
    library = torch.exp(log_library)  # (batch, 1) — per-cell total counts
    mu = library * rate + 1e-8
    theta = torch.exp(log_theta).clamp(min=1e-4, max=1e6)
    
    log_nb = (
        torch.lgamma(x_raw + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x_raw + 1)
        + theta * torch.log(theta / (theta + mu))
        + x_raw * torch.log(mu / (theta + mu))
    )
    return -log_nb.sum(dim=-1).mean()


def bernoulli_loss(x_binary, logits):
    """Binary cross-entropy per ATAC peaks (open/closed)."""
    return F.binary_cross_entropy_with_logits(logits, x_binary, reduction="mean") * x_binary.shape[1]


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0,1))."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def mvae_loss(outputs, rna_raw, atac_binary, log_theta, log_library,
              kl_weight=1.0, atac_weight=0.2):
    """
    ELBO loss totale per il MVAE v2.
    
    Miglioramenti vs v1:
    - Per-cell library size per NB loss (non più scalare globale)
    - atac_weight=0.2 bilancia ATAC (15k BCE) vs RNA (3k NB)
    """
    loss_rna = nb_loss(rna_raw, outputs["rna_rate"], log_theta, log_library)
    loss_atac = bernoulli_loss(atac_binary, outputs["atac_logits"])
    loss_kl = kl_divergence(outputs["mu_joint"], outputs["logvar_joint"])
    
    total = loss_rna + atac_weight * loss_atac + kl_weight * loss_kl
    
    return {
        "total": total,
        "rna_recon": loss_rna.item(),
        "atac_recon": loss_atac.item(),
        "kl": loss_kl.item(),
    }

print("Loss functions defined (v2 — per-cell library size)")

# %% [markdown]
# ## Training
# **Cyclical KL annealing** (Fu et al., 2019) per prevenire il posterior collapse:
# 4 cicli con ratio 0.5 — in ogni ciclo β sale linearmente da 0→1, poi rimane a 1.
# Questo permette al modello di periodicamente focalizzarsi sulla ricostruzione prima di
# ri-imporre la regolarizzazione latente. Monitoraggio separato delle loss per modalità.

# %%
N_LATENT = 20
LR = 2e-3
EPOCHS = 150
N_KL_CYCLES = 4
KL_RATIO = 0.5

model = MVAE(
    n_genes=dataset.n_genes,
    n_peaks=dataset.n_peaks,
    n_lsi=dataset.n_lsi,
    n_latent=N_LATENT,
    rna_enc_dim=512,
    atac_enc_dim=256,
    dec_dim=512,
    n_enc_blocks=3,
    n_dec_blocks=2,
    dropout=0.1,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
print(model)

def cyclical_kl_weight(epoch, n_cycles=N_KL_CYCLES, n_epochs=EPOCHS, ratio=KL_RATIO):
    """Cyclical KL annealing (Fu et al., 2019)."""
    period = n_epochs / n_cycles
    progress = (epoch % period) / period
    if progress < ratio:
        return progress / ratio
    return 1.0

history = {"train_total": [], "train_rna": [], "train_atac": [], "train_kl": [],
           "val_total": [], "val_rna": [], "val_atac": [], "val_kl": []}

best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 25

for epoch in range(EPOCHS):
    kl_weight = cyclical_kl_weight(epoch)
    
    model.train()
    epoch_losses = {"total": 0, "rna": 0, "atac": 0, "kl": 0}
    n_batches = 0
    
    for batch in train_loader:
        rna_norm = batch["rna_norm"].to(device)
        rna_raw = batch["rna_raw"].to(device)
        atac_lsi = batch["atac_lsi"].to(device)
        atac_binary = batch["atac_binary"].to(device)
        log_library = batch["log_library"].to(device)
        
        outputs = model(rna_norm, atac_lsi)
        losses = mvae_loss(outputs, rna_raw, atac_binary,
                          model.log_theta, log_library,
                          kl_weight=kl_weight, atac_weight=0.2)
        
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        epoch_losses["total"] += losses["total"].item()
        epoch_losses["rna"] += losses["rna_recon"]
        epoch_losses["atac"] += losses["atac_recon"]
        epoch_losses["kl"] += losses["kl"]
        n_batches += 1
    
    for k in epoch_losses:
        epoch_losses[k] /= n_batches
    history["train_total"].append(epoch_losses["total"])
    history["train_rna"].append(epoch_losses["rna"])
    history["train_atac"].append(epoch_losses["atac"])
    history["train_kl"].append(epoch_losses["kl"])
    
    model.eval()
    val_losses = {"total": 0, "rna": 0, "atac": 0, "kl": 0}
    n_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            rna_norm = batch["rna_norm"].to(device)
            rna_raw = batch["rna_raw"].to(device)
            atac_lsi = batch["atac_lsi"].to(device)
            atac_binary = batch["atac_binary"].to(device)
            log_library = batch["log_library"].to(device)
            
            outputs = model(rna_norm, atac_lsi)
            losses = mvae_loss(outputs, rna_raw, atac_binary,
                              model.log_theta, log_library,
                              kl_weight=1.0, atac_weight=0.2)
            
            val_losses["total"] += losses["total"].item()
            val_losses["rna"] += losses["rna_recon"]
            val_losses["atac"] += losses["atac_recon"]
            val_losses["kl"] += losses["kl"]
            n_val_batches += 1
    
    for k in val_losses:
        val_losses[k] /= n_val_batches
    history["val_total"].append(val_losses["total"])
    history["val_rna"].append(val_losses["rna"])
    history["val_atac"].append(val_losses["atac"])
    history["val_kl"].append(val_losses["kl"])
    
    scheduler.step()
    
    if val_losses["total"] < best_val_loss:
        best_val_loss = val_losses["total"]
        patience_counter = 0
        torch.save(model.state_dict(), "best_mvae_v2.pt")
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {epoch_losses['total']:.1f} (RNA:{epoch_losses['rna']:.1f} "
              f"ATAC:{epoch_losses['atac']:.1f} KL:{epoch_losses['kl']:.2f}) | "
              f"Val: {val_losses['total']:.1f} | β={kl_weight:.3f} | "
              f"LR={scheduler.get_last_lr()[0]:.2e}")
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load("best_mvae_v2.pt", weights_only=True))
print(f"\nBest validation loss: {best_val_loss:.2f}")

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, key, title, color in zip(
    axes,
    ["total", "rna", "atac", "kl"],
    ["Total ELBO", "RNA Recon (NB)", "ATAC Recon (BCE)", "KL Divergence"],
    ["#1976D2", "#E53935", "#FB8C00", "#7B1FA2"],
):
    ax.plot(history[f"train_{key}"], label="Train", color=color, alpha=0.8)
    ax.plot(history[f"val_{key}"], label="Val", color=color, ls="--", alpha=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Latent Space Analysis
# Estraiamo le rappresentazioni latenti congiunte e confrontiamole con l'analisi unimodale.

# %%
model.eval()
all_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

latent_list = []
with torch.no_grad():
    for batch in tqdm(all_loader, desc="Encoding"):
        rna_norm = batch["rna_norm"].to(device)
        atac_lsi = batch["atac_lsi"].to(device)
        z = model.get_latent(rna_norm, atac_lsi)
        latent_list.append(z.cpu().numpy())

Z = np.concatenate(latent_list, axis=0)
print(f"Latent space shape: {Z.shape}")

reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=SEED)
Z_umap = reducer.fit_transform(Z)

shared = rna.obs_names.intersection(atac.obs_names)
cell_types_aligned = rna.obs.loc[shared, "cell_type"].values
leiden_rna_aligned = rna.obs.loc[shared, "leiden_rna"].values

# %%
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# UMAP RNA-only
ax = axes[0]
rna_umap = rna.obsm.get("X_umap_rna")
if rna_umap is not None:
    for ct in np.unique(cell_types_aligned):
        mask = rna.obs.loc[shared, "cell_type"].values == ct
        ax.scatter(rna_umap[rna.obs_names.isin(shared)][mask, 0],
                   rna_umap[rna.obs_names.isin(shared)][mask, 1],
                   s=2, alpha=0.5, label=ct)
ax.set_title("RNA-only UMAP", fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

# UMAP ATAC-only
ax = axes[1]
atac_umap = atac.obsm.get("X_umap_atac")
if atac_umap is not None:
    for ct in np.unique(cell_types_aligned):
        mask = atac.obs.loc[shared, "cell_type"].values == ct
        ax.scatter(atac_umap[atac.obs_names.isin(shared)][mask, 0],
                   atac_umap[atac.obs_names.isin(shared)][mask, 1],
                   s=2, alpha=0.5, label=ct)
ax.set_title("ATAC-only UMAP", fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

# UMAP MVAE joint
ax = axes[2]
for ct in np.unique(cell_types_aligned):
    mask = cell_types_aligned == ct
    ax.scatter(Z_umap[mask, 0], Z_umap[mask, 1], s=2, alpha=0.5, label=ct)
ax.set_title("MVAE Joint Latent UMAP", fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, markerscale=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.show()

# %%
from sklearn.cluster import KMeans
import leidenalg

import scanpy as sc

adata_latent = ad.AnnData(X=Z)
adata_latent.obs["cell_type"] = cell_types_aligned
adata_latent.obsm["X_umap"] = Z_umap

sc.pp.neighbors(adata_latent, use_rep="X", n_neighbors=30)
sc.tl.leiden(adata_latent, resolution=0.8, key_added="leiden_mvae")

ari = adjusted_rand_score(cell_types_aligned, adata_latent.obs["leiden_mvae"])
nmi = normalized_mutual_info_score(cell_types_aligned, adata_latent.obs["leiden_mvae"])
sil = silhouette_score(Z, cell_types_aligned, metric="euclidean", sample_size=5000)

print(f"{'Metric':<30} {'Score':>8}")
print("=" * 40)
print(f"{'Adjusted Rand Index (ARI)':<30} {ari:>8.4f}")
print(f"{'Normalized Mutual Info (NMI)':<30} {nmi:>8.4f}")
print(f"{'Silhouette Score':<30} {sil:>8.4f}")
print(f"{'N clusters (Leiden)':<30} {adata_latent.obs['leiden_mvae'].nunique():>8d}")
print(f"{'N cell types (ground truth)':<30} {len(np.unique(cell_types_aligned)):>8d}")

fig, ax = plt.subplots(figsize=(12, 8))
ct_vs_cluster = pd.crosstab(
    cell_types_aligned,
    adata_latent.obs["leiden_mvae"].values,
    normalize="index"
)
sns.heatmap(ct_vs_cluster, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax, 
            linewidths=0.5, cbar_kws={"label": "Proportion"})
ax.set_title("Cell Type vs MVAE Leiden Cluster (row-normalized)", fontweight="bold")
ax.set_ylabel("Cell Type")
ax.set_xlabel("MVAE Leiden Cluster")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cross-modal prediction
# Verifichiamo la capacità del modello di predire una modalità dall'altra attraverso lo spazio latente condiviso.

# %%
model.eval()

rna_to_atac_auroc = []
atac_to_rna_corr = []

from sklearn.metrics import roc_auc_score

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Cross-modal eval"):
        rna_norm = batch["rna_norm"].to(device)
        atac_lsi = batch["atac_lsi"].to(device)
        rna_raw = batch["rna_raw"].to(device)
        atac_binary = batch["atac_binary"].to(device)
        
        # RNA → ATAC: encode RNA only, decode ATAC peaks
        mu_rna, logvar_rna = model.enc_rna(rna_norm)
        z_rna = mu_rna
        atac_pred = torch.sigmoid(model.dec_atac(z_rna))
        
        # ATAC → RNA: encode ATAC (LSI) only, decode RNA
        mu_atac, logvar_atac = model.enc_atac(atac_lsi)
        z_atac = mu_atac
        rna_pred = model.dec_rna(z_atac)
        
        # RNA → ATAC AUROC
        atac_true = atac_binary.cpu().numpy().flatten()
        atac_scores = atac_pred.cpu().numpy().flatten()
        idx = np.random.choice(len(atac_true), min(500000, len(atac_true)), replace=False)
        try:
            auroc = roc_auc_score(atac_true[idx], atac_scores[idx])
            rna_to_atac_auroc.append(auroc)
        except ValueError:
            pass
        
        # ATAC → RNA Pearson correlation
        rna_true = rna_raw.cpu().numpy()
        rna_p = rna_pred.cpu().numpy()
        for i in range(rna_true.shape[0]):
            r = np.corrcoef(rna_true[i], rna_p[i])[0, 1]
            if not np.isnan(r):
                atac_to_rna_corr.append(r)

print(f"\n{'Cross-Modal Prediction Results':=^50}")
print(f"RNA → ATAC (AUROC): {np.mean(rna_to_atac_auroc):.4f} ± {np.std(rna_to_atac_auroc):.4f}")
print(f"ATAC → RNA (Pearson corr): {np.mean(atac_to_rna_corr):.4f} ± {np.std(atac_to_rna_corr):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(rna_to_atac_auroc, bins=20, color="#E53935", edgecolor="white", alpha=0.8)
axes[0].axvline(np.mean(rna_to_atac_auroc), color="black", ls="--", lw=2)
axes[0].set_title("RNA → ATAC (per-batch AUROC)", fontweight="bold")
axes[0].set_xlabel("AUROC")

axes[1].hist(atac_to_rna_corr, bins=50, color="#1976D2", edgecolor="white", alpha=0.8)
axes[1].axvline(np.mean(atac_to_rna_corr), color="black", ls="--", lw=2)
axes[1].set_title("ATAC → RNA (per-cell Pearson r)", fontweight="bold")
axes[1].set_xlabel("Correlation")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cell type classifier
# Residual MLP che classifica i tipi cellulari dallo spazio latente MVAE v2.
# 
# - **Gaussian noise augmentation** sugli embedding latenti per regolarizzazione
# - **Loss pesata per classe** per gestire lo sbilanciamento (Plasmablast, pDC, DC rari)
# - **CosineAnnealingWarmRestarts** per esplorare la loss landscape più efficacemente

# %%
class CellTypeClassifier(nn.Module):
    """Residual MLP classifier per cell type prediction dal latent space MVAE v2."""
    
    def __init__(self, n_latent=20, n_classes=12, hidden_dim=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, z):
        h = self.input_proj(z)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


le = LabelEncoder()
labels_encoded = le.fit_transform(cell_types_aligned)
n_classes = len(le.classes_)

Z_train, Z_test, y_train, y_test = train_test_split(
    Z, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=SEED
)

# Class weights per gestire lo sbilanciamento
from collections import Counter
counts = Counter(y_train)
n_total = len(y_train)
class_weights = torch.tensor(
    [n_total / (n_classes * counts[i]) for i in range(n_classes)],
    dtype=torch.float32
).to(device)
print(f"Class weights: {dict(zip(le.classes_, class_weights.cpu().numpy().round(2)))}")

class LatentClassDataset(Dataset):
    """Dataset con optional Gaussian noise augmentation sugli embedding."""
    def __init__(self, Z, y, augment=False, noise_std=0.15):
        self.Z = torch.tensor(Z, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.noise_std = noise_std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        z = self.Z[idx]
        if self.augment:
            z = z + torch.randn_like(z) * self.noise_std
        return z, self.y[idx]

clf_train = DataLoader(LatentClassDataset(Z_train, y_train, augment=True, noise_std=0.15),
                       batch_size=128, shuffle=True)
clf_test = DataLoader(LatentClassDataset(Z_test, y_test, augment=False),
                      batch_size=256, shuffle=False)

print(f"Classes ({n_classes}): {le.classes_}")
print(f"Train: {len(Z_train)} | Test: {len(Z_test)}")

# %%
CLF_EPOCHS = 120

clf_model = CellTypeClassifier(
    n_latent=N_LATENT, n_classes=n_classes,
    hidden_dim=256, n_blocks=3, dropout=0.2
).to(device)

clf_optimizer = torch.optim.AdamW(clf_model.parameters(), lr=5e-4, weight_decay=1e-3)
clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    clf_optimizer, T_0=30, T_mult=2, eta_min=1e-6
)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

clf_params = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)
print(f"Classifier parameters: {clf_params:,}")

clf_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_clf_acc = 0.0

for epoch in range(CLF_EPOCHS):
    clf_model.train()
    correct, total, running_loss = 0, 0, 0
    for z_batch, y_batch in clf_train:
        z_batch, y_batch = z_batch.to(device), y_batch.to(device)
        logits = clf_model(z_batch)
        loss = criterion(logits, y_batch)
        
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()
        
        running_loss += loss.item() * z_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += z_batch.size(0)
    
    clf_history["train_loss"].append(running_loss / total)
    clf_history["train_acc"].append(correct / total)
    
    clf_model.eval()
    correct, total, running_loss = 0, 0, 0
    with torch.no_grad():
        for z_batch, y_batch in clf_test:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            logits = clf_model(z_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item() * z_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += z_batch.size(0)
    
    val_acc = correct / total
    clf_history["val_loss"].append(running_loss / total)
    clf_history["val_acc"].append(val_acc)
    clf_scheduler.step()
    
    if val_acc > best_clf_acc:
        best_clf_acc = val_acc
        torch.save(clf_model.state_dict(), "best_classifier_v2.pt")
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Acc: {clf_history['train_acc'][-1]:.4f} | "
              f"Val Acc: {val_acc:.4f} | Best: {best_clf_acc:.4f}")

clf_model.load_state_dict(torch.load("best_classifier_v2.pt", weights_only=True))
print(f"\nBest Test Accuracy: {best_clf_acc:.4f}")

# %%
clf_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for z_batch, y_batch in clf_test:
        z_batch = z_batch.to(device)
        preds = clf_model(z_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print(classification_report(all_labels, all_preds, target_names=le.classes_))

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
for ct in np.unique(cell_types_aligned):
    mask = cell_types_aligned == ct
    ax1.scatter(Z_umap[mask, 0], Z_umap[mask, 1], s=1, alpha=0.4, label=ct)
ax1.set_title("MVAE v2 Latent — Cell Types", fontweight="bold")
ax1.set_xticks([]); ax1.set_yticks([])

ax2 = fig.add_subplot(gs[0, 1])
for cl in sorted(adata_latent.obs["leiden_mvae"].unique(), key=int):
    mask = adata_latent.obs["leiden_mvae"].values == cl
    ax2.scatter(Z_umap[mask, 0], Z_umap[mask, 1], s=1, alpha=0.4, label=cl)
ax2.set_title("MVAE v2 Latent — Leiden Clusters", fontweight="bold")
ax2.set_xticks([]); ax2.set_yticks([])

ax3 = fig.add_subplot(gs[0, 2])
Z_test_umap = reducer.transform(Z_test)
for i, ct in enumerate(le.classes_):
    mask = np.array(all_preds) == i
    if mask.any():
        ax3.scatter(Z_test_umap[mask, 0], Z_test_umap[mask, 1], s=2, alpha=0.4, label=ct)
ax3.set_title("Classifier Predictions (test set)", fontweight="bold")
ax3.set_xticks([]); ax3.set_yticks([])

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(clf_history["train_acc"], label="Train", color="#1976D2")
ax4.plot(clf_history["val_acc"], label="Val", color="#E53935")
ax4.set_title("Classifier Accuracy", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Accuracy"); ax4.legend(); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(history["train_total"], label="Train", color="#1976D2")
ax5.plot(history["val_total"], label="Val", color="#E53935")
ax5.set_title("MVAE v2 Total Loss (ELBO)", fontweight="bold")
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Loss"); ax5.legend(); ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds, normalize="true")
sns.heatmap(cm, xticklabels=le.classes_, yticklabels=le.classes_,
            cmap="Blues", annot=True, fmt=".2f", ax=ax6, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax6.set_title("Confusion Matrix (normalized)", fontweight="bold")
ax6.set_xlabel("Predicted"); ax6.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, markerscale=6,
           fontsize=8, bbox_to_anchor=(0.5, -0.05))

plt.suptitle("Human Immune Multiomics Atlas — MVAE v2 Deep Learning Pipeline", 
             fontsize=14, fontweight="bold", y=1.02)
plt.savefig("results_summary_v2.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPipeline v2 completed successfully.")


