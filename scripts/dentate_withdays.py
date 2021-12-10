from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import sys
sys.path.append('/home/alanluo/VITAE-1')
import VITAE
from VITAE.utils import load_data
import tensorflow as tf
import random
import os
from matplotlib import pyplot as plt


def create_heatmap_matrix(pi):
    """Create heatmap matrix
    @pi: numpy array contains pi weights"""
    matrix = np.zeros((5, 5))
    matrix[np.triu_indices(5)] = pi
    mask = np.tril(np.ones_like(matrix), k=-1)
    return matrix, mask


def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)


def run(seed, data, name, fp):
    """
    @seed: int seed number
    @data: scanpy object"""
    reset_random_seeds(seed)
    tf.keras.backend.clear_session() 
    model = VITAE.VITAE()
    model.initialize(adata = data, covariates='days', model_type = 'Gaussian')
    model.pre_train() 
    model.pre_train(early_stopping_tolerance = 0.01, early_stopping_relative = True)
    model.init_latent_space(cluster_label= 'leiden', res = 0.4)
    model.train(early_stopping_tolerance = 0.01, early_stopping_relative = True)

    f, axes = plt.subplots(1, 5, figsize=(25,5))
    cbar_ax = f.add_axes([0, 0.2, .03, 0.7])
    idx = 0
    p = model.vae.pilayer

    for x in data.obs['days'].unique():
        tmp = tf.expand_dims(tf.constant([x], dtype=tf.float32), 0)
        pi_val = tf.nn.softmax(p(tmp)).numpy()[0]
        matrix, mask = create_heatmap_matrix(pi_val)
        sns.heatmap(matrix, vmin=0, vmax=1, cmap="YlGnBu", mask=mask, ax=axes[idx], cbar_ax=cbar_ax)
        axes[idx].set_title(f'dentate_run_{name}_seed_{seed}_day_{x}')
        idx += 1
    model.init_inference(batch_size=32, L=100)
    sc.tl.umap(model._adata_z)
    model._adata.obsp = model._adata_z.obsp
    model._adata.obsm = model._adata_z.obsm
    sc.pl.umap(model._adata, color='vitae_init_clustering', show=False, ax=axes[-1])
    f.savefig(fp + f'dentate_run_{name}_seed_{seed}.png')

if __name__ == "__main__":
    # load data
    fp = '/home/alanluo/vitae_output/'
    data = load_data("/home/alanluo/VITAE-1/data/", file_name="dentate_withdays")
    labels = pd.DataFrame({'Grouping': data['grouping']}, index = data['cell_ids'])
    labels['Grouping'] = labels['Grouping'].astype("category")
    genes = pd.DataFrame({'gene_names': data['gene_names']}, index = data['gene_names'])
    dd = sc.AnnData(X = data['count'].copy(), obs = labels, var = genes)
    dd.layers['counts'] = dd.X.copy()
    dd.obs['days'] = data['covariates']
    sc.pp.normalize_total(dd, target_sum=1e4)
    sc.pp.log1p(dd)
    sc.pp.highly_variable_genes(dd, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.scale(dd, max_value=10)
    sc.tl.pca(dd, svd_solver='arpack')
    sc.pp.neighbors(dd, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(dd, resolution = 0.4)
    sc.tl.umap(dd)
    for seed in range(20):
        run(seed, dd, seed, fp)

