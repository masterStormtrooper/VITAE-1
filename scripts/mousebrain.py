import pandas as pd
import scanpy as sc
import numpy as np
import os as os
import sys
sys.path.append('/home/alanluo/VITAE-1')
import VITAE
from VITAE.utils import load_data
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
data = load_data(path = "/project/jingshuw/trajectory_analysis/data", file_name = "mouse_brain_merged")
labels = pd.DataFrame({'Grouping': data['grouping']}, index = data['cell_ids'])
labels['Grouping'] = labels['Grouping'].astype("category")
genes = pd.DataFrame({'gene_names': data['gene_names']}, index = data['gene_names'])
dd = sc.AnnData(X = data['count'], obs = labels, var = genes)
#dd.layers['counts'] = dd.X.copy()
dd.obs['Source'] = data['covariates'][:, 2]
dd.obs['S_Score'] = data['covariates'][:, 0]
dd.obs['G2M_Score'] = data['covariates'][:, 1]
dd.obs['Day'] = np.array([item[0] for item in np.char.split(data['cell_ids'], sep ='_')])
sc.pp.normalize_total(dd, target_sum=1e4)
sc.pp.log1p(dd)
sc.pp.highly_variable_genes(dd, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.scale(dd, max_value=10)
sc.tl.pca(dd, svd_solver='arpack')
sc.pp.neighbors(dd, n_neighbors=10, n_pcs=40)
sc.tl.leiden(dd, resolution = 0.4)
### Scanpy visualization ###
sc.tl.umap(dd)
sc.pl.umap(dd, color=['Grouping', 'Source', 'Day', 'S_Score', 'G2M_Score'], legend_loc='on data', legend_fontsize=5)


model = VITAE.VITAE()

model.initialize(adata = dd,
                 npc = 64, model_type = 'Gaussian',
                 hidden_layers = [32], latent_space_dim = 16,
                 covariates = ['Source', 'S_Score', 'G2M_Score'])
                 


model.pre_train(early_stopping_tolerance = 0.01, early_stopping_relative = True)
model.visualize_latent(color=['Grouping', 'Source', 'Day', 'S_Score', 'G2M_Score'], method = "UMAP")

model.init_latent_space(res = 0.6, ratio_prune = 0.5)
model.visualize_latent(color=['Grouping', 'Source', 'Day', 'S_Score', 'G2M_Score', 'vitae_init_clustering', 'Eomes'], method = "UMAP")

model.train( early_stopping_tolerance = 0.01, early_stopping_relative = True)
model.init_inference(batch_size=32, L=10)
model.visualize_latent(color=['Grouping', 'Source', 'Day', 'S_Score', 'G2M_Score', 'vitae_init_clustering', 'vitae_new_clustering', 'Eomes'], method = "UMAP")

G = model.comp_inference_score(method='modified_map',  # 'mean', 'modified_mean', 'map', and 'modified_map'
                               no_loop=True,            # if no_loop=True, then find the maximum spanning tree
                               cutoff = 0.1)
G.edges(data = True)

import matplotlib.pyplot as plt
plt.show()

model.infer_trajectory(init_node=0,  # initial node for computing pseudotime.
                       cutoff=0.1                 # (Optional) cutoff score for edges (the default is 0.01).
                       )  
model.visualize_latent(color = ['vitae_new_clustering', 'Grouping', 'leiden','pseudotime', 
                                'projection_uncertainty'], method = "UMAP")
                                
model.visualize_latent(color = ['vitae_new_clustering', 'clusters', 'pseudotime',
                                'projection_uncertainty'], method = "UMAP")