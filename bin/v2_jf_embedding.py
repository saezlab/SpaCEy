import os
import warnings
import numpy as np
import scanpy as sc
import custom_tools
import anndata as ad
import seaborn as sns
import decoupler as dc
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
from dataset import TissueDataset
from pathlib import Path
import matplotlib


warnings.simplefilter(action='ignore')
sc.settings.verbosity = 0
# Set figure params
sc.set_figure_params(scanpy=True, facecolor="white", dpi=80, dpi_save=300)

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

def compare_proportions(population1, population2, cell_type='T-cell', use_fisher=False):
    """
    Compares the proportions of a specific cell type (e.g., T-cell) in two populations.

    Parameters:
    - population1: list of cell types in the first population
    - population2: list of cell types in the second population
    - cell_type: the cell type to compare (default is 'T-cell')
    - use_fisher: bool, whether to use Fisher's exact test instead of Chi-square (for smaller sample sizes)
    # teset deneme 

    Returns:
    - p-value: The p-value indicating statistical significance
    - stat_test: The type of test used ('Chi-square' or 'Fisher Exact')
    """
    # Count occurrences of the cell type in both populations
    t_cell_count1 = population1.count(cell_type)
    t_cell_count2 = population2.count(cell_type)
    print(t_cell_count1, t_cell_count2)
    
    # Get the total population sizes
    total1 = len(population1)
    total2 = len(population2)
    
    # Create a 2x2 contingency table for the proportions of T-cells vs non-T-cells
    contingency_table = [
        [t_cell_count1, total1 - t_cell_count1],  # Population 1 (T-cell vs non-T-cell)
        [t_cell_count2, total2 - t_cell_count2]   # Population 2 (T-cell vs non-T-cell)
    ]
    print(contingency_table)
    if use_fisher:
        # Use Fisher's exact test (works for small sample sizes)
        _, p_value = fisher_exact(contingency_table)
        stat_test = "Fisher Exact"
    else:
        # Use Chi-square test for larger samples
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        stat_test = "Chi-square"

    return p_value, stat_test

# Read json file
# exp_name = "GATV2_NegativeLogLikelihood_month_04-12-2023"
# job_id = "fombnNMthdocYhDPmAjaBQ"
exp_name = "JacksonFischer"
dataset_name = "JacksonFischer"
# job_id = "2mYvWJwUarIJkw0vvUbl3Q"
job_id = "JF"
PLT_PATH = f"../plots/analysis/{exp_name}_{job_id}"
Path(PLT_PATH).mkdir(parents=True, exist_ok=True)
device =  custom_tools.get_device()
args  = custom_tools.load_json(f"../models/{exp_name}/{job_id}.json")



# args["num_node_features"] = 33
deg = None
if "PNA" in exp_name:
    deg = custom_tools.load_pickle(f"../models/{exp_name}/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = f"../models/{exp_name}", model_type = "SD", args = args, deg=deg, device=device)
dataset = TissueDataset(os.path.join(f"../data/{dataset_name}", "month"),  "month")

# Read the explanations adata
adata_exp = sc.read_h5ad(f"../data/out_data/adatafiles/{dataset_name}/{exp_name}_{job_id}_concatenated_explanations.h5ad")
adata_exp.obs_names_make_unique()

# for mod in model.modules():
#    print(mod)



imp_threshold = 0.75
# Get the importance of the nodes
node_importance = np.array(adata_exp.obs["importance"])
node_imp_thr = np.quantile(node_importance, imp_threshold)

importances_hard_v2 = np.array(node_importance > node_imp_thr, dtype="str")
# print("importances_hard", importances_hard)
importances_hard_v2 = pd.Series(importances_hard_v2, dtype="category")
# print(importances_hard)
adata_exp.obs["importance_hard"] = importances_hard_v2.values



sns.displot(adata_exp.obs[adata_exp.obs["importance_hard"]=="True"], x="importance", bins=300)
adata_exp.obs[adata_exp.obs["importance_hard"]=="True"]


print(adata_exp.obs[adata_exp.obs["importance_hard"]=="True"].groupby("class").agg("count")["img_id"] / adata_exp.obs[adata_exp.obs["importance_hard"]=="True"].groupby("class").agg("count")["img_id"].sum())

print(adata_exp.obs.groupby("class").agg("count")["img_id"] / adata_exp.obs.groupby("class").agg("count")["img_id"].sum())



import embeddings
emd, related_data = embeddings.get_intermediate_embeddings_for_dataset(model, dataset, batch_size=1)
emd_cnv, related_data_cnv = embeddings.get_intermediate_embeddings_for_dataset(model, dataset, mode="CNV", batch_size=1)


embedding_arr = np.array(emd[0])
pid_list, img_id_list, osmonth_lst,  clinical_type_lst, tumor_grade_lst, censor_lst= [], [], [], [], [], []
for data in related_data:
    pid_list.append(str(data.p_id[0]))
    osmonth_lst.append(data.osmonth.item())
    img_id_list.append(data.img_id[0])
    clinical_type_lst.append(data.clinical_type[0])
    tumor_grade_lst.append(str(data.tumor_grade.item()))
    censor_lst.append(data.is_censored[0].item())
embedding_arr.shape



adata_emb = ad.AnnData(embedding_arr)
adata_emb.var_names = [f"emb_{i}" for i in range(embedding_arr.shape[1])]
adata_emb.obs_names = img_id_list
adata_emb.obs["img_id"] = img_id_list
adata_emb.obs["img_id"] = adata_emb.obs["img_id"].astype("category")
adata_emb.obs["osmonth"] = osmonth_lst
adata_emb.obs["p_id"] = pid_list
adata_emb.obs["clinical_type"] = clinical_type_lst
adata_emb.obs["tumor_grade"] = tumor_grade_lst
adata_emb.obs["is_censored"] = censor_lst


sc.tl.pca(adata_emb, svd_solver='arpack', random_state=0)
sc.pp.neighbors(adata_emb)
sc.tl.leiden(adata_emb, key_added = "leiden", resolution=0.1)
sc.tl.umap(adata_emb)
# sc.pl.umap(adata, color=["osmonth", "leiden"])
upper_quartile = int(np.quantile(adata_emb.obs["osmonth"], 0.75))
lower_quartile = int(np.quantile(adata_emb.obs["osmonth"], 0.25))

adata_emb.obs["high_surv"]="0"
adata_emb.obs["low_surv"]="0"

adata_emb.obs.loc[adata_emb.obs["osmonth"]>upper_quartile, "high_surv" ] = "1"
adata_emb.obs.loc[adata_emb.obs["osmonth"]<lower_quartile, "low_surv" ] = "1"


cmap = sns.palettes.get_colormap("tab20")
color_dict = dict()

for ind, clust_index in enumerate(adata_emb.obs["leiden"].cat.categories):
    color_dict[clust_index] = cmap.colors[ind]

rcParams['figure.figsize']=(10,8)
sc.pl.umap(adata_emb, color=["osmonth", "leiden"], palette=color_dict, show=True, legend_loc = 'on data') #, save="_jacksonfisher_embedding_boxplot.pdf")
plt.tight_layout()
# plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_embedding_umap.pdf"), dpi=300)


fig = plt.figure(figsize=(6, 5))
sns.boxplot(data=adata_emb.obs, x="leiden", y="osmonth", palette=color_dict)
plt.tight_layout()
# plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_embedding_boxplot.pdf"), dpi=300)
# sc.pl.umap(adata, color=["leiden"], palette=color_dict, legend_loc = 'on data')
# sc.pl.umap(adata, color=["high_surv", "low_surv"], palette=["grey", "black"])
# sc.pl.umap(adata, color=["is_censored"], palette=["grey", "black"])

# sc.pl.dotplot(adata, n_genes=5, groupby='bulk_labels', dendrogram=True)

# sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, key=f"t-test", show=True, groupby=f"leiden")



pdata_imp_vs_unimp = dc.get_pseudobulk(
    adata_exp,
    sample_col='img_id',
    groups_col='importance_hard',
    mode='mean',
    min_cells=0,
    min_counts=0
)

# create a new pseudobulk anndata with only important nodes
pdata_imp = pdata_imp_vs_unimp[pdata_imp_vs_unimp.obs_names.str.contains("True"),:].copy()
new_obs_names = [obs_n.split("_")[0] for obs_n in pdata_imp.obs_names]
pdata_imp.obs_names = new_obs_names

pdata_imp.obs["leiden"] = adata_emb.obs["leiden"]
pdata_imp.obsm["X_umap"] = adata_emb.obsm["X_umap"]

"""sc.tl.rank_genes_groups(pdata_imp, groupby=f"leiden", method='wilcoxon', groups=["0", "2"], key_added = f"wilcoxon")

rcParams['figure.figsize']=(4,4)
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.labelsize' : 10}) 

sc.pl.rank_genes_groups(pdata_imp, n_genes=5, sharey=False,  key=f"wilcoxon", show=False, groupby="leiden")
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_psedu_imp_ranking.pdf"), dpi=300)

rcParams['figure.figsize']=(4,10)
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.labelsize' : 10}) 
sc.pl.rank_genes_groups_dotplot(pdata_imp, n_genes=5, standard_scale='var', key=f"wilcoxon", figsize = (8,8), show=False, groupby="leiden")
plt.gcf().subplots_adjust(top = 0.60, bottom=0.50)
plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_psedu_imp_dotplot.pdf"), dpi=300)
"""




sc.tl.rank_genes_groups(pdata_imp, groupby=f"leiden", method='wilcoxon', key_added = f"wilcoxon_all")

rcParams['figure.figsize']=(4,4)
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.labelsize' : 10}) 

sc.pl.rank_genes_groups(pdata_imp, n_genes=5, sharey=False,  key=f"wilcoxon_all", show=True, groupby="leiden")
plt.tight_layout()
# plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_psedu_imp_ranking.pdf"), dpi=300)

rcParams['figure.figsize']=(4,3)
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.labelsize' : 10}) 
sc.pl.rank_genes_groups(pdata_imp, n_genes=5, standard_scale='var', key=f"wilcoxon_all", figsize = (8,8), show=False, groupby="leiden")
sc.pl.rank_genes_groups_dotplot(pdata_imp, n_genes=5, standard_scale='var', key=f"wilcoxon_all", figsize = (8,8), show=False, groupby="leiden")
plt.gcf().subplots_adjust(top = 0.60, bottom=0.50)
# plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_psedu_imp_dotplot_one_vs_all.pdf"), dpi=300)
# plt.savefig(os.path.join(PLT_PATH, f"{dataset_name}_psedu_imp_dotplot_one_vs_all.png"), dpi=300)



# TODO: Add leiden clusters to adata_exp

for cat in adata_emb.obs["leiden"].cat.categories:
    img_ids = adata_emb[adata_emb.obs["leiden"]==cat].obs["img_id"].cat.categories
    adata_exp.obs.loc[adata_exp.obs["img_id"].isin(img_ids), "leiden"] = cat

print(sorted(adata_emb[adata_emb.obs["leiden"]=="0"].obs["img_id"].cat.categories))


adata_exp.obs.groupby("importance_hard").agg("count")


def plot_cell_type_proportion(adata, group_col="leiden", obs_col = "cell_type", fl_path = None):
    group_list = list(adata.obs[group_col].cat.categories)
    c_type_list = list(adata.obs[obs_col].cat.categories)
    # print(len(c_type_list))
    #c_type_list.remove("Large elongated")
    #c_type_list.remove("Macrohage")
    if obs_col=="cell_type" and "Macrophage" in c_type_list:
        c_type_list.remove("Macrophage")
    print(len(c_type_list))
    c_type_list = c_type_list[:20]
    for cond in group_list:
        adata_tmp = adata[adata.obs[group_col]==cond,:]
        for c_type in c_type_list:
            if adata_tmp[adata_tmp.obs[obs_col]==c_type].shape[0]<8:
                print(c_type, adata_tmp[adata_tmp.obs[obs_col]==c_type].shape[0])
                c_type_list.remove(c_type)
            # print(c_type, adata_tmp[adata_tmp.obs[obs_col]==c_type].shape[0])
            # cond_arr[-1].append(100*(adata_tmp[adata_tmp.obs[obs_col]==c_type].shape[0]/adata_tmp.shape[0]))
    print(c_type_list)
    
    cond_prop = dict()
    cond_arr = []
    for cond in group_list:
        cond_arr.append([])
        # print(cond, cond_arr)
        cond_prop[cond] = []
        adata_tmp = adata[adata.obs[group_col]==cond,:]
        sum = 0
        for c_type in c_type_list:


            cond_arr[-1].append(100*(adata_tmp[adata_tmp.obs[obs_col]==c_type].shape[0]/(adata_tmp[adata_tmp.obs[obs_col].isin(c_type_list),:].shape[0])))

    data = np.array(cond_arr).T
    # print("data", data.shape)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # cmap = matplotlib.cm.get_cmap('tab20')
    # print(cmap.colors)
    
    X = np.arange(data.shape[1])
    
    for i in range(data.shape[0]):
        print(data[i], c_type_list[i])
        ax1.bar(X, data[i],bottom = np.sum(data[:i], 
                    axis =0), width= 0.85, color = cmap.colors[i], label=c_type_list[i]  )

    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(group_list) # , rotation=45)
    ax1.set_xlabel("Cluster", fontweight='bold')
    ax1.set_ylabel("Proportion (%)", fontweight='bold')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(bottom=0.45)
    fig.tight_layout()
    if fl_path:
        plt.savefig(fl_path)

    plt.show()




adata_exp.obs["leiden"] = adata_exp.obs["leiden"].astype("category")
# adata_exp[adata_exp.obs["importance_hard"]=="True",:]
# plot_cell_type_proportion(adata_exp[adata_exp.obs["importance_hard"]=="True",:], group_col="leiden", obs_col = "class")
# plot_cell_type_proportion(adata_exp, group_col="leiden", obs_col = "class")

# plot_cell_type_proportion(adata_exp, group_col="leiden", obs_col = "class", fl_path = f"{PLT_PATH}/major_classification_prop_all.pdf")
# plot_cell_type_proportion(adata_exp[adata_exp, group_col="leiden", obs_col = "cell_type", fl_path=f"{PLT_PATH}/cell_class_prop_imp.pdf")

plot_cell_type_proportion(adata_exp[adata_exp.obs["importance_hard"]=="True",:], group_col="leiden", obs_col = "class", fl_path = f"{PLT_PATH}/{dataset_name}_major_classification_prop_imp.pdf")
# plot_cell_type_proportion(adata_exp[adata_exp.obs["importance_hard"]=="False",:], group_col="leiden", obs_col = "class", fl_path = f"{PLT_PATH}/{dataset_name}_major_classification_prop_notimp.pdf")
# plot_cell_type_proportion(adata_exp[adata_exp.obs["importance_hard"]=="True",:], group_col="leiden", obs_col = "cell_type", fl_path=f"{PLT_PATH}/cell_class_prop_imp.pdf")
# plot_cell_type_proportion(adata_exp, group_col="leiden", obs_col = "cell_type")


# Tumor and cell type distribution
selection = (adata_exp.obs["importance_hard"]=="True") & (adata_exp.obs["class"]=="Tumor")
plot_cell_type_proportion(adata_exp[selection,:], group_col="leiden", obs_col = "cell_type", fl_path = f"{PLT_PATH}/{dataset_name}_major_classification_prop_imp_tumor.pdf")

