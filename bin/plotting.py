import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import torch
from torch_geometric import utils
import networkx as nx
from dataset import TissueDataset
from scipy.spatial import Voronoi, voronoi_plot_2d
from evaluation_metrics import r_squared_score, mse, rmse, mae
import matplotlib as mpl
import torch
# import umap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# TODO: Put this into the config file
seed = 42


S_PATH = os.path.dirname(__file__)
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "JacksonFischer")
PREPROSSED_DATA_PATH = os.path.join(S_PATH, "../data", "JacksonFischer", "raw")
PLOT_PATH = os.path.join(S_PATH, "../plots")

def plot_cell_count_distribution(comp_dataset_path):


    df_dataset = pd.read_csv(comp_dataset_path)
    number_of_cells = df_dataset.groupby(by=["ImageNumber"]).size().reset_index(name='Cell Counts')
    
    low_quartile = np.quantile(number_of_cells["Cell Counts"], 0.25)
    
    cell_count_dist_plot = sns.boxplot(data=number_of_cells, x='Cell Counts')
    fig = cell_count_dist_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/cell_count_distribution.pdf")

# plot_cell_count_distribution()

def box_plot_save(df, axis_name, path):
    osmonth_dist_plot = sns.boxplot(data=df, x=axis_name)
    fig = osmonth_dist_plot.get_figure()
    fig.savefig(path)
    plt.clf()


def plot_nom_distribution():
    dataset = TissueDataset("../data")
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    osmonhth_list = []
    for i in range(len(dataset)):
        osmonhth_list.append(dataset.__getitem__(i).y[0].item())
    
        #osmonhth_list.append(dataset.__getitem__(i))
        # print(i)
    df_osmonth = pd.DataFrame(osmonhth_list)
    
    box_plot_save(df_osmonth, 0, f"{PLOT_PATH}/osmonth_all_distribution.png")
    box_plot_save(df_osmonth[:180], 0, f"{PLOT_PATH}/osmonth_all_distribution.png")
    box_plot_save(df_osmonth[180:200], 0, f"{PLOT_PATH}/osmonth_all_distribution.png")
    box_plot_save(df_osmonth[200:], 0, f"{PLOT_PATH}/osmonth_all_distribution.png")

# plot_nom_distribution()

"""def plot_pred_vs_real(df, x_axis, y_axis, color, fl_name):
    # print(out_list, pred_list)
    sns_plot = sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color)
    plt.ylim(0, None)
    plt.xlim(0, None)
    fig = sns_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/{fl_name}.png")
    plt.clf()"""


def plot_pred_vs_real_lst(df_lst, x_axis_lst, y_axis_lst, color, order, fl_name):
    # print(out_list, pred_list)
    n_cols = len(df_lst)
    fig, axs = plt.subplots(ncols=n_cols, figsize=(18,5))

    for ind in range(n_cols):

        sns_plot = sns.scatterplot(data=df_lst[ind], x=x_axis_lst[ind], y=y_axis_lst[ind], hue=color, hue_order=order, ax = axs[ind])
        sns_plot.set_xlim(0, 6)
        sns_plot.set_ylim(0, 6)
        plt.legend(loc='upper left')    
        
        #plt.ylim(0, None)
        #plt.xlim(0, None)
    
    fig = sns_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/{fl_name}.png")
    plt.clf()

import matplotlib.lines as mlines

def plot_pred_vs_real(df, exp_name, fl_name, full_training=False):

    fig, axs = plt.subplots(1,3,figsize=(20,5))
    colors = {'TripleNeg':'red', 'HR+HER2-':'green', 'HR-HER2+':'blue', 'HR+HER2+':'orange'}
    labels = ['TripleNeg', 'HR+HER2-', 'HR-HER2+', 'HR+HER2+']
    colors = ['red', 'green', 'blue', 'orange']
    
    tvt_labels = ["train", "validation", "test"]
    if full_training:
        tvt_labels = ["train"]
    for idx, val in enumerate(tvt_labels):
        
        df_tvt = df.loc[(df['Fold#-Set'].str[2:] == val)]
        
        for idxlbl, lbl in enumerate(labels):
            # df_temp = df.loc[(df['Train Val Test'] == val) & (df['Clinical Type'] == lbl)]
            df_temp = df_tvt.loc[(df_tvt['Clinical Type'] == lbl)]
            # print(lbl, val)
            # print(df_temp)
            axs[idx].scatter(x=df_temp['True Value'], y=df_temp['Predicted'], color= colors[idxlbl], label=lbl)
            axs[idx].set_xlim(0, 300)
            axs[idx].set_ylim(0, 300)
            axs[idx].set_xlabel('Overall Survivability')
            axs[idx].set_ylabel('Predicted')
            
        r2_score = r_squared_score(df_tvt['True Value'], df_tvt['Predicted'])
        mser = mse(df_tvt['True Value'], df_tvt['Predicted'])
        maer = mae(df_tvt['True Value'], df_tvt['Predicted'])
        axs[idx].set_title(f"MSE: {mser:.3f} - R2 Score: {r2_score:.3f} - MAE: {maer:.3f}")
        axs[idx].legend(loc='lower right')

    plt.savefig(os.path.join(PLOT_PATH, exp_name, f"{fl_name}.png"))


def plot_graph(test_graph, coordinates_arr,  ax, node_size=3000, font_size=20, width=8):
    """Plots the original graph"""
    original_graph = utils.to_networkx(test_graph)
    nx.draw_networkx_nodes(original_graph,  pos=coordinates_arr, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(original_graph, pos=coordinates_arr, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(original_graph,  pos=coordinates_arr, arrows=False, width=width, ax=ax)
    return ax



def plot_only_explained_edges(test_graph, coordinates_arr, exp_edges_idx, ax, node_size=3000, font_size=20, width=8):
    
    original_graph = utils.to_networkx(test_graph)
    
    explained_graph = nx.Graph()
    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    for ind, val in enumerate(exp_edges_idx):
        if val.item():
            explained_edges.append((test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()))

    explained_graph.add_edges_from(list(explained_edges), weight=100)
    nx.draw_networkx_nodes(explained_graph,  pos=coordinates_arr, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(original_graph, pos=coordinates_arr, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(explained_graph,  pos=coordinates_arr, width=width, ax=ax)

    return ax



def plot_connected_components(test_graph, coordinates_arr, exp_edges_idx, ax, cc_threshold = 5, node_size=3000, font_size=20, width=8):
    
    original_graph = utils.to_networkx(test_graph)

    explained_graph = nx.Graph()
    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    for ind, val in enumerate(exp_edges_idx):
        if val.item():
            explained_edges.append((test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()))

    explained_graph.add_edges_from(list(explained_edges))

    for component in list(nx.connected_components(explained_graph)):
        if len(component)<=cc_threshold:
            for node in component:
                explained_graph.remove_node(node)
    nx.draw_networkx_nodes(explained_graph,  pos=coordinates_arr, node_size=node_size, ax=ax)
    # nx.draw_networkx_labels(original_graph, pos=coordinates_arr, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(explained_graph,  pos=coordinates_arr, width=width, ax=ax)
    
    return ax


def plot_node_importances(test_graph, coordinates_arr, node_score_dict, ax, quant_thr=None, node_size=3000, font_size=10, width=8):
    original_graph = utils.to_networkx(test_graph)

    color_list = []
    for n_id in original_graph.nodes:
        color_list.append(node_score_dict[n_id])

    if quant_thr:
        node_imp_thr = np.quantile(color_list, quant_thr)
        color_list = np.array(color_list > node_imp_thr).astype(int)

    node_score_str_dict = dict()
    for node_id in node_score_dict.keys():
        node_score_str_dict[node_id] = f"{node_id}_{node_score_dict[node_id]:.2f}"
        
    nx.draw(original_graph, pos=coordinates_arr, node_color=color_list, node_size=node_size, arrows=False, cmap=plt.cm.afmhot, ax=ax)
    # nx.draw_networkx_labels(original_graph, labels=node_score_str_dict, pos=coordinates_arr, font_size=font_size, ax=ax)

    nx.draw_networkx_edges(original_graph,  pos=coordinates_arr, arrows=False, width=width, ax=ax)

# TODO: Rename this function
def plot_node_importances_voronoi(test_graph, coordinates_arr, node_score_dict, ax, quant_thr=None, title=None, cmap=plt.cm.afmhot, font_size=10, width=8):
    original_graph = utils.to_networkx(test_graph)
    color_list = []
    for n_id in original_graph.nodes:
        color_list.append(node_score_dict[n_id])

    if quant_thr:
        node_imp_thr = np.quantile(color_list, quant_thr)
        color_list = np.array(color_list > node_imp_thr).astype(int)
    
    # find min/max values for normalization
    minima = min(color_list)
    maxima = max(color_list)

    minx, miny = list(np.min(coordinates_arr, axis=0))
    maxx, maxy = list(np.max(coordinates_arr, axis=0))
    
    # add dummy nodes to the corners 
    coordinates_arr = np.append(coordinates_arr, [[maxx+10, maxy+10], [minx-10,maxy+10], [maxx+10,miny-10], [minx-10,miny-10]], axis = 0)

    vor = Voronoi(coordinates_arr)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=5, ax=ax)
   
    # normalize chosen colormap
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap= cmap)

    #for r in range(len(vor.point_region)):
    #    region = vor.regions[vor.point_region[r]]
    #    # if not -1 in region:
    #    polygon = [vor.vertices[i] for i in region]
    #    ax.fill(*zip(*polygon), color=mapper.to_rgba(color_list[r]))

    for j in range(len(coordinates_arr)):
        region = vor.regions[vor.point_region[j]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(color_list[j]))
    
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if title:
        ax.set_title(title, fontsize=50)



def plot_node_types_voronoi(test_graph, coordinates_arr, ax, node_type="ct_class", cmap =plt.cm.Pastel1, ct_order= None, font_size=10):
    original_graph = utils.to_networkx(test_graph)
    color_list = []
    # ct_class
    # Get coarse cell type annotations (first dimension of ct_class)
    # print(test_graph[node_type])
    if node_type=="ct_class":
        coarse_cell_types = [ct[1] for ct in test_graph[node_type]]
    else:
        coarse_cell_types = [ct if pd.notna(ct) else "Tumor" for ct in test_graph[node_type]]

    if ct_order:
        unique_cell_types = ct_order
    else:
        unique_cell_types = list(set(coarse_cell_types))
    
    """# Assign a unique color to each cell type
    cell_type_colors = plt.cm.get_cmap("tab10", len(unique_cell_types))
    cell_type_color_dict = {ctype: cell_type_colors(i) for i, ctype in enumerate(unique_cell_types)}"""
    
    # Assign a unique pastel color to each cell type
    # pastel_colors = plt.cm.Pastel1  # Use a pastel colormap
    pastel_colors = cmap
    # Create a dictionary mapping cell types to colors from the custom colormap
    cell_type_color_dict = {
        ctype: cmap(i / max(1, len(unique_cell_types) - 1))  # Normalize index to [0,1]
        for i, ctype in enumerate(unique_cell_types)
    }

    # cell_type_color_dict = {ctype: pastel_colors(i % pastel_colors.N) for i, ctype in enumerate(unique_cell_types)}
    # print(cell_type_color_dict)
    # Map nodes to their respective cell type colors
    color_list = [cell_type_color_dict[coarse_cell_types[n_id]] for n_id in original_graph.nodes]
    
    minx, miny = list(np.min(coordinates_arr, axis=0))
    maxx, maxy = list(np.max(coordinates_arr, axis=0))
    
    # Add dummy nodes to the corners
    coordinates_arr = np.append(coordinates_arr, [[maxx+10, maxy+10], [minx-10,maxy+10], [maxx+10,miny-10], [minx-10,miny-10]], axis=0)
    
    vor = Voronoi(coordinates_arr)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=5, ax=ax)
    
    for j in range(len(coordinates_arr) - 4):  # Exclude dummy nodes
        region = vor.regions[vor.point_region[j]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=color_list[j])
    
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Create legend for cell types
    handles = [mpl.patches.Patch(color=cell_type_color_dict[ctype], label=ctype) for ctype in unique_cell_types]
    ax.legend(handles=handles, title="Cell Types", fontsize=font_size, loc='upper right')


def plot_node_importances(test_graph, coordinates_arr, node_score_dict, ax, quant_thr=None, node_size=3000, font_size=10, width=8):
    original_graph = utils.to_networkx(test_graph)

    color_list = []
    for n_id in original_graph.nodes:
        color_list.append(node_score_dict[n_id])

    if quant_thr:
        node_imp_thr = np.quantile(color_list, quant_thr)
        color_list = np.array(color_list > node_imp_thr).astype(int)

    node_score_str_dict = dict()
    for node_id in node_score_dict.keys():
        node_score_str_dict[node_id] = f"{node_id}_{node_score_dict[node_id]:.2f}"
        
    nx.draw(original_graph, pos=coordinates_arr, node_color=color_list, node_size=node_size, arrows=False, cmap=plt.cm.afmhot, ax=ax)
    # nx.draw_networkx_labels(original_graph, labels=node_score_str_dict, pos=coordinates_arr, font_size=font_size, ax=ax)

    nx.draw_networkx_edges(original_graph,  pos=coordinates_arr, arrows=False, width=width, ax=ax)

    
def plot_khop(test_graph, coordinates_arr, edgeid_to_mask_dict, n_of_hops, ax, node_size=3000, font_size=10, width=8):
    # The method returns (1) the nodes involved in the subgraph, (2) the filtered edge_index connectivity, (3) the mapping from node indices in node_idx to their new location, and (4) the edge mask indicating which edges were preserved.
    subset_nodes, subset_edge_index, mapping, edge_mask = utils.k_hop_subgraph(407, n_of_hops, test_graph.edge_index)
    # for ind in range(subset_edge_index.shape[1]):
    #     print(subset_edge_index[0,ind],subset_edge_index[1,ind], f"{edgeid_to_mask_dict[(subset_edge_index[0,ind].item(),subset_edge_index[1,ind].item())]:.2f}" )
    original_graph = utils.to_networkx(test_graph)
    # original_edges = list(original_graph.edges)
    # print("khop subset_edge_index:", subset_edge_index)
    explained_graph = nx.Graph()
    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    
    edge_label_dict = dict()
    
    total_score=0.0
    for ind, val in enumerate(edge_mask):
        if val.item():
            # print(original_edges[ind])
            n1, n2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
            explained_edges.append((n1,n2))
            edge_label_dict[(n1,n2)] = f"{edgeid_to_mask_dict[(n1,n2)]:.2f}"
            # edge_lbl_ids.append((n1,n2)) 
            # edge_label.append(f"{edgeid_to_mask_dict[(n1,n2)]:.2f}")
            total_score += edgeid_to_mask_dict[(n1,n2)]
            """"print((n1,n2), edgeid_to_mask_dict[(n1,n2)])
            if abs(edgeid_to_mask_dict[(n1,n2)]-edgeid_to_mask_dict[(n2,n1)])>0.5:
                print("problem : ", n1, n2,edgeid_to_mask_dict[(n1,n2)], edgeid_to_mask_dict[(n2,n1)] )"""

    print(f"Total score: {total_score/len(explained_edges)}")

    explained_graph.add_edges_from(list(explained_edges))
    nx.draw_networkx_nodes(explained_graph,  pos=coordinates_arr, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(explained_graph, pos=coordinates_arr, font_size=10, ax=ax)
    nx.draw_networkx_edges(explained_graph,  pos=coordinates_arr, ax=ax)
    nx.draw_networkx_edge_labels(explained_graph, edge_labels=edge_label_dict, font_size=10, pos=coordinates_arr, ax=ax)
    
    # print(test_graph.edge_index.shape)
    


def plot_all_and_explained_edges(test_graph, path, file_name, coordinates_arr, edges_idx, cc_threshold = 5):
    original_graph = utils.to_networkx(test_graph)

    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    for ind, val in enumerate(edges_idx):
        if val.item():
            explained_edges.append((test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()))

    options = ['r','b','y','g']
    node_color_dict = dict()
    edge_color_dict = dict()
    for node in original_graph.nodes:
         node_color_dict[node] = options[1]

    for edge in original_graph.edges:
        n1, n2 = edge
        edge_color_dict[(n1, n2)] = options[1]
        edge_color_dict[(n2, n1)] = options[1]

    for ind, val in enumerate(edges_idx):
        #for g_exp in gexp_edges:
        n1, n2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()

        if val.item():
            edge_color_dict[(n1, n2)] = options[0]
            edge_color_dict[(n2, n1)] = options[0]
            node_color_dict[n1] = options[0]
            node_color_dict[n2] = options[0]
            
    colors_node = []
    for node in original_graph.nodes:
        colors_node.append(node_color_dict[node])
    
    colors_edge = []
    for edge in original_graph.edges:
        n1, n2 = edge
        colors_edge.append(edge_color_dict[(n1, n2)])
        colors_edge.append(edge_color_dict[(n2, n1)])

    
    nx.draw_networkx_nodes(original_graph, node_color=colors_node, pos=coordinates_arr, node_size=30)
    nx.draw_networkx_edges(original_graph, pos=pos_1,  arrows=False)
    nx.draw_networkx_edges(original_graph, edgelist=explained_edges, edge_color=options[0], pos=coordinates_arr,  arrows=False)
    plt.savefig(os.path.join(path, file_name), dpi = 300)
    plt.clf()

# Figure 1 - a
def box_plot_cov(x_col, y_col, fl_path):
    dataset = TissueDataset(os.path.join(S_PATH,"../data/JacksonFischer", "month"))
    substr_list = ["ll", "ul", "ur", "lr"]
    img_num_lst = []
    clinical_type_lst = []
    out_lst = []
    for data in dataset:
        if data.clinical_type!="nan":
            if any(substring in data.img_id for substring in substr_list):
                if data.img_id[:-2] not in img_num_lst:
                    img_num_lst.append(data.img_id[:-2])
                    if y_col=="Age":
                        out_lst.append(data.age.item())
                    elif y_col=="OS-Month":
                        out_lst.append(data.osmonth.item())
                    else:
                        raise ValueError("A valid column name should be provided!")
                    clinical_type_lst.append(data.clinical_type)
            else:
                img_num_lst.append(data.img_id)
                if y_col=="Age":
                    out_lst.append(data.age.item())
                elif y_col=="OS-Month":
                    out_lst.append(data.osmonth.item())
                else:
                    raise ValueError("A valid column name should be provided!")
                clinical_type_lst.append(data.clinical_type)
            
            
            
    df = pd.DataFrame(list(zip(img_num_lst, clinical_type_lst, out_lst)),
               columns =["Image Number", x_col, y_col])
    print(df)
    sns.set(style="darkgrid")
    print(df["Clinical Type"].unique())
    my_pal = {"TripleNeg": "b", "HR-HER2+": "y", "HR+HER2-":"g", "HR+HER2+":"r"}
    ax = sns.boxplot( x=df[x_col], y=df[y_col], palette=my_pal )

    # adding transparency to colors
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    plt.savefig(fl_path)
    plt.clf()



# box_plot_cov(x_col="Clinical Type", y_col='OS-Month', fl_path=f"{PLOT_PATH}/manuscript_figures/survibility_plot.pdf")
# box_plot_cov(x_col="Clinical Type", y_col='Age', fl_path=f"{PLOT_PATH}/manuscript_figures/age_plot.pdf")


def plot_age_vs_survibility(fl_name="age_vs_survibility"):
    fig, ax = plt.subplots()

    dataset = TissueDataset(os.path.join(S_PATH,"../data/JacksonFischer", "month"))
    substr_list = ["ll", "ul", "ur", "lr"]
    img_num_lst = []
    clinical_type_lst = []
    age_lst = []
    osmonth_lst = []

    for data in dataset:
        if data.clinical_type!="nan":
            if any(substring in data.img_id for substring in substr_list):
                if data.img_id[:-2] not in img_num_lst:
                    img_num_lst.append(data.img_id[:-2])
                    age_lst.append(data.age.item())
                    osmonth_lst.append(data.osmonth.item())
                    clinical_type_lst.append(data.clinical_type)
            else:
                img_num_lst.append(data.img_id)
                age_lst.append(data.age.item())
                osmonth_lst.append(data.osmonth.item())
                clinical_type_lst.append(data.clinical_type)

    
    df_tvt = pd.DataFrame(list(zip(img_num_lst, clinical_type_lst, osmonth_lst, age_lst)),
               columns =["Image Number", "Clinical Type", "OS-Month", "Age"])
    print(df_tvt)
    # fig, axs = plt.subplots(1,3,figsize=(20,5))
    # colors = {'TripleNeg':'red', 'HR+HER2-':'green', 'HR-HER2+':'blue', 'HR+HER2+':'orange'}
    labels = ['TripleNeg', 'HR-HER2+', 'HR+HER2-', 'HR+HER2+']
    colors = ['b', 'y', 'g', 'r']
    
    # print(len(set_img),set_img)
    for idxlbl, lbl in enumerate(labels):
        # df_temp = df.loc[(df['Train Val Test'] == val) & (df['Clinical Type'] == lbl)]
        df_temp = df_tvt.loc[(df_tvt['Clinical Type'] == lbl)]
        # print(lbl, val)
        # print(df_temp)
        ax.scatter(x=df_temp['OS-Month'], y=df_temp['Age'], color= colors[idxlbl], label=lbl)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 100)
        ax.set_xlabel('OS-Month')
        ax.set_ylabel('Age')
        

    plt.savefig(os.path.join(PLOT_PATH, "manuscript_figures", f"{fl_name}.pdf"))

# plot_age_vs_survibility(fl_name="age_vs_survibility")
# plot_cell_count_distribution("./data/JacksonFischer/raw/basel_zurich_preprocessed_compact_dataset.csv")

# Visualization functions kept here
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import custom_tools
import numpy as np

def classify_breast_cancer(row):
    er = row['ER Status']
    pr = row['PR Status']
    her2 = row['HER2 Status']
    
    hr_positive = (er == 'Positive') or (pr == 'Positive')

    if not hr_positive and her2 == 'Negative':
        return 'TripleNeg'
    elif not hr_positive and her2 == 'Positive':
        return 'HR-HER2+'
    elif hr_positive and her2 == 'Negative':
        return 'HR+HER2-'
    elif hr_positive and her2 == 'Positive':
        return 'HR+HER2+'
    else:
        return np.nan  # In case of unexpected values


def visualize_clinical_data(c_data=None, s_c_data=None, clinical_type_column_name = "clinical_type", plt_path = "../plots/manuscript_figures"):
    c_data  = pd.read_csv("../data/METABRIC/brca_metabric_clinical_data.tsv", sep="\t", index_col=False)
    s_c_data = pd.read_csv("../data/METABRIC/single_cell_data.csv", index_col=False)
    c_data.columns = c_data.columns.str.strip()
    c_data['Subtype'] = c_data.apply(classify_breast_cancer, axis=1)
    # Print columns
    print("Clinical data columns: ", c_data.columns)
    print("Single cell data columns: ", s_c_data.columns)

    # Keep rows in c_data with PIDs in single_cell_data
    c_data = c_data[c_data["Patient ID"].isin(s_c_data["metabricId"])]

    if clinical_type_column_name not in c_data.columns:
        c_data = custom_tools.type_processor(c_data)

    # Define custom order
    custom_order = ["TripleNeg", "HR-HER2+", "HR+HER2-", "HR+HER2+"]

    # Define darker shades
    my_pal = {"TripleNeg": "#4682B4",  # Steel Blue (Darker Blue)
            "HR-HER2+": "#DAA520",   # Goldenrod (Darker Yellow)
            "HR+HER2-": "#228B22",   # Forest Green (Darker Green)
            "HR+HER2+": "#B22222"}   # Firebrick (Darker Red)

    # Set figure size
    plt.figure(figsize=(8, 6))

    # Create the box plot with pastel colors
    sns.boxplot(data=c_data, x='Subtype', y='Overall Survival (Months)', 
                order=custom_order, palette=my_pal)

    # Improve plot readability
    plt.xticks(rotation=45)
    plt.title('Distribution of Overall Survival (Months) per Subtype')
    plt.xlabel('Breast Cancer Subtype')
    plt.ylabel('Overall Survival (Months)')
    plt.tight_layout()
    plt.savefig(os.path.join(plt_path, "METABRIC_clinical_type_vs_survival.pdf"))
    plt.clf()

    

def UMAP_plot(embeddings, emb_id, related_data, attribute_name, experiment_name=None, job_id=None, mode="FC", attributes = np.empty(0)):
    """Creates 2D UMAP plot for given embeddings and related data.
    It assumes both data are in the same order.

    Args:
        embeddings (_type_): Embeddings of the data
        related_data (_type_): Clinical variables about the embeddings
        attribute_name (_type_): In which attribute the embeddings are colored
        attributes (numpy.array): If the attribute values are already given, they can be passed as a numpy array

    Returns:
        Outputs UMAP plot and saves it as pdf
    """
    emd512 = embeddings
    emd_array = emd512
    # Convert the embeddings tensor to a numpy array
    if type(emd512) != np.ndarray:
        emd_array = emd512.numpy()

    def handle_types(data):
        if isinstance(data, list):
            return data[0]
        # if tensor
        elif isinstance(data, torch.Tensor):
            return data.item()
    # Extract the attribute values from the related_data list
    if attributes.size == np.empty(0).size:
        attributes = [handle_types(getattr(data_batch, attribute_name)) for data_batch in related_data]
    
    # Get unique attribute values and create a colormap
    unique_attributes = list(set(attributes))
    num_attributes = len(unique_attributes)

    if num_attributes <= 10:
        color_map = plt.get_cmap('tab10', num_attributes)
        attribute_color = {attr: color_map(i) for i, attr in enumerate(unique_attributes)}
        legend_type = 'class'
    else:
        attribute_color = {attr: plt.cm.get_cmap('viridis')(i/num_attributes) for i, attr in enumerate(unique_attributes)}
        legend_type = 'value'

    # Convert attributes to colors based on the colormap
    attribute_colors = [attribute_color[attr] for attr in attributes]

    # Create a UMAP reducer
    reducer = umap.UMAP(random_state=42, n_neighbors=10)

    # Apply UMAP transformation
    umap_result = reducer.fit_transform(emd_array)

    # Plot the UMAP results with colored attributes and legend
    plt.figure(figsize=(10, 8))

    # Create scatter plot for each attribute with corresponding color
    for attr, color in attribute_color.items():
        mask = np.array(attributes) == attr
        plt.scatter(umap_result[mask, 0], umap_result[mask, 1], c=[color], label=str(attr), s=10)

    # Add legend with appropriate title and min/max value entries
    legend = plt.legend(title=f'Legend ({legend_type})')

    if legend_type == 'value':
        min_attr = min(unique_attributes)
        max_attr = max(unique_attributes)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=attribute_color[min_attr], markersize=10, label=f'Min: {min_attr:.2f}'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=attribute_color[max_attr], markersize=10, label=f'Max: {max_attr:.2f}')]
        plt.legend(handles=legend_elements, title=f'Legend ({legend_type})')
    else:
        plt.legend(title=f'Legend ({legend_type})')

        
    plt.title(f'UMAP Projection with Colored Attributes and Legend: {attribute_name}')

    out_path = os.path.join(PLOT_PATH, experiment_name, job_id)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    print("Saving plot here:", out_path)
    plt.savefig(os.path.join(out_path, f"{attribute_name}_{mode}_{emb_id}.pdf"))
