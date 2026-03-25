import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from tqdm import tqdm

# dataset = "METABRIC"
# dataset = "JacksonFischer"
dataset = "Lung"

RAW_DATA_PATH = os.path.join("../data",dataset,"raw")
OUT_DATA_PATH = os.path.join("../data", "out_data", dataset)
PLOT_PATH = os.path.join("../plots", dataset,"graph_voronoi_plots")
GRAPH_DIV_THR = 2500
CELL_COUNT_THR = 100



def get_dataset_from_csv(path="a", index_col=None):
    """
    path = None
    if dataset == "JacksonFischer":
        path = "../data/JacksonFischer/raw/basel_zurich_preprocessed_compact_dataset.csv"
    elif dataset == "METABRIC":
        path = "../data/METABRIC/raw/merged_data.csv"
    """
    
    if index_col!=None:
        return pd.read_csv(path, index_col=index_col)
    else:
        return pd.read_csv(path)


def get_cell_count_df(df_dataset, cell_count_thr, sample_col = "ImageNumber"):    
    # df_dataset = get_dataset_from_csv(path, index_col=index_col)
    df_cell_count = df_dataset.groupby(by=[sample_col]).size().reset_index(name='Cell Counts')

    # first_quartile value is 705.75
    df_cell_count = df_cell_count.loc[(df_cell_count['Cell Counts'] >= cell_count_thr)]#  & (df_cell_count['Cell Counts'] <= 2000)]
    return df_cell_count

def generate_graphs_using_points(df_image, imgnum_edge_thr_dict, img_num,  pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos=None, plot=False,PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    

    points = df_image[[loc_x_col, loc_y_col]].to_numpy()
    # print(points)
    
    # print(df_image.columns)
    
    # point_labels = list(df_image["ObjectNumber"].values)
    
    # divide the graph based on the mean x and mean y values if # of cells is greater than 75 percentile
    if pos:
        img_num_lbl = f"{img_num}{pos}"
    else:
        img_num_lbl = img_num
    
    point_to_lbl_dict = dict()
    for ind, pt in enumerate(points):
        point_to_lbl_dict[tuple(pt)]  = ind
    
    incidence_set = set()

    tri = Delaunay(points)
    small_edges = set()
    large_edges = set()
    for tr in tri.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]

            if (edge_idx1, edge_idx0) in small_edges:
                continue  # already visited this edge from other side
            if (edge_idx1, edge_idx0) in large_edges:
                continue
            p0 = points[edge_idx0]
            p1 = points[edge_idx1]

            # print(p0, p1)
            edge_length = np.linalg.norm(p1 - p0)
            if  edge_length <  imgnum_edge_thr_dict[img_num]:
                small_edges.add((edge_idx0, edge_idx1))
                incidence_set.add((point_to_lbl_dict[tuple(p0)], point_to_lbl_dict[tuple(p1)], edge_length))
                incidence_set.add((point_to_lbl_dict[tuple(p1)], point_to_lbl_dict[tuple(p0)], edge_length))
            else:
                large_edges.add((edge_idx0, edge_idx1))

    if plot:
        plt.clf()
        """plt.figure(dpi=300)
        plt.plot(points[:, 0], points[:, 1], '.', markersize=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_onlycells.png")

        plt.clf()
        plt.figure(dpi=300)
        plt.triplot(points[:,0], points[:,1], tri.simplices, linewidth=1, markersize=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_delaunay.png")

        plt.clf()"""
        
        plt.figure(dpi=300)
        plt.plot(points[:, 0], points[:, 1], '.', markersize=2)
        for i, j in small_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'r', linewidth=1)
        for i, j in large_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'c--', alpha=0.2, linewidth=1)

        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_adapt_thr.pdf")
        
        # print(tri.simplices)

        # plt.triplot(points[:,0], points[:,1], tri.simplices)
        # plt.plot(points[:,0], points[:,1], '.')
        # fig = plt.get_figure()

        
        plt.clf()
        
        plt.figure(dpi=300)
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_voronoi.pdf", dpi=300)
        plt.clf()
    
    
    edge_index_arr = np.array([list(edge)[:2] for edge in incidence_set], dtype=np.int32)
    edge_length_arr= np.array([list(edge)[-1] for edge in incidence_set])
    
    
    assert edge_index_arr.shape[0]==edge_length_arr.shape[0]

    clinical_info_dict = dict()

    # TODO: Refactor this part
    if "JacksonFischer" in RAW_DATA_PATH:
        clinical_info_dict["grade"] = df_image["grade"].values[0]
        clinical_info_dict["tumor_size"] = df_image["tumor_size"].values[0]
        clinical_info_dict["treatment"] = df_image["treatment"].values[0]
        clinical_info_dict["age"] = df_image["age"].values[0]
        clinical_info_dict["DiseaseStage"] = df_image["DiseaseStage"].values[0]
        clinical_info_dict["diseasestatus"] = df_image["diseasestatus"].values[0]
        clinical_info_dict["clinical_type"] = df_image["clinical_type"].values[0]
        clinical_info_dict["DFSmonth"] = df_image["DFSmonth"].values[0]
        clinical_info_dict["OSmonth"] = df_image["OSmonth"].values[0]
        clinical_info_dict["Patientstatus"] = df_image["Patientstatus"].values[0]
        # clinical_info_dict["cell_type"] = df_image["cell_type"].values[0]
        # clinical_info_dict["class"] = df_image["class"].values[0]
        clinical_info_dict["cell_count"] = len(df_image)
    elif "METABRIC" in RAW_DATA_PATH:
        metabric_clinical_feat = ['SOM_nodes', 'pg_cluster', 'description',
                                  'Study ID', 'Patient ID', 'PID', 'age', 'Type of Breast Surgery',
                                  'cell_type', 'Cancer Type Detailed', 'Cellularity', 'treatment',
                                  'Pam50 + Claudin-low subtype', 'Cohort', 'ER status measured by IHC',
                                  'ER Status', 'grade', 'HER2 status measured by SNP6', 'HER2 Status',
                                  'Tumor Other Histologic Subtype', 'Hormone Therapy',
                                  'Inferred Menopausal State', 'Integrative Cluster',
                                  'Primary Tumor Laterality', 'Lymph nodes examined positive',
                                  'Mutation Count', 'DiseaseStage', 'Oncotree Code', 'OSmonth',
                                  'Overall Survival Status', 'PR Status', 'Radio Therapy', 'DFSmonth',
                                  'Relapse Free Status', 'Number of Samples Per Patient', 'Sample Type',
                                  'Sex', '3-Gene classifier subtype', 'TMB (nonsynonymous)', 'tumor_size',
                                   'Tumor Stage', 'diseasestatus', 'clinical_type']
    elif "Lung" in RAW_DATA_PATH:
        lung_clinical_feat = ["sample_id", "Sex","Age","BMI","Smoking Status","Pack Years","Stage","Progression","Death","Survival or loss (years)","Predominant histological pattern"]
        marker_list = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "Histone H3", "MPO", "Pancytokeratin"]
        for col in lung_clinical_feat:
            clinical_info_dict[col] = df_image[col].values[0]
            if col == "sample_id" and pos:
                clinical_info_dict[col] = f"{clinical_info_dict[col]}_{pos}"
        
        clinical_info_dict["cell_count"] = len(df_image)


    # save the edge indices and as a list 
    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_edge_index_length.pickle'), 'wb') as handle:
        pickle.dump((edge_index_arr, edge_length_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    nonfeat_cols = []
    for col in df_image.columns:
        if "MeanIntensity" not in col:
            nonfeat_cols.append(col) 
    # save the feature vector as numpy array, first column is the cell id
    # "ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y", "PID", "grade", "tumor_size", "age", "treatment", "DiseaseStage", "diseasestatus", "clinical_type", "DFSmonth", "OSmonth"

    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_features.pickle'), 'wb') as handle:
        if "Lung" in RAW_DATA_PATH:
            pickle.dump(np.array(df_image[marker_list]), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:   
            pickle.dump(np.array(df_image.drop(nonfeat_cols, axis=1)), handle, protocol=pickle.HIGHEST_PROTOCOL)

    if "METABRIC" in RAW_DATA_PATH or "Lung" in RAW_DATA_PATH:
        with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_ct_class.pickle'), 'wb') as handle:
            pickle.dump(df_image[["cell_type"]].to_numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_ct_class.pickle'), 'wb') as handle:
            pickle.dump(df_image[["cell_type", "class"]].to_numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # save the feature vector as numpy array, first column is the cell id
    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_coordinates.pickle'), 'wb') as handle:
        pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_clinical_info.pickle'), 'wb') as handle:
        pickle.dump(clinical_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def generate_graphs_using_points_lung_data(df_image, imgnum_edge_thr_dict, img_num,  pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos=None, plot=False,PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    print("df_image", df_image)
    points = df_image[[loc_x_col, loc_y_col]].to_numpy()

    if pos:
        img_num_lbl = f"{img_num}{pos}"
    else:
        img_num_lbl = img_num
    
    point_to_lbl_dict = dict()
    for ind, pt in enumerate(points):
        point_to_lbl_dict[tuple(pt)]  = ind
    
    incidence_set = set()

    tri = Delaunay(points)
    small_edges = set()
    large_edges = set()
    for tr in tri.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]

            if (edge_idx1, edge_idx0) in small_edges:
                continue  # already visited this edge from other side
            if (edge_idx1, edge_idx0) in large_edges:
                continue
            p0 = points[edge_idx0]
            p1 = points[edge_idx1]

            # print(p0, p1)
            edge_length = np.linalg.norm(p1 - p0)
            if  edge_length <  imgnum_edge_thr_dict[img_num]:
                small_edges.add((edge_idx0, edge_idx1))
                incidence_set.add((point_to_lbl_dict[tuple(p0)], point_to_lbl_dict[tuple(p1)], edge_length))
                incidence_set.add((point_to_lbl_dict[tuple(p1)], point_to_lbl_dict[tuple(p0)], edge_length))
            else:
                large_edges.add((edge_idx0, edge_idx1))

    if plot:
        plt.clf()
        plt.figure(dpi=300)
        plt.plot(points[:, 0], points[:, 1], '.', markersize=2)
        for i, j in small_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'r', linewidth=1)
        for i, j in large_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'c--', alpha=0.2, linewidth=1)

        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_adapt_thr.pdf")
                
        plt.clf()
        
        plt.figure(dpi=300)
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_voronoi.pdf", dpi=300)
        plt.clf()
    
    
    edge_index_arr = np.array([list(edge)[:2] for edge in incidence_set], dtype=np.int32)
    edge_length_arr= np.array([list(edge)[-1] for edge in incidence_set])
    
    
    assert edge_index_arr.shape[0]==edge_length_arr.shape[0]

    clinical_info_dict = dict()

    marker_list = ['Myeloperoxidase (MPO)', 'FSP1 / S100A4', 'SMA', 'Histone H3', 'FAP',
       'HLA-DR', 'CD146', 'Cadherin-11', 'Carbonic Anhydrase IX',
       'Collagen I + Fibronectin', 'VCAM1', 'CD20', 'CD68',
       'Indoleamine 2- 3-dioxygenase (IDO)', 'CD3', 'Podoplanin', 'MMP11',
       'CD279 (PD-1)', 'CD73', 'MMP9', 'p75 (CD271)', 'TCF1/TCF7', 'CD10',
       'Vimentin', 'FOXP3', 'CD45RA + CD45R0', 'PNAd', 'CD8a',
       'CD248 / Endosialin', 'LYVE-1', 'PDGFR-b', 'CD34', 'CD4', 'vWF + CD31',
       'CXCL12', 'CCL21', 'Pan Cytokeratin + Keratin Epithelial', 'Cadherin-6',
       'Iridium_191', 'Iridium_193', 'Ki-67', 'Caveolin-1', 'CD15']
    lung_old_clinical_feat = ['ImageNumber', 'Area', 'Compartment', 
                            'BatchID', 'Panel', 'TmaID', 'TmaBlock', 'acID', 'mclust',
                            'TMA', 'Tma_ac', 'ROI_xy',
                            'RoiID', 'Patient_Nr', 'DX.name', 'Age',
                            'Gender', 'Typ', 'Grade', 'Size', 'Vessel', 'Pleura', 'T.new', 'N',
                            'M.new', 'Stage', 'R', 'Chemo', 'Radio', 'Chemo3', 'Radio4', 'Relapse',
                            'Chemo5', 'Radio6', 'DFS', 'Ev.O', 'OS', 'Smok', 'Nikotin', 'ROI',
                            'Patient_ID', 'LN.Met', 'Dist.Met', 'NeoAdj', 'Area_px_Stroma',
                            'Area_px_Tumour', 'Area_px_Core', 'Area_mm_Stroma', 'Area_mm_Tumour',
                            'Area_mm_Core', 'Sample_ID']
    marker_list = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "Histone H3", "MPO", "Pancytokeratin"]
    lung_clinical_feat = ["sample_id", "Sex","Age","BMI","Smoking Status","Pack Years","Stage","Progression","Death","Survival or loss (years)","Predominant histological pattern"]
    print("df_image", df_image)
    for col in lung_clinical_feat:
        clinical_info_dict[col] = df_image[col].values[0]
    
    clinical_info_dict["cell_count"] = len(df_image)
    

    # save the edge indices and as a list 
    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_edge_index_length.pickle'), 'wb') as handle:
        pickle.dump((edge_index_arr, edge_length_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_features.pickle'), 'wb') as handle:
        pickle.dump(np.array(df_image[marker_list]), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_ct_class.pickle'), 'wb') as handle:
        pickle.dump(df_image[["cell_type"]].to_numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # save the feature vector as numpy array, first column is the cell id
    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_coordinates.pickle'), 'wb') as handle:
        pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_clinical_info.pickle'), 'wb') as handle:
        pickle.dump(clinical_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def get_edge_length_dist(df_dataset, cell_count_thr, quant,sample_col= "ImageNumber", loc_x="Location_Center_X", loc_y="Location_Center_Y",  plot_dist=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    """
    Calculate the distribution of edge lengths for each image in the dataset.

    Args:
        cell_count_thr (int): Cell count threshold used to filter the dataset.
        quant (float): The quantile value to calculate the edge length threshold.
        plot_dist (bool, optional): Whether to plot and save the edge length distribution for each image.
                                    Defaults to False.

    Returns:
        dict: A dictionary containing the image number as key and its corresponding edge length threshold as value.
    """
    df_cell_count = get_cell_count_df(df_dataset, cell_count_thr, sample_col=sample_col)
    imgnum_edge_thr_dict = dict()
    all_edge_count_list = []
    
    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        
        img_num = row[sample_col]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset[sample_col]==img_num]
        points = [list(item) for item in list(df_image[[loc_x, loc_y]].values)]
        # print(points)
        # point_labels = [list(item) for item in list(df_image["ObjectNumber"].values)]
        point_to_lbl_dict = dict()
        for ind, pt in enumerate(points):
            point_to_lbl_dict[tuple(pt)]  = ind
        
        tri = Delaunay(points)
        edge_set = set()
        edge_length = []
        # tri.simplices are array of points each element in the array has 3 points
        for tr in tri.simplices:
            for i in range(3):
                edge_idx0 = tr[i]
                edge_idx1 = tr[(i+1)%3]

                if (edge_idx1, edge_idx0) in edge_set:
                    continue  # already visited this edge from other side
                edge_set.add((edge_idx0, edge_idx1))
                p0 = points[edge_idx0]
                p1 = points[edge_idx1]
                
                # print(np. where(points ==p0))
                
                # print(p0, p1)

                edge_length.append(np.linalg.norm(np.array(p1) - np.array(p0)))
        

        edge_thr = np.quantile(edge_length, quant)
        imgnum_edge_thr_dict[img_num] = edge_thr
    
        if plot_dist:
            plt.clf()
            plt.figure(dpi=300)
            edge_dist_plot = sns.displot(data=edge_length, kde=True)
            plt.axvline(x=edge_thr, linestyle='--', color="black")
            # fig = edge_dist_plot.get_figure()
            plt.savefig(f"{PLOT_PATH}/{img_num}_edge_distribution.pdf")
            plt.clf()
        
        
        # edge_length
        all_edge_count_list.append(edge_length)
    
    with open(os.path.join(RAW_DATA_PATH, 'edge_thr.pickle'), 'wb') as handle:
        pickle.dump(imgnum_edge_thr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
             

def create_graphs_delauney_triangulation(df_dataset, cell_count_thr=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, sample_col= "ImageNumber", pid_col = "PID", x_loc ="Location_Center_X", y_loc = "Location_Center_Y", divide_sample=True,  plot=False, RAW_DATA_PATH = RAW_DATA_PATH, PLOT_PATH=PLOT_PATH):

    # df_dataset = get_dataset_from_csv(data_path)
    # get_cell_count_df(df_dataset, cell_count_thr, sample_col = "ImageNumber"):    
    df_cell_count = get_cell_count_df(df_dataset, cell_count_thr,sample_col =sample_col)
    imgnum_edge_thr_dict = dict()
    # print("df_cell_count", df_cell_count)

    # print(df_dataset["PID"])

    with open(os.path.join(RAW_DATA_PATH, 'edge_thr.pickle'), 'rb') as handle:
            imgnum_edge_thr_dict = pickle.load(handle)

    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        # print(row)
        img_num = row[sample_col]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset[sample_col]==img_num]
        pid = df_image[pid_col].values[0]
    
        new_cell_ids = list(range(len(df_image)))
        
        # print(cell_count, GRAPH_DIV_THR, divide_sample)
        # TODO: make this parametric 
        if cell_count >= GRAPH_DIV_THR and divide_sample:
            # print("Dividing sample")
            # find the center of the cells
            x_center, y_center = df_image[[x_loc, y_loc]].describe().loc["mean"][x_loc], df_image[[x_loc, y_loc]].describe().loc["mean"][y_loc]
            # print(x_center, y_center)
            # ll lower-left  ul upper-left lr lower right points 
            """ll_points = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] <= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()
            ul_points = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] >= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()
            lr_points = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] <= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()
            ur_points = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] >= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()"""

            ll_df_image = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] <= y_center))]
            ul_df_image = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] >= y_center))]
            lr_df_image = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] <= y_center))]
            ur_df_image = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] >= y_center))]


            """generate_graphs_using_points(ll_df_image, imgnum_edge_thr_dict, img_num, pid, "ll", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(ul_df_image, imgnum_edge_thr_dict, img_num, pid, "ul", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(lr_df_image, imgnum_edge_thr_dict, img_num, pid, "lr", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(ur_df_image, imgnum_edge_thr_dict, img_num, pid, "ur", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)"""
            generate_graphs_using_points(ll_df_image, imgnum_edge_thr_dict, img_num, pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos="ll", plot=plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(ul_df_image, imgnum_edge_thr_dict, img_num, pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos="ul", plot=plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(lr_df_image, imgnum_edge_thr_dict, img_num, pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos="lr", plot=plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(ur_df_image, imgnum_edge_thr_dict, img_num, pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos="ur", plot=plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)

        else:
            # points = df_image[["Location_Center_X", "Location_Center_Y"]].to_numpy()
            # generate_graphs_using_points(df_image, imgnum_edge_thr_dict, img_num, pid, loc_x_col="Center_X", loc_y_col="Center_Y", pos=None, plot = plot, PLOT_PATH=PLOT_PATH)
            generate_graphs_using_points(df_image, imgnum_edge_thr_dict, img_num, pid, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos=None, plot = plot, PLOT_PATH=PLOT_PATH)
       
def check_cell_ids_sequential():
    df_dataset = get_dataset_from_csv()
    df_cell_count = get_cell_count_df(0)


    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        # print(row)
        img_num = row["ImageNumber"]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset["ImageNumber"]==img_num]

        pid = df_image["PID"].values[0]
        points = df_image[["Location_Center_X", "Location_Center_Y"]].to_numpy()
        point_labels = list(df_image["ObjectNumber"].values)
        max_val = max(point_labels)
        # print(max_val)
        #print(list(range(1,max_val+1)))
        # print(point_labels)
        print(list(range(1,max_val+1)))
        assert point_labels==list(range(1,max_val+1))



# To generate the dataset run the below functions
# 1) get_edge_length_dist(CELL_COUNT_THR, 0.975, plot_dist=True)
# 2) get_cell_count_df(CELL_COUNT_THR)
# 3) create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True)

# create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True)

# check if all cell ids are available 
# check_cell_ids_sequential()

def data_processing_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    # TODO make this parametric
    from data_preparation import create_preprocessed_sc_feature_fl
    print("Creating compact dataset...")
    # create_preprocessed_sc_feature_fl()
    print("Calculating edge length distribution....")
    get_edge_length_dist(data_path=data_path, cell_count_thr=CELL_COUNT_THR, quant= 0.975, plot_dist=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)
    print("Calculating cell count distribution....")
    df_cell_count = get_cell_count_df(CELL_COUNT_THR, path=data_path)
    quartile = int(np.quantile(df_cell_count["Cell Counts"], 0.50))

    print("Creating graphs...")
    create_graphs_delauney_triangulation(cell_count_thr=CELL_COUNT_THR, GRAPH_DIV_THR=quartile, plot=True,RAW_DATA_PATH = RAW_DATA_PATH, data_path=data_path, PLOT_PATH=PLOT_PATH)

# data_processing_pipeline("./data/JacksonFischer/raw/merged_preprocessed_dataset.csv")

def data_processing_metabric_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    # TODO make this parametric
    print("Calculating cell count distribution....")
    df_cell_count = get_cell_count_df(df_dataset, CELL_COUNT_THR, sample_col = "Sample_ID")
    # print("Creating compact dataset...")
    # METABRIC_preprocess(visualize = False)
    print("Calculating edge length distribution....")
    get_edge_length_dist(data_path=data_path, cell_count_thr=CELL_COUNT_THR, quant= 0.975, plot_dist=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)
    
    print("Calculating cell count distribution....")
    df_cell_count = get_cell_count_df(CELL_COUNT_THR, path=data_path)
    quartile = int(np.quantile(df_cell_count["Cell Counts"], 0.50))
    print(RAW_DATA_PATH)
    print("Creating graphs...")
    create_graphs_delauney_triangulation(cell_count_thr=CELL_COUNT_THR, GRAPH_DIV_THR=quartile, plot=True,RAW_DATA_PATH = RAW_DATA_PATH, data_path=data_path, PLOT_PATH=PLOT_PATH)


# data_processing_metabric_pipeline("./data/METABRIC/raw/merged_preprocessed_dataset.csv",CELL_COUNT_THR=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)

def data_processing_lung_old_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    # print("Creating compact dataset...")
    # METABRIC_preprocess(visualize = False)
    df_dataset = get_dataset_from_csv(data_path, index_col=0)
    get_edge_length_dist(df_dataset, cell_count_thr=CELL_COUNT_THR, quant=0.975, sample_col="Sample_ID", loc_x="Center_X", loc_y="Center_Y", PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)
    
    print("Calculating cell count distribution....")
    # df_cell_count = get_cell_count_df(CELL_COUNT_THR, path=data_path)
    print(RAW_DATA_PATH)
    print("Creating graphs...")
    create_graphs_delauney_triangulation(df_dataset, cell_count_thr=CELL_COUNT_THR, sample_col= "Sample_ID", pid_col = "Patient_ID",  x_loc ="Center_X", y_loc = "Center_Y", divide_sample=False,  plot=True, RAW_DATA_PATH = RAW_DATA_PATH, PLOT_PATH=PLOT_PATH)

# data_processing_lungold_pipeline("./data/Lung/lung_imc_dataset.csv")

def data_processing_lung_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    # print("Creating compact dataset...")
    # METABRIC_preprocess(visualize = False)
    

    df_dataset = get_dataset_from_csv(data_path, index_col=None)
    get_edge_length_dist(df_dataset, cell_count_thr=CELL_COUNT_THR, quant=0.975, sample_col="sample_id", loc_x="Location_Center_X", loc_y="Location_Center_Y", PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)
    
    print("Calculating cell count distribution....")
    # df_cell_count = get_cell_count_df(CELL_COUNT_THR, path=data_path)
    print(RAW_DATA_PATH)
    print("Creating graphs...")
    create_graphs_delauney_triangulation(df_dataset, cell_count_thr=CELL_COUNT_THR, sample_col= "sample_id", pid_col = "sample_id",  x_loc ="Location_Center_X", y_loc = "Location_Center_Y", divide_sample=False,  plot=False, RAW_DATA_PATH = RAW_DATA_PATH, PLOT_PATH=PLOT_PATH)

# data_processing_lung_pipeline("./data/Lung/raw/merged_preprocessed_dataset.csv")


CODEX_CRC_SOURCE_EXCLUDE_COLUMNS = {
    "Unnamed: 0",
    "CellID",
    "ClusterID",
    "EventID",
    "File Name",
    "Region",
    "TMA_AB",
    "TMA_12",
    "Index in File",
    "groups",
    "patients",
    "spots",
    "cell_id:cell_id",
    "tile_nr:tile_nr",
    "X:X",
    "Y:Y",
    "X_withinTile:X_withinTile",
    "Y_withinTile:Y_withinTile",
    "Z:Z",
    "size:size",
    "Profile_Homogeneity:Fiter1",
    "ClusterSize",
    "ClusterName",
    "neighborhood10",
    "neighborhood number final",
    "neighborhood name",
}

CODEX_CRC_PREPROCESSED_EXCLUDE_COLUMNS = {
    "sample_id",
    "cell_id",
    "cell_type",
    "Location_Center_X",
    "Location_Center_Y",
    "area_pixels",
    "patient_id",
    "group_id",
    "spot_id",
    "region_id",
    "TMA_AB",
    "TMA_12",
    "cluster_id",
    "event_id",
    "tile_nr",
    "z_plane",
    "profile_homogeneity",
    "cluster_size",
    "neighborhood10",
    "neighborhood_number_final",
    "neighborhood_name",
    "cd4_icos_positive",
    "cd4_ki67_positive",
    "cd4_pd1_positive",
    "cd68_cd163_icos_positive",
    "cd68_cd163_ki67_positive",
    "cd68_cd163_pd1_positive",
    "cd68_icos_positive",
    "cd68_ki67_positive",
    "cd68_pd1_positive",
    "cd8_icos_positive",
    "cd8_ki67_positive",
    "cd8_pd1_positive",
    "treg_icos_positive",
    "treg_ki67_positive",
    "treg_pd1_positive",
}


def get_codex_crc_feature_columns(df_dataset):
    if {"sample_id", "Location_Center_X", "Location_Center_Y"}.issubset(df_dataset.columns):
        exclude_columns = CODEX_CRC_PREPROCESSED_EXCLUDE_COLUMNS
    else:
        exclude_columns = CODEX_CRC_SOURCE_EXCLUDE_COLUMNS

    feature_cols = []
    for col in df_dataset.columns:
        if col in exclude_columns or "+" in col:
            continue
        if pd.api.types.is_numeric_dtype(df_dataset[col]):
            feature_cols.append(col)

    return feature_cols


def create_preprocessed_codex_crc_dataset(data_path, RAW_DATA_PATH):
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    df_dataset = get_dataset_from_csv(data_path, index_col=None)

    if {"sample_id", "cell_type", "Location_Center_X", "Location_Center_Y"}.issubset(df_dataset.columns):
        feature_cols = get_codex_crc_feature_columns(df_dataset)
        merged_output_path = os.path.join(RAW_DATA_PATH, "merged_preprocessed_dataset.csv")
        if os.path.abspath(data_path) != os.path.abspath(merged_output_path):
            df_dataset.to_csv(merged_output_path, index=False)
        return df_dataset, feature_cols

    feature_cols = get_codex_crc_feature_columns(df_dataset)
    rename_columns = {
        "File Name": "sample_id",
        "cell_id:cell_id": "cell_id",
        "ClusterName": "cell_type",
        "X:X": "Location_Center_X",
        "Y:Y": "Location_Center_Y",
        "size:size": "area_pixels",
        "patients": "patient_id",
        "groups": "group_id",
        "spots": "spot_id",
        "Region": "region_id",
        "ClusterID": "cluster_id",
        "EventID": "event_id",
        "tile_nr:tile_nr": "tile_nr",
        "Z:Z": "z_plane",
        "Profile_Homogeneity:Fiter1": "profile_homogeneity",
        "ClusterSize": "cluster_size",
        "neighborhood number final": "neighborhood_number_final",
        "neighborhood name": "neighborhood_name",
        "CD4+ICOS+": "cd4_icos_positive",
        "CD4+Ki67+": "cd4_ki67_positive",
        "CD4+PD-1+": "cd4_pd1_positive",
        "CD68+CD163+ICOS+": "cd68_cd163_icos_positive",
        "CD68+CD163+Ki67+": "cd68_cd163_ki67_positive",
        "CD68+CD163+PD-1+": "cd68_cd163_pd1_positive",
        "CD68+ICOS+": "cd68_icos_positive",
        "CD68+Ki67+": "cd68_ki67_positive",
        "CD68+PD-1+": "cd68_pd1_positive",
        "CD8+ICOS+": "cd8_icos_positive",
        "CD8+Ki67+": "cd8_ki67_positive",
        "CD8+PD-1+": "cd8_pd1_positive",
        "Treg-ICOS+": "treg_icos_positive",
        "Treg-Ki67+": "treg_ki67_positive",
        "Treg-PD-1+": "treg_pd1_positive",
    }

    columns_to_keep = [
        "File Name",
        "cell_id:cell_id",
        "ClusterName",
        "X:X",
        "Y:Y",
        "size:size",
        "patients",
        "groups",
        "spots",
        "Region",
        "TMA_AB",
        "TMA_12",
        "ClusterID",
        "EventID",
        "tile_nr:tile_nr",
        "Z:Z",
        "Profile_Homogeneity:Fiter1",
        "ClusterSize",
        "neighborhood10",
        "neighborhood number final",
        "neighborhood name",
        "CD4+ICOS+",
        "CD4+Ki67+",
        "CD4+PD-1+",
        "CD68+CD163+ICOS+",
        "CD68+CD163+Ki67+",
        "CD68+CD163+PD-1+",
        "CD68+ICOS+",
        "CD68+Ki67+",
        "CD68+PD-1+",
        "CD8+ICOS+",
        "CD8+Ki67+",
        "CD8+PD-1+",
        "Treg-ICOS+",
        "Treg-Ki67+",
        "Treg-PD-1+",
    ] + feature_cols

    df_preprocessed = df_dataset[columns_to_keep].rename(columns=rename_columns)
    merged_output_path = os.path.join(RAW_DATA_PATH, "merged_preprocessed_dataset.csv")
    df_preprocessed.to_csv(merged_output_path, index=False)

    return df_preprocessed, feature_cols


def generate_graphs_using_points_codex_crc(df_image, imgnum_edge_thr_dict, img_num, pid, feature_cols, loc_x_col="Location_Center_X", loc_y_col="Location_Center_Y", pos=None, plot=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    points = df_image[[loc_x_col, loc_y_col]].to_numpy()

    if pos:
        img_num_lbl = f"{img_num}{pos}"
    else:
        img_num_lbl = img_num

    point_to_lbl_dict = dict()
    for ind, pt in enumerate(points):
        point_to_lbl_dict[tuple(pt)] = ind

    incidence_set = set()

    tri = Delaunay(points)
    small_edges = set()
    large_edges = set()
    for tr in tri.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i + 1) % 3]

            if (edge_idx1, edge_idx0) in small_edges:
                continue
            if (edge_idx1, edge_idx0) in large_edges:
                continue
            p0 = points[edge_idx0]
            p1 = points[edge_idx1]

            edge_length = np.linalg.norm(p1 - p0)
            if edge_length < imgnum_edge_thr_dict[img_num]:
                small_edges.add((edge_idx0, edge_idx1))
                incidence_set.add((point_to_lbl_dict[tuple(p0)], point_to_lbl_dict[tuple(p1)], edge_length))
                incidence_set.add((point_to_lbl_dict[tuple(p1)], point_to_lbl_dict[tuple(p0)], edge_length))
            else:
                large_edges.add((edge_idx0, edge_idx1))

    if plot:
        plt.close("all")
        plt.figure(dpi=300)
        plt.plot(points[:, 0], points[:, 1], ".", markersize=2)
        for i, j in small_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], "r", linewidth=1)
        for i, j in large_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], "c--", alpha=0.2, linewidth=1)

        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_adapt_thr.pdf")
        plt.close()

        plt.figure(dpi=300)
        vor = Voronoi(points)
        voronoi_plot_2d(vor, show_vertices=False, line_colors="orange", line_width=1, line_alpha=0.6, point_size=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_voronoi.pdf", dpi=300)
        plt.close()

    edge_index_arr = np.array([list(edge)[:2] for edge in incidence_set], dtype=np.int32)
    edge_length_arr = np.array([list(edge)[-1] for edge in incidence_set])

    assert edge_index_arr.shape[0] == edge_length_arr.shape[0]

    sample_id = df_image["sample_id"].iloc[0]
    neighborhood_mode = df_image["neighborhood_name"].dropna().mode()
    dominant_neighborhood_name = neighborhood_mode.iloc[0] if not neighborhood_mode.empty else np.nan
    dominant_cluster = df_image["cell_type"].mode().iloc[0]

    clinical_info_dict = {
        "sample_id": sample_id if not pos else f"{sample_id}_{pos}",
        "patient_id": df_image["patient_id"].iloc[0],
        "group_id": df_image["group_id"].iloc[0],
        "spot_id": df_image["spot_id"].iloc[0],
        "region_id": df_image["region_id"].iloc[0],
        "TMA_AB": df_image["TMA_AB"].iloc[0],
        "TMA_12": df_image["TMA_12"].iloc[0],
        "cell_count": len(df_image),
        "unique_cell_types": int(df_image["cell_type"].nunique()),
        "dominant_cell_type": dominant_cluster,
        "dominant_neighborhood_name": dominant_neighborhood_name,
    }

    with open(os.path.join(RAW_DATA_PATH, f"{img_num_lbl}_{pid}_edge_index_length.pickle"), "wb") as handle:
        pickle.dump((edge_index_arr, edge_length_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f"{img_num_lbl}_{pid}_features.pickle"), "wb") as handle:
        pickle.dump(np.array(df_image[feature_cols]), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f"{img_num_lbl}_{pid}_ct_class.pickle"), "wb") as handle:
        pickle.dump(df_image[["cell_type"]].to_numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f"{img_num_lbl}_{pid}_coordinates.pickle"), "wb") as handle:
        pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f"{img_num_lbl}_{pid}_clinical_info.pickle"), "wb") as handle:
        pickle.dump(clinical_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_graphs_delauney_triangulation_codex_crc(df_dataset, feature_cols, cell_count_thr=CELL_COUNT_THR, sample_col="sample_id", pid_col="sample_id", plot=False, RAW_DATA_PATH=RAW_DATA_PATH, PLOT_PATH=PLOT_PATH):
    df_cell_count = get_cell_count_df(df_dataset, cell_count_thr, sample_col=sample_col)

    with open(os.path.join(RAW_DATA_PATH, "edge_thr.pickle"), "rb") as handle:
        imgnum_edge_thr_dict = pickle.load(handle)

    for _, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        img_num = row[sample_col]
        df_image = df_dataset[df_dataset[sample_col] == img_num]
        pid = df_image[pid_col].values[0]
        generate_graphs_using_points_codex_crc(
            df_image,
            imgnum_edge_thr_dict,
            img_num,
            pid,
            feature_cols=feature_cols,
            loc_x_col="Location_Center_X",
            loc_y_col="Location_Center_Y",
            pos=None,
            plot=plot,
            PLOT_PATH=PLOT_PATH,
            RAW_DATA_PATH=RAW_DATA_PATH,
        )


def data_processing_codex_crc_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, PLOT_PATH=None, RAW_DATA_PATH=None):
    if RAW_DATA_PATH is None:
        RAW_DATA_PATH = os.path.join(os.path.dirname(data_path), "raw")
    if PLOT_PATH is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_path))))
        PLOT_PATH = os.path.join(project_root, "plots", "codex_crc", "graph_voronoi_plots")

    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PLOT_PATH, exist_ok=True)

    print("Creating compact CRC dataset...")
    df_dataset, feature_cols = create_preprocessed_codex_crc_dataset(data_path, RAW_DATA_PATH=RAW_DATA_PATH)

    print("Calculating edge length distribution....")
    get_edge_length_dist(
        df_dataset,
        cell_count_thr=CELL_COUNT_THR,
        quant=0.975,
        sample_col="sample_id",
        loc_x="Location_Center_X",
        loc_y="Location_Center_Y",
        plot_dist=False,
        PLOT_PATH=PLOT_PATH,
        RAW_DATA_PATH=RAW_DATA_PATH,
    )

    print("Creating CRC graphs...")
    create_graphs_delauney_triangulation_codex_crc(
        df_dataset,
        feature_cols=feature_cols,
        cell_count_thr=CELL_COUNT_THR,
        sample_col="sample_id",
        pid_col="sample_id",
        plot=True,
        RAW_DATA_PATH=RAW_DATA_PATH,
        PLOT_PATH=PLOT_PATH,
    )
