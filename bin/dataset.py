import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch_geometric
import custom_tools
from torch_geometric.data import Data
from pandas.core.dtypes.missing import notna
from data_processing import get_cell_count_df
from data_processing import GRAPH_DIV_THR, CELL_COUNT_THR
from torch_geometric.data import InMemoryDataset, download_url
import pytorch_lightning as pl





S_PATH = os.path.realpath(__file__)


# Overwriting the S_PATH, it doesnt seem to work
S_PATH = os.path.abspath(os.path.dirname(__file__))


# S_PATH = os.path.dirname(__file__)

"""RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
PLOT_PATH = os.path.join(S_PATH, "../plots")"""



class TissueDataset(InMemoryDataset):
    def __init__(self, root, unit="month", wanted_label = "OSmonth", transform=None, pre_transform=None):
        """Creates the dataset for given "root" location.

        Args:
            root (str): _description_
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        custom_tools.set_seeds(42)
        self.wanted_label = wanted_label
        self.unit = unit
        super().__init__(root, transform, pre_transform)
        print(f"Target prediction: {wanted_label}")
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["merged_preprocessed_dataset.csv"]

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # self.data = pd.read_csv(os.path.join(self.root, "..", "raw", self.raw_file_names[0]))
        # print(self.data)
        # GRAPH_DIV_THR


        # img_pid_set = set([(item[0], item[1]) for item in self.data[["ImageNumber", "PID"]].values])
        data_list = []
        count = 0
        # pseudo_count=1.1
        for fl in os.listdir(os.path.join(self.root, "..", "raw")):
            if fl.endswith("features.pickle"):
                # print(fl)
                img, pid = fl.split("_")[:2]
                with open(os.path.join(self.root, "..", "raw", f'{img}_{pid}_clinical_info.pickle'), 'rb') as handle:
                    clinical_info_dict = pickle.load(handle)
                
                # the criteria to select the
                # if (clinical_info_dict["grade"]==3 or  clinical_info_dict["grade"]==2 or  clinical_info_dict["grade"]==1) and pd.notna(clinical_info_dict["OSmonth"]) and clinical_info_dict["diseasestatus"]=="tumor": # and pd.notna(clinical_info_dict["clinical_type"]): This was for Jackson Fisher
                if pd.notna(clinical_info_dict["OSmonth"]):# and clinical_info_dict["diseasestatus"]=="tumor": # and pd.notna(clinical_info_dict["clinical_type"]):
                # if (clinical_info_dict["grade"]==3 or  clinical_info_dict["grade"]==2) and pd.notna(clinical_info_dict["OSmonth"]) and clinical_info_dict["diseasestatus"]=="tumor" and pd.notna(clinical_info_dict["clinical_type"]):
                    # print(clinical_info_dict["grade"])
                    censored = None
                    
                    if "Patientstatus" in clinical_info_dict.keys():
                        censored = 0 if clinical_info_dict["Patientstatus"].lower().startswith("death") else 1
                    elif "Overall Survival Status" in clinical_info_dict.keys():
                        censored = 0 if "deceased" in clinical_info_dict["Overall Survival Status"].lower() else 1
                        # print(censored, clinical_info_dict["Overall Survival Status"].lower())

                    with open(os.path.join(self.root, "..", "raw", f'{img}_{pid}_features.pickle'), 'rb') as handle:
                        feature_arr = pickle.load(handle)
                        feature_arr = np.array(feature_arr)

                    with open(os.path.join(self.root, "..", "raw", f'{img}_{pid}_edge_index_length.pickle'), 'rb') as handle:
                        edge_index_arr, edge_length_arr = pickle.load(handle)
                        edge_index_arr = np.array(edge_index_arr)

                    with open(os.path.join(self.root, "..", "raw", f'{img}_{pid}_ct_class.pickle'), 'rb') as handle:
                        ct_class_arr = pickle.load(handle)
                        ct_class_arr = np.array(ct_class_arr)
                    
                    with open(os.path.join(self.root, "..", "raw", f'{img}_{pid}_coordinates.pickle'), 'rb') as handle:
                        coordinates_arr = pickle.load(handle)
                        coordinates_arr = np.array(coordinates_arr)
                        if self.wanted_label == "OSmonth":
                            # print("Label = OSMonth!")
                            y_val = clinical_info_dict["OSmonth"]
                            if self.unit == "month":
                                y_val = clinical_info_dict["OSmonth"]+1.1
                            elif self.unit == "week_lognorm":
                                y_val = np.log(clinical_info_dict["OSmonth"]*4+1.1)
                            elif self.unit == "week":
                                y_val = clinical_info_dict["OSmonth"]*4+1.1
                            
                            # print(feature_arr)
                            # print(clinical_info_dict)
                            data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=y_val, osmonth=clinical_info_dict["OSmonth"], clinical_type=str(clinical_info_dict["clinical_type"]), treatment=clinical_info_dict["treatment"], disease_stage=clinical_info_dict["DiseaseStage"],  tumor_grade=str(clinical_info_dict["grade"]), img_id=img, p_id=pid ,age=clinical_info_dict["age"], disease_status=clinical_info_dict["diseasestatus"], dfs_month=clinical_info_dict["DFSmonth"], is_censored=censored, ct_class = ct_class_arr) # ,age=clinical_info_dict["age"], disease_status=clinical_info_dict["disease_status"], dfs_month=clinical_info_dict["DFSmonth"]
                            # data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=np.log(clinical_info_dict["OSmonth"]*4+1.1), osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"], img_id=img, p_id=pid)
                            # data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=round(clinical_info_dict["OSmonth"]/12.0,3), osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"], img_id=img, p_id=pid)
                        else:
                            y_val = clinical_info_dict[self.wanted_label]
                            if self.wanted_label == "tumor_grade":
                                y_val = int(clinical_info_dict["Grade"]) - 1
                                data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=torch.tensor(y_val, dtype=torch.long), osmonth=clinical_info_dict["OS"], clinical_type=str(clinical_info_dict["Typ"]), disease_stage=clinical_info_dict["Stage"],  tumor_grade=clinical_info_dict["Grade"], img_id=img, p_id=pid ,age=clinical_info_dict["age"], disease_status=clinical_info_dict["diseasestatus"], dfs_month=clinical_info_dict["DFS"], ct_class = ct_class_arr)
                            else:
                                data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=torch.Tensor([y_val]), osmonth=clinical_info_dict["OS"], clinical_type=str(clinical_info_dict["Typ"]), disease_stage=clinical_info_dict["Stage"],  tumor_grade=clinical_info_dict["Grade"], img_id=img, p_id=pid ,age=clinical_info_dict["age"], disease_status=clinical_info_dict["diseasestatus"], dfs_month=clinical_info_dict["DFS"], ct_class = ct_class_arr) # ,age=clinical_info_dict["age"], disease_status=clinical_info_dict["disease_status"], dfs_month=clinical_info_dict["DFSmonth"]
                    
                    data_list.append(data)
                    count+=1

        print(f"Number of samples: {count}")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        print("data_list[0]", data_list[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        


class LungDataset(InMemoryDataset):
    def __init__(self, root, wanted_label = "Progression", train=True, transform=None, pre_transform=None):
        """Creates the dataset for given "root" location.

        Args:
            root (str): _description_
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        custom_tools.set_seeds(42)
        self.wanted_label = wanted_label
        self.train = train
        super().__init__(root, transform, pre_transform)
        print(f"Target prediction: {wanted_label}")
        print("processed_paths", self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["merged_preprocessed_dataset.csv"]

    @property
    def processed_file_names(self, train=True):
        if self.train:
            return [f'data_{self.wanted_label}_train.pt']
        else:
            return [f'data_{self.wanted_label}_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        data_list = []
        count = 0
        # pseudo_count=1.1
        for fl in os.listdir(os.path.join(self.root, "raw")):
            if fl.endswith("features.pickle"):
                # print(fl)
                img_pid = fl.split("_features.pickle")[0]
                
                with open(os.path.join(self.root,  "raw", f'{img_pid}_clinical_info.pickle'), 'rb') as handle:
                    clinical_info_dict = pickle.load(handle)
                
                if "Patientstatus" in clinical_info_dict.keys():
                    censored = 0 if clinical_info_dict["Patientstatus"].lower().startswith("death") else 1
                elif "Overall Survival Status" in clinical_info_dict.keys():
                    censored = 0 if "deceased" in clinical_info_dict["Overall Survival Status"].lower() else 1

                with open(os.path.join(self.root, "raw", f'{img_pid}_features.pickle'), 'rb') as handle:
                    feature_arr = pickle.load(handle)
                    feature_arr = np.array(feature_arr)

                with open(os.path.join(self.root, "raw", f'{img_pid}_edge_index_length.pickle'), 'rb') as handle:
                    edge_index_arr, edge_length_arr = pickle.load(handle)
                    edge_index_arr = np.array(edge_index_arr)

                with open(os.path.join(self.root, "raw", f'{img_pid}_ct_class.pickle'), 'rb') as handle:
                    ct_class_arr = pickle.load(handle)
                    ct_class_arr = np.array(ct_class_arr)
                
                with open(os.path.join(self.root, "raw", f'{img_pid}_coordinates.pickle'), 'rb') as handle:
                    coordinates_arr = pickle.load(handle)
                    coordinates_arr = np.array(coordinates_arr)
                # print("clinical_info_dict[self.wanted_label]", clinical_info_dict[self.wanted_label], torch.Tensor(clinical_info_dict[self.wanted_label]))

                if self.wanted_label == "tumor_grade":
                    y_val = int(clinical_info_dict["Grade"]) - 1
                    data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), 
                                edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), 
                                pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), 
                                y=torch.tensor(y_val, dtype=torch.long), 
                                osmonth=clinical_info_dict["OS"], 
                                clinical_type=str(clinical_info_dict["Typ"]), 
                                disease_stage=clinical_info_dict["Stage"],  
                                tumor_grade=clinical_info_dict["Grade"], 
                                img_id=img_pid, 
                                p_id=clinical_info_dict["Patient_Nr"], 
                                age=clinical_info_dict["Age"],  
                                dfs_month=clinical_info_dict["DFS"], 
                                ct_class = ct_class_arr)
                elif self.wanted_label == "Progression":
                    y_val = clinical_info_dict["Progression"]
                    if not pd.isna(y_val):
                        
                        # print("y_val", type(y_val))
                        #  "Sex","Age","BMI","Smoking Status","Pack Years","Stage","Progression","Death","Survival or loss (years)","Predominant histological pattern"
                        # print("clinical_info_dict", clinical_info_dict)
                        data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), 
                                    edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), 
                                    pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), 
                                    y=torch.tensor(y_val, dtype=torch.long), 
                                    osmonth=clinical_info_dict["Survival or loss (years)"], 
                                    sample_id= str(img_pid),
                                    img_id= str(img_pid),
                                    clinical_type=str(clinical_info_dict["Predominant histological pattern"]), 
                                    disease_stage=clinical_info_dict["Stage"])
                else:
                    y_val = clinical_info_dict[self.wanted_label]
                    # print("y_val", y_val, torch.Tensor([y_val]))
                    data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), 
                                edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), 
                                pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), 
                                y= torch.Tensor([y_val]).type(torch.LongTensor),
                                osmonth=clinical_info_dict["OS"], 
                                clinical_type=str(clinical_info_dict["Typ"]), 
                                disease_stage=clinical_info_dict["Stage"],  
                                tumor_grade=clinical_info_dict["Grade"], 
                                img_id=img_pid, 
                                p_id=clinical_info_dict["Patient_Nr"], 
                                age=clinical_info_dict["Age"],  
                                dfs_month=clinical_info_dict["DFS"], 
                                ct_class = ct_class_arr)
                if (self.train and not img_pid.startswith("LUAD_V")) or (not self.train and img_pid.startswith("LUAD_V")):
                    data_list.append(data)
                    count += 1
                else:
                    pass

        print(f"Number of samples: {count}")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Debug: print unique tumor_grade labels
        if self.wanted_label == "tumor_grade":
            labels = [data.y.item() for data in data_list]
            print("Unique tumor_grade labels:", set(labels))