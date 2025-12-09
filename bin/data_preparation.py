from numpy import integer
import pandas as pd
import os
import re

RAW_DATA_PATH = os.path.join("../data", "JacksonFischer")


def read_basel_patient_meta_data(RAW_DATA_PATH=RAW_DATA_PATH):
    """
    Create a pattern dictionary for mapping the patients, images and features

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Basel_PatientMetadata" and values
        are tuples in the form of (<the patterns to be caught>, <patient_id>)
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_PatientMetadata.csv")
    df_meta_file = pd.read_csv(f_path, index_col=False)
    
    # print(df_meta_file["FileName_FullStack"])
    # keys are "FileName_FullStack" values are patterns
    dict_basel_patterns = dict()
    for _, row in df_meta_file.iterrows():

        #get patient id
        p_id = row["PID"]
        grade = row["grade"]
        tumor_size = row["tumor_size"]
        age = row["age"]
        treatment = row["treatment"]
        diseasestage = row["DiseaseStage"]
        diseasestatus = row["diseasestatus"]
        clinical_type = row["clinical_type"]
        dfsmonth = row["DFSmonth"]
        osmonth = row["OSmonth"]
        patientstatus = row["Patientstatus"]
        # response = row["response"]

        # get the required fields from "FileName_FullStack" column
        fl_name = row["FileName_FullStack"]
        fields = fl_name.split("_")
        first_part = "_".join(fields[:7])

        # get the required fields from "core" column
        core = row["core"]
        core_X_ind = core.index("X")
        second_part = core[core_X_ind:]
        
        # create pattern
        pattern = re.compile(f"^({first_part})_[0-9]+_({second_part})_*")
        dict_basel_patterns[fl_name] = (pattern, p_id, grade, tumor_size, age, treatment, diseasestage, diseasestatus, clinical_type, dfsmonth, osmonth, patientstatus) # , response)
        

    return dict_basel_patterns

def read_zurich_patient_meta_data(RAW_DATA_PATH=RAW_DATA_PATH):
    """
    Create a pattern dictionary for mapping the patients, images and features

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Zuri_PatientMetadata" and values
        are tuples in the form of (<the patterns to be caught>, <patient_id>)
    """
    f_path = os.path.join(RAW_DATA_PATH, "Zuri_PatientMetadata.csv")
    df_meta_df = pd.read_csv(f_path, index_col=False)
    
    # print(df_meta_file["FileName_FullStack"])
    # keys are "FileName_FullStack" values are patterns
    dict_zurich_patterns = dict()
    for _, row in df_meta_df.iterrows():

        #get patient id
        p_id = row["PID"]
        grade = row["grade"]
        tumor_size = row["tumor_size"]
        age = row["age"]
        treatment = row["treatment"]
        diseasestage = row["DiseaseStage"]
        diseasestatus = row["diseasestatus"]
        clinical_type = row["clinical_type"]
        dfsmonth = row["DFSmonth"]
        osmonth = row["OSmonth"]
        patientstatus = row["Patientstatus"]
        
        # response = row["response"]


        # get the required fields from "FileName_FullStack" column
        fl_name = row["FileName_FullStack"]
        fields = fl_name.split("_")
        first_part = "_".join(fields[:7])
        

        # get the required fields from "core" column
        core = row["core"].split("slide")[1]
        # pattern = re.compile("[ABC]y[0-9]+x[0-9]+[_]*[0-9]*")
        core_X_ind = -1
        if "A" in core: core_X_ind = core.index("A")
        elif "B" in core: core_X_ind = core.index("B")
        elif "C" in core: core_X_ind = core.index("C")
        else: pass
        
        second_part = core[core_X_ind:]
        # "B[0-9]{2}\\.[0-9]+_
        # create pattern
        pattern = re.compile(f"^({first_part})_B[0-9][0-9]\.[0-9]+_({second_part})_*")
        dict_zurich_patterns[fl_name] = (pattern, p_id, grade, tumor_size, age, treatment, diseasestage, diseasestatus, clinical_type,  dfsmonth, osmonth, patientstatus)# , response)
        

    return dict_zurich_patterns


#  "Basel_Zuri_WholeImage file" to map the image number
def get_imgnumber_pid_flname_mapping_dict():
    """
    Create a dictionary for mapping file names, patient ids and img ids

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Zuri-Basel_PatientMetadata" and values
        are tuples in the form of (<file name from whole image flt>, <img number>), <patient_id>
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_WholeImage.csv")
    whole_image_df = pd.read_csv(f_path, index_col=False)
    # whole_image_cols = whole_image_df.columns

    img_numbers_to_be_ignored =[]
    
    dict_basel_patterns = read_basel_patient_meta_data()
    dict_zurich_patterns = read_zurich_patient_meta_data()

    dict_flname_imgnumber_pid = dict()
    dict_imgnumber_pid = dict()
    for _, row in whole_image_df.iterrows():
        # print(row)
        fl_name = row["FileName_FullStack"]
        img_number = row["ImageNumber"]
        pattern_dict = dict_basel_patterns if fl_name.startswith("Basel") else dict_zurich_patterns
        
        # get image ids for Zurich
        # if fl_name.startswith("ZTMA"):
        pattern_found = False
        for cohort_fl_name, pattern_pid in pattern_dict.items():
            pattern, p_id, grade, tumor_size, age, treatment, diseasestage, diseasestatus, clinical_type, dfsmonth, osmonth, patientstatus = pattern_pid
            is_match = pattern.findall(fl_name)
            # print(fl_name, is_match)
            if is_match!=[]:
                if pattern_found:
                    print("Something is weird! Pattern found more than once!")
                pattern_found = True
                dict_flname_imgnumber_pid[cohort_fl_name] = (fl_name, img_number,  p_id, grade, tumor_size, age, treatment, diseasestage, diseasestatus, clinical_type, dfsmonth, osmonth, patientstatus)
                dict_imgnumber_pid[img_number] = (fl_name, img_number,  p_id, grade, tumor_size, age, treatment, diseasestage, diseasestatus, clinical_type, dfsmonth, osmonth, patientstatus)
        
        ignore_warnings = True
        # print warning message if pattern cannot be found at all!    
        if not pattern_found:
            img_numbers_to_be_ignored.append(img_number)
            if not ignore_warnings: 
                print("No matching pattern! \n  Whole image fl_name:", fl_name)
            

    return dict_flname_imgnumber_pid, dict_imgnumber_pid, img_numbers_to_be_ignored

def get_basel_zurich_staining_panel():
    """
    Create a dictionary for fullstackids column to gene names

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FullStack" ids and values are gene names
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_StainingPanel.csv")
    staining_panel_df = pd.read_csv(f_path, index_col=False)

    # ids to be eliminated
    unwanted_ids = list(range(1,9)) + list(range(48, 50)) + [26, 32, 36, 42]
    dict_fullstackid_gene = dict()


    for _, row in staining_panel_df.iterrows():
        target = row["Target"]
        full_stack = row["FullStack"]
        if full_stack not in unwanted_ids:
            dict_fullstackid_gene[full_stack] = target

    return dict_fullstackid_gene


# TODO: read the single cell image file and get the required features and create the final dataset
def create_preprocessed_sc_feature_fl():
    """
    Perform mapping among files, select features, drop rows with NAs, remove the cells from unwanted images and save the compact file

    Parameters:
        None
    Returns:
        None  
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_SingleCell.csv")
    cols = pd.read_csv(f_path, index_col=False, nrows=0).columns.tolist()

    cols_to_be_selected = ["ImageNumber",  "ObjectNumber", "Location_Center_X", "Location_Center_Y"]

    df_ct_annotations = pd.read_csv(os.path.join(RAW_DATA_PATH, "Metacluster_annotations.csv"), sep=";")
    metacluster_to_ct_class_dict = dict()
    for ind, row in df_ct_annotations.iterrows():
        metacluster_to_ct_class_dict[row["Metacluster "]] = row["Cell type"], row["Class"] # create a tuple first index is cell type, second index is class
    
    
    # This part is added later to add cell type annotations / it is a bit redundant and can be refactored later
    # ./data/JacksonFischer/Basel_SC_locations.csv
    imgcell_to_id_dict = dict()
    sc_loc_dataset_fl_lst = ["Basel_SC_locations.csv", "Zurich_SC_locations.csv"]
    for fl in sc_loc_dataset_fl_lst:
        df_sc_locations = pd.read_csv(os.path.join(RAW_DATA_PATH, fl))
        for ind, row in df_sc_locations.iterrows():
            imgcell_to_id_dict[row["ImageNumber"], row["ObjectNumber"]] = row["id"] # create a tuple 0th index is cell type, 1st index is class 

    cellid_2_metacluster_dict = dict()
    sc_ct_annot_list = ["Basel_metaclusters.csv", "Zurich_matched_metaclusters.csv"]
    for fl in sc_ct_annot_list:
        df_sc_annot = pd.read_csv(os.path.join(RAW_DATA_PATH, fl))
        for ind, row in df_sc_annot.iterrows():
            cellid_2_metacluster_dict[row["id"]] = row["cluster"] # create a tuple 0th index is cell type, 1st index is class 
    
    # print(imgcell_to_id_dict.keys())
    # get channel ids to be kept
    dict_fullstackid_gene = get_basel_zurich_staining_panel()
    for col in cols:
         if col.startswith("Intensity_MeanIntensity_FullStack_c"):
             channel_id = int(col.split("Intensity_MeanIntensity_FullStack_c")[1])
             if channel_id in dict_fullstackid_gene.keys():
                 cols_to_be_selected.append(col)
                 

    # print(cols_to_be_selected)

    dict_flname_imgnumber_pid, dict_imgnumber_pid, img_numbers_to_be_ignored = get_imgnumber_pid_flname_mapping_dict()

    i=1
    chunks = []
    ch_size = 100000
    with  pd.read_csv(f_path, chunksize=ch_size, index_col=False) as reader:   
        for chunk in reader:
            print(f"Prossesing chunk #{i}... Chunk size: {ch_size}")
            chunks.append(chunk[cols_to_be_selected])

            # print("chunks", chunks)
            
            # remove the rows with cells from unwanted images
            chunks[-1]  = chunks[-1][~chunks[-1].ImageNumber.isin(img_numbers_to_be_ignored)]
            p_id_lst, grade_lst, tumor_size_lst, age_lst, treatment_lst, diseasestage_lst, diseasestatus_lst, clinical_type_lst, dfsmonth_lst, osmonth_lst, patientstatus_lst = [], [], [], [], [], [], [], [], [], [], [] 
            ct_list, class_list = [], []
            fl_name_lst, img_number_lst = [], []
            
            for _, row in chunks[-1].iterrows():
                fl_name, img_number,  p_id, grade, tumor_size, age, treatment, diseasestage, diseasestatus, clinical_type,  dfsmonth, osmonth, patientstatus = dict_imgnumber_pid[int(row["ImageNumber"])]
                
                p_id = str(p_id)+"-basel" if fl_name.startswith("Basel") else str(p_id)+"-zurich"

                cell_type, class_name = "Unknown", "Unknown"
                try:
                    id = imgcell_to_id_dict[int(row["ImageNumber"]), int(row["ObjectNumber"])]
                    meta_cluster_id = cellid_2_metacluster_dict[id]
                    cell_type, class_name = metacluster_to_ct_class_dict[meta_cluster_id]
                except:
                    pass

                p_id_lst.append(p_id)
                grade_lst.append(grade)
                tumor_size_lst.append(tumor_size)
                age_lst.append(age)
                treatment_lst.append(treatment)
                diseasestage_lst.append(diseasestage)
                diseasestatus_lst.append(diseasestatus)
                clinical_type_lst.append(clinical_type)
                dfsmonth_lst.append(dfsmonth)#  = row["DFSmonth"]
                osmonth_lst.append(osmonth)# = row["OSmonth"]
                patientstatus_lst.append(patientstatus)# = row["OSmonth"]
                fl_name_lst.append(fl_name)
                ct_list.append(cell_type)
                class_list.append(class_name)

            
            chunks[-1]["PID"] = p_id_lst
            chunks[-1]["grade"] = grade_lst
            chunks[-1]["tumor_size"] = tumor_size_lst
            chunks[-1]["age"] = age_lst
            chunks[-1]["treatment"] = treatment_lst
            chunks[-1]["DiseaseStage"] = diseasestage_lst
            chunks[-1]["diseasestatus"] = diseasestatus_lst
            chunks[-1]["clinical_type"] = clinical_type_lst
            chunks[-1]["DFSmonth"] = dfsmonth_lst
            chunks[-1]["OSmonth"] = osmonth_lst
            chunks[-1]["Patientstatus"] = patientstatus_lst
            chunks[-1]["fl_name"] = fl_name_lst
            chunks[-1]["cell_type"] = ct_list
            chunks[-1]["class"] = class_list
            
            i += 1
    
    # concatenate chunks and remove the rows with NA values
    df_new_dataset = pd.concat(chunks)
    df_new_dataset.dropna(subset = df_new_dataset.columns[:37], how="any",  inplace=True)
    df_new_dataset[['ObjectNumber', "ImageNumber"]] = df_new_dataset[['ObjectNumber', "ImageNumber"]].astype(integer, )
    print(df_new_dataset)
    print("Saving the compact dataset to ", os.path.join(RAW_DATA_PATH, "raw", "merged_preprocessed_dataset.csv"))
    df_new_dataset.to_csv(os.path.join(RAW_DATA_PATH, "raw", "merged_preprocessed_dataset.csv"), index=False)

# create_preprocessed_sc_feature_fl()

def generate_clinical_type(row):
        ER = row['ER Status']
        HER2 = row['HER2 Status']
        if ER == "Positive" and HER2 == "Positive":
            return "HR+HER2+"
        if ER == "Negative" and HER2 == "Positive":
            return "HR-HER2+"
        if ER == "Positive" and HER2 == "Negative":
            return "HR+HER2-"
        if ER == "Negative" and HER2 == "Negative":
            return "TripleNeg"
        # Check if HER2 is nan
        if HER2 != "Positive" and HER2 != "Negative":
            return "HER2-NAN"

def METABRIC_preprocess(visualize = False):
    # Reads single cell data and clinical data, merges them and saves the compact file
    # Clinical data: data/METABRIC/brca_metabric_clinical_data.tsv
    # Single cell data: data/METABRIC/single_cell_data.csv

    # Read clinical data
    clinical_data = pd.read_csv("../data/METABRIC/brca_metabric_clinical_data.tsv", sep="\t", index_col=False)
    single_cell_data = pd.read_csv("../data/METABRIC/single_cell_data.csv", index_col=False)

    # Merging the single_cell_data and clinical_data on Patient ID (metabricId)
    merged_data = pd.merge(single_cell_data, clinical_data, left_on='metabricId', right_on='Patient ID', how='inner')

    # Function to generate clinical_type based on ER and HER2 status
    

    # Applying the function to create the clinical_type column
    merged_data['clinical_type'] = merged_data.apply(generate_clinical_type, axis=1)

    # Columns to be renamed
    mean_ion_count_columns = [
        'HH3_total', 'CK19', 'CK8_18', 'Twist', 'CD68', 'CK14', 'SMA', 'Vimentin',
        'c_Myc', 'HER2', 'CD3', 'HH3_ph', 'Erk1_2', 'Slug', 'ER', 'PR', 'p53', 'CD44',
        'EpCAM', 'CD45', 'GATA3', 'CD20', 'Beta_catenin', 'CAIX', 'E_cadherin', 'Ki67',
        'EGFR', 'pS6', 'Sox9', 'vWF_CD31', 'pmTOR', 'CK7', 'panCK', 'c_PARP_c_Casp3',
        'DNA1', 'DNA2', 'H3K27me3', 'CK5', 'Fibronectin'
    ]

    # Renaming the columns
    renamed_columns = {col: f'Intensity_MeanIntensity_FullStackc_{i}' for i, col in enumerate(mean_ion_count_columns)}

    # Applying the renaming to the merged_data DataFrame
    merged_data.rename(columns=renamed_columns, inplace=True)

    # Rename the other columns
    columns_to_rename = {
        'Sample ID': 'PID',
        'Age at Diagnosis': 'age',
        'Neoplasm Histologic Grade': 'grade',
        'Tumor Size': 'tumor_size',
        'Chemotherapy': 'treatment',
        'Overall Survival (Months)': 'OSmonth',
        'Nottingham prognostic index': 'DiseaseStage',
        'Patient\'s Vital Status': 'diseasestatus',
        'Relapse Free Status (Months)': 'DFSmonth',
        'cell_type': 'description',

    }

    merged_data.rename(columns=columns_to_rename, inplace=True)

    # Save the merged dataframe to a new CSV file
    merged_data.to_csv('../data/METABRIC/raw/merged_preprocessed_dataset.csv', index=False)  

