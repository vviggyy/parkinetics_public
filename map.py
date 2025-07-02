'''
Maps clinical scores associated with each patient identifier to rows in an output file 
of the pipeline.py (i.e. merges clinical scores with metrics files)

'''

import os
import numpy as np 
import pandas as pd
import argparse


def parse_args():
    """Handles command-line arguments.

    Returns:
        args (a parser Namespace): Stores command line attributes.
    """
    parser = argparse.ArgumentParser(description="handles arguments for mapping")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to signal metrics file (.csv)",
    )
    parser.add_argument(
        "-c",
        "--clinical",
        type=str,
        help="Path to clinical metrics file (.csv)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="Path to output file (use proper naming conventions)",
        default="out/mapping_output.csv"
    )
    parser.add_argument(
        "-a",
        "--all",
        type=bool,
        help="Whether you want to add all clinical metrics to signal metrics, or just calculated ones (e.g. hand score, Total UPDRS)",
        default=False
    )
    
    return parser.parse_args()

def calculate_clinical_scores(clin_df: pd.DataFrame) -> pd.DataFrame:
    
    clin_df2 = pd.DataFrame()
    clin_df2["participant"] = clin_df["participant"]
    tot_1_17 = clin_df.loc[:, clin_df.columns[clin_df.columns.get_loc("1_1_intellectual_impairment"): clin_df.columns.get_loc("2_17_sensory_complaints")+1]]
    clin_df2["total_score_1_17"] = tot_1_17.sum(axis=1) #total score 1-17

    print(tot_1_17)

    tot_18_31 = clin_df.loc[:, clin_df.columns[clin_df.columns.get_loc("3_18_speech"): clin_df.columns.get_loc("3_31_bbh")+1]]
    clin_df2["total_score_18_31"] = tot_18_31.sum(axis=1) #total score 18-31

    clin_df2["total_score"] = clin_df2["total_score_1_17"] + clin_df2["total_score_18_31"]

    tot_func = clin_df.loc[:, clin_df.columns[clin_df.columns.get_loc("2_8_handwriting"): clin_df.columns.get_loc("2_11_hygiene")+1]]
    clin_df2["total_functional_score"] = tot_func.sum(axis = 1)

    T_score = clin_df["2_16_tremor"] + clin_df.loc[:, clin_df.columns[clin_df.columns.get_loc("3_20_tremor_head"): clin_df.columns.get_loc("3_21_APT_left_arm")]].sum(axis = 1)
    AKT_score = clin_df.loc[:, clin_df.columns[clin_df.columns.get_loc("3_22_rigidity_neck"): clin_df.columns.get_loc("3_26_la_left")]].sum(axis = 1)
    clin_df2["tremor_score"] = T_score
    clin_df2["AKT_score"] = AKT_score
    dom = []
    for i in range(0, len(T_score)):
        if (T_score[i] / 8) >= 1.5 * AKT_score[i] / 12:
            dom.append("T")
        elif ( AKT_score[i] / 12) >= 1.5 * T_score[i] / 8:
            dom.append("A")
        else:
            dom.append("M")
    clin_df2["Dominance"] = dom
    used_arm_score = 0
    used_arm = []
    for index, row in clin_df.iterrows():
        if (row["hand_used"] == "R"):
            used_arm_score = row["3_20_tremor_right_arm"] + row["3_21_APT_right_arm"] + row["3_22_rigidity_right_arm"] + row["3_23_ft_right"] + row["3_24_hm_right"] + row["3_25_rahm_right"]
        else:
            used_arm_score = row["3_20_tremor_left_arm"] + row["3_21_APT_left_arm"] + row["3_22_rigidity_left_arm"] + row["3_23_ft_left"] + row["3_24_hm_left"] + row["3_25_rahm_left"]
        used_arm.append(used_arm_score)
        used_arm_score = 0
    clin_df2['score_hand_used'] = used_arm    
    return clin_df2
    
def main():
    """Main control flow.
    """
    args = parse_args() #get command-line attributes
    
    #read in both feature and clinical metrics...
    features = pd.read_csv(args.input) 
    if "xlsx" in args.clinical:
        df = pd.read_excel(args.clinical)
        name = os.path.splitext(args.clinical)[0] + ".csv"
        df.to_csv(name, index=False)
        clinical = pd.read_csv(name)
    else:
        clinical = pd.read_csv(args.clinical)

    clinical.fillna(0)
    calced_scores = calculate_clinical_scores(clin_df=clinical) #calculate further clinical scores (e.g. hand score, total UPDRS)
    
    if args.all: #merge both
        print("Merging both raw clinical subscores and calculated total scores...") #use console log, instead...
        merged_raw = features.merge(clinical, left_on="participant", right_on="participant", how="inner" )
        merged_total = merged_raw.merge(calced_scores, left_on="participant", right_on="participant", how="inner")
        #NOTE
        #metrics - "left" matrix
        #mapping - "right" matrix
        #left_on - "name of the column of the left matrix that has the mapping id
        #right_on = "name of the column of the right matrix that has the mapping id
        ## NOTE: if both the same name, use "on = <arg>"
        # how = inner --> only keep left matrix columns if there isn't a match (i.e if patients go up to less than 195), other optinos
    else: #merge just calc'ed scores
        print("Merging just further calculated scores...")
        merged_total = features.merge(calced_scores, left_on="participant", right_on = "participant", how="inner")
    
    merged_total.to_csv(rf"out/mappings/{args.out}" + ".csv")

if __name__ == "__main__":
    main()
    