import argparse
import logging
from src.utils.dataframe import prepare_vindr_dataframe
import pandas as pd
import os
from src.utils.dataframe import save_df_to_npy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vindr Mammo data preparation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    args = parser.parse_args()
    parser.set_defaults(synthetize=False)
    class_list = ['mass', 'suspicious_calcification', 'suspicious_lymph_node']

    os.makedirs(args.out_dir, exist_ok=True)

    logging_message = "[AROB-2025-KAPTIOS-VINDR]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    logging.info('Running Vindr Mammo dataset preparation')
    df_find = pd.read_csv(os.path.join(
        args.data_dir, 'finding_annotations.csv'))

    df = prepare_vindr_dataframe(df_find, class_list)
    save_df_to_npy(df, class_list, args.data_dir, args.out_dir)
