"""
This script generates a sample dataset for the image regression task.
Specifically we set up the problem of estimating the 2D vector that corresponds
to an arrow drawn on an image. In this script we sythentically generate the images
and write the corresponding targets to a csv file.
"""

from typing import Tuple
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from JAM_model import  Jam2img
from utils import  generate_two_UAV_positions

def synthesise_dataset(
        subfolder_name: str,
        num_images: int,
        image_size: Tuple[int, int],
        rng_seed: int,
        index
) -> pd.DataFrame:
    """
    Synthesise a dataset of images and targets. The images are saved to the subdirectory
    and the targets are saved to a csv file. The function returns the dataframe containing
    the targets.
    """
    # Set the random seed
    # np.random.seed(rng_seed)
    random.seed(rng_seed)
    # Create the dataframe
    df = pd.DataFrame(columns=['image_path', 'x', 'y','a', 'b'])

    # Create the images and targets
    for i in range(num_images):
        # Generate the uav
        # uav1 = (random.randint(200, 800)  , random.randint(200, 800))
        # uav2= (random.randint(200, 800)  , random.randint(200, 800))
        Radar = np.array([200, 200])
        uav1, uav2 = generate_two_UAV_positions(Radar)
        print(i,uav1,uav2)
        a=i+index
        # Save the image
        dataset_filename = f'JAM_data/{subfolder_name}/images/image_{a}.png'
        Path(dataset_filename).parent.mkdir(parents=True, exist_ok=True)
        Jam2img(uav1,uav2,dataset_filename)
        # Add the target to the dataframe
        df.loc[i] = [dataset_filename, uav1[0], uav1[1], uav2[0], uav2[1]]

    # Save the dataframe
    df.to_csv(f'JAM_data/{subfolder_name}_{index}.csv', index=False)
    # df.to_csv(f'JAM_data/{subfolder_name}.csv', index=False)   
    # df.to_csv(f'JAM_data/train.csv', index=False)
    return df


if __name__ == '__main__':
    # synthesise_dataset('train', 2000, (400, 400), 4,5500)
    synthesise_dataset('test', 1000, (400, 400), 43,0)
