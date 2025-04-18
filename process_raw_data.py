"read data from csv file"
"sample data in a certain percentage: T, X, Y and u"
"the generated data file name with % sign means the percentage of the original data"

import os
import numpy as np
import pandas as pd

def read_data(t):
    data = pd.read_csv(f"./Data/heat_eqaution_0.1_100%.csv")
    u_t = [col for col in data.columns if col == "u (1) @ t="+str(t)]
    X = data['X'].values
    Y = data['Y'].values
    u = data[u_t].values
    u = u.squeeze()
    # make plot

    import matplotlib.pyplot as plt
    plt.tricontour(X, Y, u, 15, cmap='jet')
    plt.tricontourf(X, Y, u, 15, cmap='jet')
    cbar=plt.colorbar()
    plt.plot(X, Y, 'ko', ms=3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.savefig(f"heat_equation_t={t}.png")
    plt.show()
    plt.close()
    return X,Y,u
    
def sample_data(file_path, percentage):
    data = pd.read_csv(file_path)
    sample_data = data.sample(frac=percentage)
    sample_data.to_csv("./Data/"+os.path.basename(file_path).replace('.csv','')+'_'+str(percentage)+'00%.csv', index=False)
    return sample_data

def reshape_data(file_path, dt): # change the data shape to trainable format
    data = pd.read_csv(file_path)
    reshaped_data = pd.DataFrame()
    for i in range(data.shape[1]-2):
        column_data = {
        'X': data.iloc[:, 0],
        'Y': data.iloc[:, 1],
        'T': np.array([dt * i] * data.shape[0]),
        'U': data.iloc[:, i + 2]
        }
        reshaped_column = pd.DataFrame(column_data)
        reshaped_data = pd.concat([reshaped_data, reshaped_column], axis=0)
    reshaped_data.to_csv("./Data/"+os.path.basename(file_path).replace('.csv','')+'_training.csv', index=False)

    return reshaped_data

def reshape_data_uv(file_path, dt): # change the data shape to trainable format
    data = pd.read_csv(file_path)
    reshaped_data = pd.DataFrame()
    for i in range((data.shape[1]-2)//2):
        column_data = {
        'X': data.iloc[:, 0],
        'Y': data.iloc[:, 1],
        'T': np.array([dt * i] * data.shape[0]),
        'U': data.iloc[:, 2*i + 2],
        'V': data.iloc[:, 2*i + 3]
        }
        reshaped_column = pd.DataFrame(column_data)
        reshaped_data = pd.concat([reshaped_data, reshaped_column], axis=0)
    reshaped_data.to_csv("./Data/"+os.path.basename(file_path).replace('.csv','')+'_training.csv', index=False)

    return reshaped_data

if __name__ == '__main__':
    "use guide"
    "1. use sample_data function to smaple data with a specific percentage (data is from comsol generated data)"
    "2. use reshape_data function to reshape the data to trainable format"
    "3. the data can be checcked by read_data function"
    # read_data(t=1.9)
    sample_data(file_path="./Data/Diffusion_reaction_update.csv", percentage=0.2)   # using csv from comsol generated data
    #reshape_data(file_path="./Data/heat_equation_flux0.1_time0.5_100%.csv", dt=0.05)  # using csv from sample_data function
    #reshape_data_uv(file_path="./Data/Diffusion_reaction_update_20%.csv", dt=0.05)  # using csv from sample_data function