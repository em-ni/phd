import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the CSV files
df_free_motion = pd.read_csv('data/free_motion/cv_output.csv')
df_force_small_bending = pd.read_csv('data/force_small_bending/cv_output.csv')
df_sofa_free_motion = pd.read_csv('data/sofa_free_motion/cv_output.csv')
df_sofa_force_small_bending = pd.read_csv('data/sofa_force_small_bending/cv_output.csv')

# Extract the required columns
variable_free_motion = df_free_motion.iloc[:, 2]  # third column of free_motion
variable_force_small_bending = df_force_small_bending.iloc[:, 2]  # third column of force_small_bending
variable_sofa_free_motion = df_sofa_free_motion.iloc[:, 0]  # first column of sofa_free_motion
variable_sofa_force_small_bending = df_sofa_force_small_bending.iloc[:, 0]  # first column of sofa_force_small_bending

# Compute RMSE between corresponding variables
rmse_free_motion = sqrt(mean_squared_error(variable_free_motion, variable_sofa_free_motion))
rmse_force_small_bending = sqrt(mean_squared_error(variable_force_small_bending, variable_sofa_force_small_bending))

# Convert from pixels to mm
conv_rate = 3*10**-2 # [mm/pixel]

# Print the RMSE values
print(f"RMSE between free motion and sofa free motion: {1/(rmse_free_motion*conv_rate)} 1/mm")
print(f"RMSE between force small bending and sofa force small bending: {1/(rmse_force_small_bending*conv_rate)} 1/mm")
