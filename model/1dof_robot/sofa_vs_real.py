import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

def print_data(csv_path: str, label: str, is_sofa: bool = False):
	try:
		df = pd.read_csv(csv_path, header=None)
		# Two expected layouts (0-indexed):
		# Real: 0 pressure, 1 force, 2 radius_px, 3 curvature_per_px, 4 arc_length_px, ...
		# SOFA: 0 radius_px, 1 curvature_per_px, 2 arc_length_px, 3 x_tip, 4 y_tip, 5 x_base, 6 y_base
		if is_sofa:
			if df.shape[1] < 3:
				print(f"Skipping '{label}' (unexpected SOFA format: fewer than 3 columns)")
				return
			radius_mm = df.iloc[:, 0] * conv_rate
			arc_length_mm = df.iloc[:, 2] * conv_rate
			out = pd.DataFrame({
				'arc_length_mm': arc_length_mm.round(3),
				'radius_mm': radius_mm.round(3),
			})
		else:
			if df.shape[1] < 5:
				print(f"Skipping '{label}' (unexpected REAL format: fewer than 5 columns)")
				return
			pressure = df.iloc[:, 0]
			radius_mm = df.iloc[:, 2] * conv_rate
			arc_length_mm = df.iloc[:, 4] * conv_rate
			out = pd.DataFrame({
				'pressure': pressure,
				'arc_length_mm': arc_length_mm.round(3),
				'radius_mm': radius_mm.round(3),
			})

		print(f"\n{label}")
		print(out.to_string(index=False))
	except FileNotFoundError:
		print(f"File not found for '{label}': {csv_path}")
	except Exception as e:
		print(f"Failed to print table for '{label}': {e}")

def max_norm_arc_var_pct(arc_len_px_series):
	if len(arc_len_px_series) == 0:
		return 0.0
	arc = arc_len_px_series.astype(float) * conv_rate
	max_v = arc.max()
	min_v = arc.min()
	if max_v == 0:
		return 0.0
	return (float(max_v - min_v) / float(max_v)) * 100.0

def print_free_motion_combined():
	real_path = 'data/L_10cm_OD_1.5mm/characterization_and_scale_force/free_motion/cv_output.csv'
	sofa_path = 'data/L_10cm_OD_1.5mm/characterization_and_scale_force/sofa_free_motion/cv_output.csv'

	df_real = pd.read_csv(real_path, header=None)
	df_sofa = pd.read_csv(sofa_path, header=None)

	max_len = max(len(df_real), len(df_sofa))
	idx = range(max_len)

	pressure = pd.Series(df_real.iloc[:, 0].astype(float).values, index=range(len(df_real))).reindex(idx) / 10
	radius_mm_real = pd.Series((df_real.iloc[:, 2] * conv_rate).round(3).values, index=range(len(df_real))).reindex(idx)
	radius_mm_sofa = pd.Series((df_sofa.iloc[:, 0] * conv_rate).round(3).values, index=range(len(df_sofa))).reindex(idx)

	table = pd.DataFrame({
		'p [MPa]': pressure,
		'r [mm]': radius_mm_real,
		'rs [mm]': radius_mm_sofa,
	})

	print("\nFree motion")
	print(table.to_string(index=False))

	var_real = max_norm_arc_var_pct(df_real.iloc[:, 4])
	var_sofa = max_norm_arc_var_pct(df_sofa.iloc[:, 2])
	print(f"Max normalized arc length variation (free motion, real): {var_real:.2f}%")
	print(f"Max normalized arc length variation (free motion, SOFA): {var_sofa:.2f}%")

def print_force_small_bending_combined():
	real_path = 'data/L_10cm_OD_1.5mm/characterization_and_scale_force/force_small_bending/cv_output.csv'
	sofa_path = 'data/L_10cm_OD_1.5mm/characterization_and_scale_force/sofa_force_small_bending/cv_output.csv'

	df_real = pd.read_csv(real_path, header=None)
	df_sofa = pd.read_csv(sofa_path, header=None)

	max_len = max(len(df_real), len(df_sofa))
	idx = range(max_len)

	pressure = pd.Series(df_real.iloc[:, 0].astype(float).values, index=range(len(df_real))).reindex(idx) / 10
	radius_mm_real = pd.Series((df_real.iloc[:, 2] * conv_rate).round(3).values, index=range(len(df_real))).reindex(idx)
	force_real = pd.Series(df_real.iloc[:, 1].astype(float).values, index=range(len(df_real))).reindex(idx) * 10

	radius_mm_sofa = pd.Series((df_sofa.iloc[:, 0] * conv_rate).round(3).values, index=range(len(df_sofa))).reindex(idx)
	# SOFA force is provided in the last column of the CSV
	force_sofa = pd.Series(df_sofa.iloc[:, -1].astype(float).values, index=range(len(df_sofa))).reindex(idx)

	table = pd.DataFrame({
		'p [MPa]': pressure,
		'r [mm]': radius_mm_real,
		'rs [mm]': radius_mm_sofa,
		'F [N]': force_real,
		'Fs [N]': force_sofa,
	})

	print("\nForce small bending")
	print(table.to_string(index=False))

	var_real = max_norm_arc_var_pct(df_real.iloc[:, 4])
	var_sofa = max_norm_arc_var_pct(df_sofa.iloc[:, 2])
	print(f"Max normalized arc length variation (force small bending, real): {var_real:.2f}%")
	print(f"Max normalized arc length variation (force small bending, SOFA): {var_sofa:.2f}%")


conv_rate = 3*10**-2 # [mm/pixel]

# Free motion
df_free_motion = pd.read_csv('data/L_10cm_OD_1.5mm/characterization_and_scale_force/free_motion/cv_output.csv')
df_sofa_free_motion = pd.read_csv('data/L_10cm_OD_1.5mm/characterization_and_scale_force/sofa_free_motion/cv_output.csv')
variable_free_motion = df_free_motion.iloc[:, 2]  # third column of free_motion
variable_sofa_free_motion = df_sofa_free_motion.iloc[:, 0]  # first column of sofa_free_motion
rmse_free_motion = sqrt(mean_squared_error(variable_free_motion, variable_sofa_free_motion))
print_free_motion_combined()
print(f"RMSE between free motion and sofa free motion: {1/(rmse_free_motion*conv_rate)} 1/mm")

# Force small bending
df_force_small_bending = pd.read_csv('data/L_10cm_OD_1.5mm/characterization_and_scale_force/force_small_bending/cv_output.csv')
df_sofa_force_small_bending = pd.read_csv('data/L_10cm_OD_1.5mm/characterization_and_scale_force/sofa_force_small_bending/cv_output.csv')
variable_force_small_bending = df_force_small_bending.iloc[:, 2]  # third column of force_small_bending
variable_sofa_force_small_bending = df_sofa_force_small_bending.iloc[:, 0]  # first column of sofa_force_small_bending
rmse_force_small_bending = sqrt(mean_squared_error(variable_force_small_bending, variable_sofa_force_small_bending))
print_force_small_bending_combined()
print(f"RMSE (curvature) between force small bending and sofa force small bending: {1/(rmse_force_small_bending*conv_rate)} 1/mm")

# Compute RMSE between force columns (real vs SOFA) over overlapping range
force_real_all = df_force_small_bending.iloc[:, 1].astype(float) * 10  # match table scaling
force_sofa_all = df_sofa_force_small_bending.iloc[:, -1].astype(float)
n_common = min(len(force_real_all), len(force_sofa_all))
if n_common > 0:
	rmse_force_cols = sqrt(mean_squared_error(
		force_real_all.iloc[:n_common],
		force_sofa_all.iloc[:n_common]
	))
	# Printed in mN to match label
	print(f"RMSE (force) between force small bending and sofa force small bending: {rmse_force_cols:.3f} mN")
else:
	print(f"RMSE (force) between force small bending and sofa force small bending: N/A (no overlap)")

