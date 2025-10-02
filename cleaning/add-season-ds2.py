import pandas as pd
import os

cwd = os.getcwd()
ds2_path = 'datasets/cleaned/cleaned_dataset2.csv'
df = pd.read_csv(os.path.join(cwd, ds2_path))

# Add a 'season' feature to dataset
df['season'] = df['month'] // 3

output_path = 'datasets/cleaned/cleaned_dataset2_seasons.csv'
df.to_csv(os.path.join(cwd, output_path), index=False)

# Remove 3rd season due to lack of data (12 records == ~0.57% )
# df = df[df['month'] < 6] # Do in analysis file because may be used in other analysis.
