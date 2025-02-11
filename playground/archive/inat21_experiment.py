import pandas as pd

import json

def main():
    with open("inat21_mini.json", "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['images'])
    random_sample_df = df.sample(n=2000, random_state=42) 
    random_sample_df['species'] = random_sample_df['file_name'].str.split("/").str[-2].str[6:].str.replace('_', ' ')
    random_sample_df.rename(columns = {'latitude' : 'Latitude', 
                                       'longitude' : 'Longitude'}, inplace = True)

    random_sample_df.to_csv('inat21_random2k.csv')


if __name__ == "__main__":
    main()
