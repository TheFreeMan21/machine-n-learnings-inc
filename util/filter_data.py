import pandas as pd
import numpy as np

def filtering(df_original_path, alpha=2, gamma=0.1, train=True):

    df=pd.read_csv(df_original_path)
    #Since the density is the continiuous variable, we can drop the Area column
    df.drop('Area', axis=1, inplace=True)
    #Rescale BonusMalus, since in the literature it is said to be between 0.5 and 3.5
    df['BonusMalus']=df['BonusMalus']/100
    
    if train:
        #Filtering the Exposure since it goes out of bounds (0-1)
        df = df[df['Exposure']<=1]
        #Filterig VehAge to be less than 25, since on the roads these are the most common vehicles
        df = df[(df['VehAge']<=25)]


        regions_remove_outliers = ['R25','R82','R54','R94','R93','R91','R52',
                                    'R72','R31','R73','R23','R22','R41','R42',
                                    'R83','R21','R74','R43', 'R11']

        def remove_outliers(group):
            # Only remove outliers for regions in the list
            if group.name in regions_remove_outliers:
                q1, q3 = group['Density'].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                return group[(group['Density'] >= lower) & (group['Density'] <= upper)]
            else:
                return group

        df = df.groupby('Region', group_keys=False).apply(remove_outliers)

        min_malus = 0.95 ** (df['DrivAge']-18)
        bad_mal_mask = df['BonusMalus'] >= min_malus
        age_avg = df[bad_mal_mask].groupby('DrivAge')['BonusMalus'].mean()
        impossible_malus_mask = df['BonusMalus']<0.95**(df['DrivAge']-18)
        df.loc[impossible_malus_mask, 'BonusMalus'] = df.loc[impossible_malus_mask, 'DrivAge'].map(age_avg)

    df['Risk'] = (np.log(1+(gamma+df['ClaimNb']**alpha)/(df['Exposure']))/(1+(np.log(1+(gamma+df['ClaimNb']**alpha)/(df['Exposure'])))))
    return df