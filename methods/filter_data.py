import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def filtering(df, alpha=1, gamma=0.1):

    # add risk score column
    df["Risk"] = np.log(1 + (gamma + (df["ClaimNb"]**alpha) / df["Exposure"])) / (1 + np.log(1 + (gamma + (df["ClaimNb"]**alpha) / df["Exposure"])))
    

    df.drop('Area', axis=1, inplace=True)

    df['BonusMalus'] = df['BonusMalus'] / 100

    # Remove drivers signed up to insurance for over a year
    year_minus_mask = df['Exposure'] < 1
    df = df[year_minus_mask]

    # Remove vehicles whose age is 25 or older (vintage cars)
    non_vintage_mask = df['VehAge'] <= 25
    df = df[non_vintage_mask]
    df['Density_scaled'] = StandardScaler().fit_transform(df[['Density']])

    #Let's fix the BonusMalus, since only those could have 0.5 who had 13 years of accident free driving. that means until the age of 31
    #nobody can have malus 0.5 however there is no limit for the top value (The overall top limit is 3.50, bottom limit 0.5)
    #BonusMalus= PreviousMalus * 0.95 if no accident else 1.25
    #We replaced the incorrect values with the average value of the age
    min_malus = 0.95 ** (df['DrivAge']-18)
    bad_mal_mask = df['BonusMalus'] >= min_malus
    age_avg = df[bad_mal_mask].groupby('DrivAge')['BonusMalus'].mean()
    impossible_malus_mask = df['BonusMalus']<0.95**(df['DrivAge']-18)
    df.loc[impossible_malus_mask, 'BonusMalus'] = df.loc[impossible_malus_mask, 'DrivAge'].map(age_avg)

    # df.drop(labels=[df[mask]],inplace=True)#The data is from 2004-2005 there is low chance of having a car which is from before 1940
    #for x in ['IDpol','ClaimNb','Exposure','VehPower','VehAge','DrivAge','BonusMalus',]:
    #    print(df[x].describe())
    #    plt.boxplot(df[x])
    #    plt.text(0,0,x)
    #    plt.show()
    #print(df['IDpol'].is_unique)

    plt.scatter(df['DrivAge'],df['BonusMalus'])
    plt.show()
    print(df.shape)

    #'IDpol' unique we dont have to drop duplicates
    #'DrivAge' we should drop the > (80 or 90) entries

    # possible_risks: claimNB/Exposure -- simple misses out important factors
    #                log(1 + (claimNB**1.3/(Exposure+beta)))  beta if we find the data too noise
    # alpha=10 # If we want to penaltize the claimnb more
    # beta=0 # If we want to finetune the exposure part 
    # df['Risk'] = (np.log(1+(df['ClaimNb']**alpha)/(df['Exposure']+beta))/(1+(np.log(1+(df['ClaimNb']**alpha)/(df['Exposure']+beta)))))
    # plt.hist(df['Risk'],50)
    # plt.yscale('log')
    # plt.show()
    # plt.hist(df['ClaimNb'],10)
    # plt.yscale('log')
    # plt.show()

    regions_remove_outliers = ['R25','R82','R54','R94','R93','R91','R52',
                            'R72','R31','R73','R23','R22','R41','R42',
                            'R83','R21','R74','R43', 'R11']

    def remove_outliers_selective(group):
        # Only remove outliers for regions in the list
        if group.name in regions_remove_outliers:
            q1, q3 = group['Density'].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return group[(group['Density'] >= lower) & (group['Density'] <= upper)]
        else:
            return group

    df = df.groupby('Region', group_keys=False).apply(remove_outliers_selective)

    return df
