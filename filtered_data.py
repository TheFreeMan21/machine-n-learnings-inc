import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('src\claims_train.csv')

df.drop('Area', axis=1, inplace=True)

df['BonusMalus'] = df['BonusMalus'] / 100

# Remove drivers signed up to insurance for over a year
year_minus_mask = df['Exposure'] < 1
df = df[year_minus_mask]

# Remove vehicles whose age is 25 or older (vintage cars)
non_vintage_mask = df['VehAge'] <= 25
df = df[non_vintage_mask]

df['Density'] =df['Density']/max(df['Density'])

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

#possible_risks: claimNB/Exposure -- simple misses out important factors
#                log(1 + (claimNB**1.3/(Exposure+beta)))  beta if we find the data too noise
alpha=10 # If we want to penaltize the claimnb more
beta=0 # If we want to finetune the exposure part 
df['Risk'] = (np.log(1+(df['ClaimNb']**alpha)/(df['Exposure']+beta))/(1+(np.log(1+(df['ClaimNb']**alpha)/(df['Exposure']+beta)))))
plt.hist(df['Risk'],50)
plt.yscale('log')
plt.show()
# plt.hist(df['ClaimNb'],10)
# plt.yscale('log')
# plt.show()