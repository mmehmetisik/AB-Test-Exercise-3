
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################
# Uygulama 3: # Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ortalaması Arasında İstatistiki Olarak Anlamlı
# Farklılık var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

# Outcome = 0 diyabet hastası değil
# Outcome = 1 diyabet hastası
df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2  Diyabet Hastası Olan ve Olmayanların Yaşları Ortalamaları Arasında İstatistiki Olarak Anlamlı Farklılık Yoktur.
# H1: M1 != M2 Diyabet Hastası Olan ve Olmayanların Yaşları Ortalamaları Arasında İstatistiki Olarak Anlamlı Farklılık vardır.


# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value değeri 0.05 den küçük olduğu için H0 hipotezi reddedilir.
# Normallik varsayımı sağlanmamaktadır.
# Normallik varsayımı sağlanmadığı için nonparametrik test uygulanır. mannwhitneyu testi uygulanır.
# Normallik varsayımı sağlanmadığı için, Varyans homojenliği testini yapmamıza gerek yoktur.

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#H0 hipotezi reddedilir.
#Diyabet Hastası Olan ve Olmayanların Yaşları Ortalamaları Arasında İstatistiki Olarak Anlamlı Farklılık Yoktur Hipotezi reddedilir.
#Diyabet Hastası Olan ve Olmayanların Yaşları Ortalamaları Arasında İstatistiki Olarak Anlamlı Farklılık vardır.

#Yani yaşı yüksek olanlar diyabet hastalığına sahiptir.