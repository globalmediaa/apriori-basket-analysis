import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Veri hazırlama
df = pd.read_csv('./store_data.csv',names=['products'],header=None)

# Burada veriyi iki boyutlu diziye çeviriyoruz
data = list(df["products"].apply(lambda x:x.split(',')))

te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data, columns=te.columns_)
df1 = apriori(df, min_support=0.01, use_colnames=True)
df1.sort_values(by="support",ascending=False)
df1['length'] = df1['itemsets'].apply(lambda x:len(x))

product = input("Aranan Urun: ")
prod_key = 'has_' + product
df1[prod_key] = df1['itemsets'].apply(lambda x: (product in x))
print(df1[(df1['length'] > 1) & (df1['support'] >= 0.01) & df1[prod_key]].sort_values(by="support",ascending=False))
