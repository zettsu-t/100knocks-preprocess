#!/usr/bin/python3
# coding: utf-8

## データサイエンス100本ノック (構造化データ加工編) Python版の私の解答 by プログラマたんbot
## 丸一日かけて、100-6本打ってみました。
## まだ答え合わせをしていないので、解答は間違っている可能性があります。
## ★ P-070..P-074 と P-090 は後で

import os
import re
from geopy.distance import geodesic
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

## この問題で使用するデータを読み込む
## 公式解答集ではDBから読み込んでいるが、ここではCSVファイルを読み込む。
## 型を正確に推定するために、low_memory=Falseにする

df_receipt = pd.read_csv('../data/receipt.csv', low_memory=False)
df_store = pd.read_csv('../data/store.csv', low_memory=False)
df_customer = pd.read_csv('../data/customer.csv', low_memory=False)
df_product = pd.read_csv('../data/product.csv', low_memory=False)
df_category = pd.read_csv('../data/category.csv', low_memory=False)
## なぜか列名に空白が入っている
df_geocode = pd.read_csv('../data/geocode.csv', low_memory=False).rename(columns={' latitude': 'latitude'})

## P-001
df_receipt.loc[0:9,]

## P-002
df_receipt.loc[0:9, ['sales_ymd', 'customer_id', 'product_cd', 'amount']]

## P-003
## inplace=Trueにしない限り元のオブジェクトを置き換えないので、
## 変更後のオブジェクトを返す
df_receipt.loc[0:9, ['sales_ymd', 'customer_id', 'product_cd', 'amount']].rename(columns={'sales_ymd': 'sales_date'})
df_receipt[0:9, ['sales_ymd', 'customer_id', 'product_cd', 'amount']].rename(columns={'sales_ymd': 'sales_date'})

## P-004
df_receipt.loc[df_receipt.customer_id == 'CS018205000001', ['sales_ymd', 'customer_id', 'product_cd', 'amount']]

## P-005
df_receipt.loc[(df_receipt.customer_id == 'CS018205000001') & (df_receipt.amount >= 1000),
               ['sales_ymd', 'customer_id', 'product_cd', 'amount']]

## P-006
df_receipt.loc[(df_receipt.customer_id == 'CS018205000001') & ((df_receipt.amount >= 1000) | (df_receipt.quantity >= 5)),
               ['sales_ymd', 'customer_id', 'product_cd', 'amount']]

## P-007
df_receipt.loc[(df_receipt.customer_id == 'CS018205000001') & (df_receipt.amount >= 1000) & (df_receipt.amount <= 2000),
               ['sales_ymd', 'customer_id', 'product_cd', 'amount']]

## P-008
df_receipt.loc[(df_receipt.customer_id == 'CS018205000001') & (df_receipt.product_cd != 'P071401019'),
               ['sales_ymd', 'customer_id', 'product_cd', 'amount']]

## P-009
df_expected = df_store.query('not(prefecture_cd == "13" | floor_area > 900)')
df_actual = df_store.query('prefecture_cd != "13" & floor_area <= 900')
assert df_expected.equals(df_actual)

## P-010
df_store[df_store['store_cd'].apply(lambda x: x.startswith('S14'))].head(n=10)

## P-011
df_customer[df_customer['customer_id'].apply(lambda x: x.endswith('1'))].head(n=10)

## P-012
df_store[df_store['address'].apply(lambda x: x.find('横浜市') >= 0)].head(n=10)

## P-013
df_customer[df_customer['status_cd'].apply(lambda x: re.match(r"^[A-F]", x) is not None)].head(n=10)

## P-014
## stringr::str_detectはpartial matching、reはfull matching
df_customer[df_customer['status_cd'].apply(lambda x: re.match(r".*[1-9]$", x) is not None)].head(n=10)

## P-015
df_customer[df_customer['status_cd'].apply(lambda x: re.match(r"^[A-F].*[1-9]$", x) is not None)].head(n=10)

## P-016
## Pythonのr正規表現リテラルは、\を二重にしない
df_store[df_store['tel_no'].apply(lambda x: re.match(r"^\d{3}\-\d{3}\-\d{4}$", x) is not None)]

## P-017
## 元はstrなので日時に変換する
df_customer['birth_day'] = pd.to_datetime(df_customer['birth_day'], format='%Y-%m-%d')
df_customer.sort_values('birth_day').head(n=10)

## P-018
df_customer.sort_values(by='birth_day', ascending=False).head(n=10)

## P-019
## methodに何を選ぶか
df_receipt['rank'] = df_receipt['amount'].rank(method='dense', ascending=False)
df_receipt.sort_values('rank').head(n=10)

## P-020
df_receipt['rank'] = df_receipt['amount'].rank(method='first', ascending=False)
df_receipt.sort_values('rank').head(n=10)

## P-021
df_receipt.shape[0]

## P-022
len(df_receipt['customer_id'].unique())

## P-023
df_receipt.loc[:, ['store_cd', 'amount', 'quantity']].groupby('store_cd').sum()

## P-024
## 元はstrなので日時に変換する
df_receipt['sales_ymd'] = pd.to_datetime(df_receipt['sales_ymd'], format='%Y%m%d')
df_receipt.loc[:, ['customer_id', 'sales_ymd']].groupby('customer_id').apply(lambda x: x.sort_values(by='sales_ymd', ascending=False)).head(10)

## P-025
df_receipt.loc[:, ['customer_id', 'sales_ymd']].groupby('customer_id').apply(lambda x: x.sort_values(by='sales_ymd', ascending=True)).head(10)

## P-026
df_left = df_receipt.loc[:, ['customer_id', 'sales_ymd']].groupby('customer_id').apply(lambda x: x.nlargest(1,['sales_ymd'])).reset_index(drop=True)
df_right = df_receipt.loc[:, ['customer_id', 'sales_ymd']].groupby('customer_id').apply(lambda x: x.nsmallest(1,['sales_ymd'])).reset_index(drop=True)
df_merged = df_left.rename(columns={'sales_ymd': 'sales_ymd_last'}).merge(df_right.rename(columns={'sales_ymd': 'sales_ymd_first'}), how='inner')
df_merged.loc[df_merged.sales_ymd_last != df_merged.sales_ymd_first]

## P-027
df_receipt.loc[:, ['store_cd', 'amount']].groupby('store_cd').mean().sort_values('amount', ascending=False).head(5)

## P-028
df_receipt.loc[:, ['store_cd', 'amount']].groupby('store_cd').median().sort_values('amount', ascending=False).head(5)

## P-029
df_receipt.loc[:, ['store_cd', 'amount']].groupby('store_cd').agg(lambda x:x.value_counts().index[0])

## P-030, 031
## 標本分散か不偏分散か?
df_receipt.loc[:, ['store_cd', 'amount']].groupby('store_cd').agg(lambda x:x.var()).sort_values('amount', ascending=False).head(5)
df_receipt.loc[:, ['store_cd', 'amount']].groupby('store_cd').agg(lambda x:x.std()).sort_values('amount', ascending=False).head(5)

## P-032
df_receipt.loc[:, ['amount']].quantile([.25, .5, .75])

## P-033
df_mean = df_receipt.loc[:, ['store_cd', 'amount']].groupby('store_cd').mean().reset_index(drop=True)
df_mean.loc[df_mean.amount >= 330]

## P-034, P-035
## 顧客IDが'Z'から始まるのものは非会員を表すため除外したデータは、
## 今後の問題で頻繁に出るのでここで作る。 df_receipt_z_excluded という変数に格納する。
df_receipt_z_excluded = df_receipt[df_receipt['customer_id'].apply(lambda x: not x.startswith('Z'))]
df_amount_mean = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
amount_mean = df_amount_mean['amount'].mean()
amount_mean
df_amount_mean[df_amount_mean.amount >= amount_mean].head(10)

df_receipt_z_excluded <- df_receipt %>%
    dplyr::filter(!stringr::str_starts(customer_id, 'Z'))

## P-036, P-037
## R-036, R-037の私の解答は、項目名を絞りすぎ
col_names = list(df_receipt.columns)
col_names.append('store_name')
df_receipt.merge(df_store, how='inner')[col_names].head(10)

col_names = list(df_product.columns)
col_names.append('category_small_name')
df_product.merge(df_category, how='inner')[col_names].head(10)

## P-038
df_merged = df_customer.merge(df_receipt_z_excluded, on=['customer_id'], how='left')
df_merged = df_merged[df_merged.gender_cd == 1]
df_merged = df_merged.loc[:, ['customer_id', 'amount']]
df_merged = df_merged.groupby('customer_id').sum().fillna(0)

## P-039
## 同順位を無視して20位までで打ち切る(issue発行済)
df_r039_n_day = df_receipt_z_excluded.loc[:, ['customer_id', 'sales_ymd']]
df_r039_n_day = df_r039_n_day.groupby('customer_id').nunique().nlargest(20, 'sales_ymd').reset_index(drop=True)
df_r039_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']]
df_r039_amount = df_r039_amount.groupby('customer_id').nunique().nlargest(20, 'amount').reset_index(drop=True)
df_r039_n_day.merge(df_r039_amount, on=['customer_id'], how='left')

## P-040
df_store.assign(key=1).merge(df_product.assign(key=1), how='outer')

## P-041
df_receipt.loc[:, ['amount', 'sales_ymd']].groupby('sales_ymd').sum().diff().head()

## P-042
diff1 = df_receipt.loc[:, ['amount', 'sales_ymd']].groupby('sales_ymd').sum().diff(periods=1).reset_index().rename(columns={'amount': 'diff1'})
diff2 = df_receipt.loc[:, ['amount', 'sales_ymd']].groupby('sales_ymd').sum().diff(periods=2).reset_index().rename(columns={'amount': 'diff2'})
diff3 = df_receipt.loc[:, ['amount', 'sales_ymd']].groupby('sales_ymd').sum().diff(periods=3).reset_index().rename(columns={'amount': 'diff3'})
diff1.merge(diff2, on='sales_ymd', how='left').merge(diff3, on='sales_ymd', how='left').merge(df_receipt, on='sales_ymd', how='inner')

## P-043
df_sales_summary = df_receipt.merge(df_customer, how='inner').replace({'男性': '0', '女性': '1', '不明': '9'})
df_sales_summary['age'] = (df_sales_summary['age'] // 10) * 10
df_sales_summary = df_sales_summary.loc[:, ['amount', 'gender', 'age']].groupby(['gender', 'age']).sum().reset_index()
df_sales_summary = df_sales_summary.pivot(index='age', columns='gender', values='amount').reset_index()

## P-044
df_sales_summary = pd.melt(df_sales_summary, id_vars=['age'], var_name='gender', value_name='g')
df_sales_summary['gender'] = df_sales_summary['gender'].replace({'0': '00', '1': '01', '9': '99'})

## P-045
## 元々strだったのをdateにしてある
df = df_customer.loc[:,['customer_id', 'birth_day']]
df['birth_day'] = df['birth_day'].apply(lambda x: x.strftime('%Y%m%d'))

## P-046
## フォーマットを間違えても警告を出さない!
df_customer['application_date'] = pd.to_datetime(df_customer['application_date'], format='%Y%m%d')
df_customer.loc[:,['customer_id', 'application_date']]

## P-047
df_receipt['sales_ymd'] = pd.to_datetime(df_receipt['sales_ymd'], format='%Y%m%d')
df_receipt.loc[:,['sales_ymd', 'receipt_no', 'receipt_sub_no']]

## P-048
df_receipt['sales_epoch'] = pd.to_datetime(df_receipt['sales_epoch'], unit='s')
df_receipt.loc[:,['sales_epoch', 'receipt_no', 'receipt_sub_no']]

## P-049, 050, 051
df_receipt['year'] = pd.DatetimeIndex(df_receipt['sales_epoch']).year
df_receipt['month'] = pd.DatetimeIndex(df_receipt['sales_epoch']).month
df_receipt['day'] = pd.DatetimeIndex(df_receipt['sales_epoch']).day
df_receipt.loc[:,['year', 'receipt_no', 'receipt_sub_no']]
df_receipt.loc[:,['month', 'receipt_no', 'receipt_sub_no']]
df_receipt.loc[:,['day', 'receipt_no', 'receipt_sub_no']]

## P-052
df_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
df_amount['amount_high'] = (df_amount_mean.amount > 2000) + 0

## P-053
df_customer['tokyo'] = df_customer['postal_cd'].str.split(pat=r"\D+", n=1, expand=True).iloc[:,0].apply(lambda x: 2 - (int(x) >= 100 and int(x) <= 209))
df = df_customer.merge(df_receipt, how='inner')
df.loc[:, ['tokyo', 'customer_id']].groupby('tokyo').nunique()

## P-054
df_customer['pref'] = df_customer['address'].str.split(pat=r"(埼玉県|千葉県|東京都|神奈川県)", n=1, expand=True).loc[:,1].replace({'埼玉県': '11', '千葉県': '12', '東京都': '13', '神奈川県': '14'})
df_customer.loc[:, ['customer_id', 'address', 'pref']]

## P-055
df_amount = df_receipt.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
df_amount['rank'] = pd.qcut(df_amount['amount'], q=[0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4])

## P-056
df = df_customer.loc[:,['customer_id', 'age', 'birth_day']]
df['age'] = ['{}s'.format(x) for x in np.minimum(((df_customer['age'] // 10) * 10).to_list(), 60)]

## P-057
df = df_customer.loc[:,['customer_id', 'age', 'gender']]
df['age'] = ['{}s'.format(x) for x in np.minimum(((df_customer['age'] // 10) * 10).to_list(), 60)]
df['age_gender'] = df['age'].map(str) + df['gender'].map(str)

## P-058
pd.get_dummies(df_customer, columns=['gender_cd'])

## P-059
df_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
scaler = StandardScaler()
scaler.fit(df_amount)
scaler.transform(df_amount)

## P-060
df_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
scaler = MinMaxScaler()
scaler.fit(df_amount)
scaler.transform(df_amount)

## P-061, 062
df_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
df_amount['amount'] = np.log10(df_amount['amount'])

df_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
df_amount['amount'] = np.log(df_amount['amount'])

## P-063
df_product['profit'] = (df_product['unit_price'] - df_product['unit_cost']) / df_product['unit_price']

## P-064
df = df_product.dropna()
df['profit'] = (df['unit_price'] - df['unit_cost']) / df['unit_price']

## P-065, P-066, P-067
## NAに対する計算結果はNAになるので、特に対処しない。
profit_scale = 1.0 / 0.7
df_product['unit_price_floor'] = np.floor(df_product['unit_cost'] * profit_scale)
df_product['unit_price_round'] = np.round(df_product['unit_cost'] * profit_scale)
df_product['unit_price_ceil'] = np.ceil(df_product['unit_cost'] * profit_scale)
sum(np.isnan(df_product['unit_price_floor']))

## P-068
df_product['price_tax_included'] = np.floor(df_product['unit_price'] * 1.1)

## P-069
df_r069_joined = df_receipt.merge(df_product, how='inner')
df_r069_full = df_r069_joined.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum().reset_index()
df_r069_07 = df_r069_joined[df_r069_joined.category_major_cd == 7]
df_r069_07 = df_r069_07.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum().reset_index().rename(columns={'amount': 'amount07'})
df_r069_joined = df_r069_07.merge(df_r069_full, how='inner')
df_r069_joined['ratio'] = df_r069_joined['amount07'] / df_r069_joined['amount']

## ★ P-070..P-074後で!
## 日付は変換済とする
df_receipt = pd.read_csv('../data/receipt.csv', low_memory=False)
df_customer = pd.read_csv('../data/customer.csv', low_memory=False)
df_r070_date = df_receipt.merge(df_customer, how='inner')
df_r070_date['application_date'] = pd.to_datetime(df_r070_date['application_date'], format='%Y%m%d')
df_r070_date['sales_ymd'] = pd.to_datetime(df_r070_date['sales_ymd'], format='%Y%m%d')
df_r070_date['elapsed'] = pd.date_range(str(pd.DatetimeIndex(df_r070_date['application_date'])), str(pd.DatetimeIndex(df_r070_date['sales_ymd'])))

## P-076
df = df_customer.groupby('gender').apply(lambda x: x.sample(frac=0.1))
print(df_customer[df_customer.gender=="男性"].shape)
print(df_customer[df_customer.gender=="女性"].shape)
print(df[df.gender=="男性"].shape)
print(df[df.gender=="女性"].shape)

## P-077
df_amount = df_receipt_z_excluded.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum()
df_amount[(df_amount.amount < np.mean(df_amount.amount) - 3.0 * np.std(df_amount.amount)) |
          (df_amount.amount > np.mean(df_amount.amount) + 3.0 * np.std(df_amount.amount))]

## P-078
iqr = df_amount.amount.quantile([.25, .5, .75])
df_amount[(df_amount.amount < iqr.iloc[1] - 1.5 * (iqr.iloc[2]- iqr.iloc[0])) |
          (df_amount.amount > iqr.iloc[1] + 1.5 * (iqr.iloc[2]- iqr.iloc[0]))]

## P-079
col_names = df_product.columns
na_counts = [len(df_product) - df_product[x].count() for x in col_names]
pandas.DataFrame({'column':col_names, 'nan_count':na_counts})

## P-080
print(df_product.shape)
print(df_product.dropna().shape)

## P-081
mean_price = np.mean(df_product['unit_price'])
mean_cost = np.mean(df_product['unit_cost'])
df_product_2 = df_product.fillna(value={'unit_price':mean_price, 'unit_cost':mean_cost})

## P-082
median_price = np.nanmedian(df_product['unit_price'])
median_cost = np.nanmedian(df_product['unit_cost'])
df_product_3 = df_product.fillna(value={'unit_price':median_price, 'unit_cost':median_cost})

## P-083
df_product_4 = df_product.groupby('category_small_cd').apply(lambda x: x.fillna(value={'unit_price':np.nanmedian(x['unit_price']), 'unit_cost':np.nanmedian(x['unit_cost'])}))

## P-084
df_joined_r084 = df_customer.merge(df_receipt, on='customer_id', how='left').fillna(value={'amount':0})
df_joined_r084['sales_epoch'] = pd.to_datetime(df_joined_r084['sales_epoch'], unit='s')
df_joined_r084['sales_year'] = pd.DatetimeIndex(df_joined_r084['sales_epoch']).year

df_r084_full = df_joined_r084.groupby('customer_id').sum()
df_r084_2019 = df_joined_r084[df_joined_r084.sales_year == 2019].groupby('customer_id').sum().rename(columns={'amount': 'amount2019'})
df_r084 = df_r084_full.merge(df_r084_2019, how='left').fillna(value={'amount2019':0})
df_r084.shape
df_r084.dropna().shape

## P-085
df_085 = df_customer.merge(df_geocode, on='postal_cd')
df_085 = df_085.loc[:,['postal_cd', 'longitude', 'latitude']].groupby('postal_cd').mean()
df_customer_1 = df_customer.merge(df_085, on='postal_cd')

## P-086
df_customer_2 = df_customer_1.merge(df_store, left_on='application_store_cd', right_on='store_cd').rename(columns={'address_x': 'address_customer', 'address_y': 'address_store'})

## 緯度(latitude)、経度(longitude)
df_customer_2['distance'] = df_customer_2.apply(lambda x: geodesic((x['latitude_x'], x['longitude_x']), (x['latitude_y'], x['longitude_y'])).km, axis=1)

## P-087
df = df_receipt.loc[:, ['customer_id', 'amount']].groupby('customer_id').sum().reset_index()
df_customer_u = df_customer.merge(df, how='left').fillna(value={'amount':0}).sort_values(['amount', 'customer_id']).drop_duplicates(subset=['customer_name', 'postal_cd'])

## P-088
df_customer_n = df_customer.drop('customer_id', 1).merge(df_customer_u.loc[:,['customer_id', 'customer_name', 'postal_cd']], how='inner')

## P-089
df_sample_r089 = df_customer_u[df_customer_u.amount > 0]
df_train, df_test = train_test_split(df_sample_r089, train_size=0.8)
print(df_train.shape)
print(df_test.shape)

## ★ P-090 後で

## P-091
n_samples = 100
df_r091 = df_r084.copy()
df_r091_non_zeros = df_r091[df_r091.amount > 0].sample(n_samples)
df_r091_zeros = df_r091[df_r091.amount == 0].sample(n_samples)
print(df_r091_non_zeros.shape)
print(df_r091_zeros.shape)

## P-092
## gender_cdがあればgenderは一意に決まるので、別のテーブルに括り出す。

df_customer.drop('gender', 1)
df_customer.loc[:,['gender_cd', 'gender']].drop_duplicates(subset=['gender_cd', 'gender'])

## P-093
df_product_r093 = df_product.merge(df_category, on=['category_major_cd', 'category_medium_cd', 'category_small_cd'], how='left')

## P-094, P-097
io_dirname = 'output'
os.makedirs(io_dirname, exist_ok=True)
df_product_r093.to_csv(os.path.join(io_dirname, 'r094.csv'), sep=',', header=True, index=False, encoding='utf-8-sig')
df_r097 = pd.read_csv(os.path.join(io_dirname, 'r094.csv'), low_memory=False)
assert df_product_r093.shape == df_r097.shape

## P-095
df_product_r093.to_csv(os.path.join(io_dirname, 'r095.csv'), sep=',', header=True, index=False, encoding='cp932')

## P-096, P-098
df_product_r093.to_csv(os.path.join(io_dirname, 'r096.csv'), sep=',', header=False, index=False, encoding='utf-8-sig')
df_r098 = pd.read_csv(os.path.join(io_dirname, 'r096.csv'), header=None, low_memory=False)
assert df_product_r093.shape == df_r098.shape

## P-099, P-100
df_product_r093.to_csv(os.path.join(io_dirname, 'r099.csv'), sep="\t", header=True, index=False, encoding='utf-8-sig')
df_r100 = pd.read_csv(os.path.join(io_dirname, 'r099.csv'), sep="\t", low_memory=False)
assert df_product_r093.shape == df_r100.shape

## Done
