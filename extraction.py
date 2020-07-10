import pandas as pd
import numpy as np
from datetime import date
import time 

t1=time.time()

#expanding columns view
pd.set_option('display.max_columns', 500)
#expanding rows view
pd.set_option('display.max_rows', 500)

#importing csv, excel files 
users=pd.read_excel('my_users.xlsx')
transactions_1=pd.read_csv('transactions_1.csv')
transactions_2=pd.read_csv('transactions_2.csv')
transactions_3=pd.read_csv('transactions_3.csv')

#converting 'created_date' column to datetime type
users['created_date']=pd.to_datetime(users['created_date'])
transactions_1['created_date']=pd.to_datetime(transactions_1['created_date'])
transactions_2['created_date']=pd.to_datetime(transactions_2['created_date'])
transactions_3['created_date']=pd.to_datetime(transactions_3['created_date'])

#concatenate 3 transactions DataFrames to one big DataFrame
transactions=pd.concat([transactions_1,transactions_2,transactions_3])

#function to get age of every user
def get_age(year):
    return 2020-year

#creating new column 'age' for every user
users['age']=np.vectorize(get_age)(users['birth_year'])


#function to find how many days have passed since every transaction
def days_between(mydate):
    return pd.to_datetime(date(2020,6,11))-mydate

#applying days_between to transactions dataframe and creatin a new column 'days_from_trans'
transactions['days_from_trans']=np.vectorize(days_between)(transactions['created_date'])

#applying days_between to users dataframe and creating a new column 'days_of acc'
users['days_of_acc']=np.vectorize(days_between)(users['created_date'])

#remove year column in users dataframe
users.drop(['Year','created_date'],axis=1,inplace=True)

#splitting the column and keep only days value 
transactions['days_from_trans']=transactions['days_from_trans'].astype(str)
transactions['days_from_trans']=transactions['days_from_trans'].str.split(' ').str[0].astype(int)

#splitting the column and keep only days value 
users['days_of_acc']=users['days_of_acc'].astype(str)
users['days_of_acc']=users['days_of_acc'].str.split(' ').str[0].astype(int)


#finding how many days have passed since first and last transaction for every user.
days_from_last_transactions=transactions.groupby('user_id')['days_from_trans'].min()
days_from_first_transactions=transactions.groupby('user_id')['days_from_trans'].max()



#merging users dataframe with transactions df
users=users.merge(days_from_first_transactions, how='left', on='user_id')
users=users.merge(days_from_last_transactions, how='left', on='user_id')

#rename the columns
users=users.rename(columns={'days_from_trans_x' : 'days_from_first_trans', 'days_from_trans_y' :  'days_from_last_trans'})




#left join users with transactions
merged_transactions=users.merge(transactions, on='user_id', how='left')

#creating 4 metrics sum, mean, frequency, cashback
sum_amount_per_user=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['amount_usd'].sum()
mean_amount_per_user=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['amount_usd'].mean()
frequency_of_transactions=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['amount_usd'].count()
cashback_amount=transactions[transactions['transactions_type']=='CASHBACK'].groupby('user_id')['amount_usd'].sum()

#joining 4 metrics to users dataframe
users=users.merge(sum_amount_per_user, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'sum_amount_of_transactions'})

users=users.merge(mean_amount_per_user, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'mean_amount_of_transactions'})

users=users.merge(frequency_of_transactions, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'count_of_transactions'})

users=users.merge(cashback_amount, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'cashback_amount'})


#fill nan values with 0
users['cashback_amount'].fillna(0,inplace=True)

users.drop(['birth_year'],axis=1,inplace=True)
#remove created date in transactions dataframe
transactions.drop(['created_date'],axis=1,inplace=True)

#fill nan values with 0
users['mean_amount_of_transactions'].fillna(0,inplace=True)

#fill nan values with 0
users['count_of_transactions'].fillna(0,inplace=True)

#fill nan values with 0
users['sum_amount_of_transactions'].fillna(0,inplace=True)

users=users.dropna()


#export final dataframe to csv.
users.to_csv('the_users.csv', index=False)




t2=time.time()

print(t2-t1)

