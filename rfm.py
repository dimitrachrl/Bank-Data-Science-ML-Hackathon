import pandas as pd
import numpy as np
from datetime import date


merged_transactions=pd.read_csv('merged_transactions_1.csv')


merged_transactions['created_date']=pd.to_datetime(merged_transactions['created_date'])


merged_transactions['created_date'] = merged_transactions['created_date'].dt.strftime('%Y-%m')


a=merged_transactions[merged_transactions['transactions_state']=='COMPLETED'].groupby(['created_date','user_id',])['amount_usd'].count()

b=merged_transactions[merged_transactions['transactions_state']=='COMPLETED'].groupby(['created_date'])['amount_usd'].count()

c=merged_transactions[merged_transactions['transactions_state']=='COMPLETED'].groupby(['created_date'])['amount_usd'].sum()



myusers=pd.read_csv('the_users.csv')


drops=[feature for feature in  myusers.columns if feature not in ['days_from_last_trans','sum_amount_of_transactions','count_of_transactions']]

for feature in drops:
    myusers.drop(feature,axis=1,inplace=True)
    
    
    
quantiles = myusers.quantile(q=[0.25,0.5,0.75])



def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

#create rfm segmentation table 
rfm_segmentation = myusers
rfm_segmentation['R_Quartile'] = rfm_segmentation['days_from_last_trans'].apply(RScore, args=('days_from_last_trans',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['count_of_transactions'].apply(FMScore, args=('count_of_transactions',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['sum_amount_of_transactions'].apply(FMScore, args=('sum_amount_of_transactions',quantiles,))

rfm_segmentation.head()

rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) \
                            + rfm_segmentation.F_Quartile.map(str) \
                            + rfm_segmentation.M_Quartile.map(str)
                            
                            
                            
def churn_detector(column):
    if '1' in column:
        return 'Churned'
    else:
        return 'Not Churned'
    

rfm_segmentation['RFMScorew']=rfm_segmentation['RFMScore'].map(churn_detector)
    



bb=rfm_segmentation[(rfm_segmentation['R_Quartile']<=2) & (rfm_segmentation['F_Quartile']<=2)]















