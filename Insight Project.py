# Part 1: Who is most likely to regain weight?

from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Connect to PgAdmin Database: 

engine = create_engine('postgresql+psycopg2://username:password@host:port/database_name')
data= pd.read_sql_table(table_name=' ',con=engine,schema=' ',columns=['col1', 'col2'])


#Import Databases from Pgadmin (SQL):
Scales= pd.read_sql_table(table_name='scales',con=engine2,schema='public')
posts= pd.read_sql_table(table_name='posts',con=engine2,schema='public')
schedulings= pd.read_sql_table(table_name='schedulings',con=engine2,schema='public')
mixpanel_export= pd.read_sql_table(table_name='mixpanel_export',con=engine1,schema='mixpanel_integration_12')
users= pd.read_sql_table(table_name='users',con=engine2,schema='public')
status= pd.read_csv('status.csv')
status=status.rename(columns = {'is':'user_id'})
status2=status
scales1= Scales[['user_id','id','weight_pounds','timestamp']]

# Sort dataframe, first by user id and then by creating
scales1 = scales1.sort_values(by = ['user_id', 'timestamp'])

# Exlcude rows where weight is < 100
scales1 = scales1[scales1.weight_pounds > 100]
scales1['weight_pct_chng']=scales1.groupby('user_id')['weight_pounds'].pct_change()*100
idx = np.unique(scales1.user_id.values, return_index=1)[1]

# Exlcude rows where weight change is greater than 4% to remove outliers
scales1.iloc[idx, scales1.columns.get_loc('weight_pct_chng')] = 0
scales2=scales1[(scales1['weight_pct_chng'] >= -4)&(scales1['weight_pct_chng'] <= 4)]
userid=pd.DataFrame(data=scales2.user_id.unique(), columns=['user_id'])

#Select desired columns from the imported tables:
users1=users[['id','email','created_at','admin','first_name','last_name','provider','demo','remote','gender','birthdate','drchrono_id','app','pays_cash','program_start_date','last_reviewed_at','profile_status_id','appointment_frequency_id','diabetic','fitbit_access_token','drchrono_chart_id','healthdata_source']]
appointments1=Chrono[['First Name','Last Name','Date of Service','Appt Status','Billed Time', 'Billing Status','Duration','Blood Pressure Systolic','Blood Pressure Diastolic', 'Height', 'Weight', 'Chart ID']]
appointments1=appointments1.rename(columns = {'Chart ID':'drchrono_chart_id'})
mixpanel_export1=mixpanel_export[['User App','User Appointment Frequency','User Completed Cards','User Email','User Food Pictures Count','User Last Name','User Notifications Count','event','mp_reserved_city','mp_reserved_region','time']]
messages1=messages[['user_id','date','content','sender_id']]
partnerships1=partnerships[['user_id','provider_id','primary']]
posts1=posts[['user_id','created_at','completed']]
schedulings1=schedulings[['user_id','appointment_type','status']]

# Group and merge databases:
schedule2= pd.read_csv('schedule.csv')
avg_blood_pressure_systolic=pd.DataFrame(appointments1.groupby('drchrono_chart_id')['Blood Pressure Systolic'].mean())
avg_blood_pressure_diastolic=pd.DataFrame(appointments1.groupby('drchrono_chart_id')['Blood Pressure Diastolic'].mean())
height=pd.DataFrame(appointments1.groupby('drchrono_chart_id')['Height'].mean())
dd=pd.concat([avg_blood_pressure_systolic,avg_blood_pressure_diastolic,height],axis=1)
dd.reset_index(level=0, inplace=True)
number_of_appointments=schedule2[schedule2['Canceled']!='canceled']
number_of_appointments=number_of_appointments[['Email','Date Scheduled','Canceled']]
number_of_appointments=pd.DataFrame(number_of_appointments.groupby(['Email'])['Canceled'].size())
number_of_appointments.reset_index(level=0, inplace=True)
number_of_appointments=number_of_appointments.rename(columns = {'Canceled':'num_appoint'})
number_of_appointments=number_of_appointments.rename(columns = {'Email':'email'})
users2=pd.merge(number_of_appointments,users1,how='outer',on=('email'))
users2=users2.rename(columns={'id':'user_id'})
users2=users1
users2=users2.rename(columns = {'id':'user_id'})
messages2=pd.DataFrame(messages1.groupby('user_id').size())
messages2=messages2.rename(columns={0:'total_messages'})
from textblob import TextBlob
messages1['score'] = messages1['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
messages2=messages1.sort_values(by = ['user_id', 'date'])
mes_group = messages2.groupby('sender_id').agg({'score':'mean', 'content':'count'}).reset_index()
mes_group=mes_group.rename(columns={'score':'sender score'})
mes_group=mes_group.rename(columns={'content':'content count by sender'})
mes_group=mes_group.rename(columns={'sender_id':'user_id'})
mes_group2 = messages2.groupby('user_id').agg({'score':'mean', 'content':'count'}).reset_index()
mes_group=mes_group.rename(columns={'score':'user score'})
mes_group=mes_group.rename(columns={'content':'content count by user'})
partnership2=pd.DataFrame(partnerships1.groupby('user_id').first()['primary'])
posts2=pd.DataFrame(posts1.groupby('user_id').size())
posts2=posts2.rename(columns={0:'total_posts'})
online_appointments=pd.DataFrame(schedulings1.groupby('user_id')['appointment_type'].apply(lambda x: x[x.str.contains('Online')].count()))
online_appointments=online_appointments.rename(columns = {'appointment_type':'online_appointments'})
in_person_appointments=pd.DataFrame(schedulings1.groupby('user_id')['appointment_type'].apply(lambda x: x[x.str.contains('person')].count()))
in_person_appointments=in_person_appointments.rename(columns = {'appointment_type':'in_person_appointments'})
scheduling2=pd.concat([online_appointments,in_person_appointments],axis=1)
status2=status1
status2=status2.rename(columns = {'id':'user_id'})
status2=status2.rename(columns = {'text':'status'})
df1=pd.merge(users2,mes_group,how='left',on=('user_id'))
df1=pd.merge(df1,mes_group2,how='left',on=('user_id'))
df1=pd.merge(df1,partnership2,how='left',on=('user_id'))
df1=pd.merge(df1,posts2,how='left',on=('user_id'))
df1=pd.merge(df1,scheduling2,how='left',on=('user_id'))
df1=pd.merge(df1,status2,how='left',on=('user_id'))
df1=pd.merge(df1,dd,how='left',on=('drchrono_chart_id'))
df2=df1.drop(['email','first_name', 'last_name','drchrono_id','last_reviewed_at', 'appointment_frequency_id', 'drchrono_chart_id','program_start_date','fitbit_access_token','healthdata_source'],axis=1)
df2['created_at']= df2['created_at'].dt.date
today=datetime.date(2019, 1, 31)
df2['days_in_program']=(today-df2['created_at'])
df2['days_in_program']=df2.days_in_program / np.timedelta64(1, 'D')
df2=df2.drop(['created_at'],axis=1)

# Delete these row indexes from dataFrame
indexNames = df2[ (df2['admin'] == False) & (df2['app'] == True) &  (df2['demo'] == False) &  (df2['status'].isin(['Active','Completed']))].index
df2=df2.iloc[indexNames]
df2 = df2.sort_values(by = ['user_id'])
df2['age']=today-df2['birthdate'].dt.date
df2['age']=(df2.age / np.timedelta64(1, 'D'))/365
df2=df2.drop(['admin','app','demo','provider','birthdate'],axis=1)

#Flagging those patients who regained 10% of their weight back:
last_meas=scales2.groupby('user_id')['weight_pounds'].last()
min_meas=scales2.groupby('user_id')['weight_pounds'].min()
first_meas=scales2.groupby('user_id')['weight_pounds'].first()
pct_chng_from_min=(last_meas-min_meas)/first_meas*100
pct_chng_from_min=pd.DataFrame(pct_chng_from_min)
pct_chng_from_min.reset_index(level=0, inplace=True)
pct_chng_from_min=pct_chng_from_min.rename(columns={'weight_pounds':'pct_regained_weight'})
indexNames = pct_chng_from_min[ (pct_chng_from_min['pct_regained_weight'] > 10].index
pct_chng_from_min=pct_chng_from_min.iloc[indexNames]
df2=pd.merge(df2,pct_chng_from_min,how='left',on=('user_id'))
df2.pct_regained_weight.fillna(0, inplace=True)
df2['flag'] = 0
df2['flag'][df2['pct_regained_weight']>0] = 1


#Preprocessing
df2['age']=df2['age'].fillna(df2['age'].median())
df2['Blood Pressure Systolic']=df2['Blood Pressure Systolic'].fillna(df2['Blood Pressure Systolic'].mean())
df2['Blood Pressure Diastolic']=df2['Blood Pressure Diastolic'].fillna(df2['Blood Pressure Diastolic'].mean())
df2['Height']=df2['Height'].fillna(df2['Height'].mean())
df2['total_posts'] = df2['total_posts'].replace('nan', np.nan).fillna(0)
df2['online_appointments'] = df2['online_appointments'].replace('nan', np.nan).fillna(0)
df2['in_person_appointments'] = df2['in_person_appointments'].replace('nan', np.nan).fillna(0)
from sklearn import preprocessing
df22=df2[['profile_status_id','pays_cash','diabetic', 'primary', 'status','remote']]
df2=df2.drop(['gender','pays_cash','diabetic', 'primary', 'status','remote'],axis=1)
le = preprocessing.LabelEncoder()
df22 = df22.apply(le.fit_transform)
df222=pd.concat([df2,df22],axis=1)
df222['total_posts']=df222['total_posts']/df222['days_in_program']
df222['online_appointments']=df222['online_appointments']/df222['days_in_program']
df222['in_person_appointments']=df222['in_person_appointments']/df222['days_in_program']
df222['num_appoint'] = df222['num_appoint'].replace('nan', np.nan).fillna(0)
df222['score'] = df222['score'].replace('nan', np.nan).fillna(0)
df222['content'] = df222['content'].replace('nan', np.nan).fillna(0)
df222['sender score'] = df222['sender score'].replace('nan', np.nan).fillna(0)
df222['content count by sender'] = df222['content count by sender'].replace('nan', np.nan).fillna(0)
df222['flag']=df2.flag
df222.columns=['Number of Appointments','user_id', 'profile_status_id', 'Patient Sentiment Score','Number of Messages Sent by Patient', 'Overall Sentiment Score', 'Total Messages', 'total_posts', 'Number of Online Appointments', 'Number of In-person Appointments','Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Height', 'Number of Days in program', 'Age','pct_regained_weight','flag', 'profile_status_idd', 'Insurance Status','Diabetic','Primary', 'Status','Remote']

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
X = X_patients.as_matrix()
y = y_patients.as_matrix()
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

feat_importances = pd.Series(forest.feature_importances_, index=X_patients.columns)
feat_importances.nlargest(14).plot(kind='barh')

#Cross Validation
from sklearn.model_selection import cross_val_score
X = X_patients.as_matrix()
y = y_patients.as_matrix()
cv_scores = cross_val_score(clf1, X, y)
print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))
     
     
     
     

#Part 2: Live Measurement of Weight Changes:

# Report weight statuts, using Autoregressive model:
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import time
import datetime
scales3=pd.read_csv('scales3.csv')
series = Series.from_csv('scales3.csv', header=0)
today = time.time()
one_month_ago=today-2680437
two_month_ago=today-5360875
three_month_ago=today-8041312
# split dataset
users11=scales3['user_id'].unique()
for x in users11:
    try:
        a=scales3[(scales3['user_id'] == x)&(scales3['Date']<one_month_ago)&(scales3['Date']>three_month_ago)].weight
        b=scales3[(scales3['user_id'] == x)&(scales3['Date']>one_month_ago)].weight
        X = a.values
        train, test = a.values, b.values
        # train autoregression
        model = AR(train)
        model_fit = model.fit()
        # make predictions
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
        error = test.mean()- predictions.mean()
        if error >10:
            print('User ID',x, 'has gained',"%.1f" %error,'pounds, in average, during the past month.')
        else:
            print('User ID',x, 'has lost',"%.1f" %error,'pounds, in average, during the past month.')
       
    except:
        pass


# Plot weight change measurement:
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import time
import datetime
scales3=pd.read_csv('scales3.csv')
series = Series.from_csv('scales3.csv', header=0)
today = time.time()
one_month_ago=today-2680437
two_month_ago=today-5360875
three_month_ago=today-8041312
# split dataset
a=scales3[(scales3['user_id'] == 4)&(scales3['Date']<one_month_ago)&(scales3['Date']>three_month_ago)].weight
b=scales3[(scales3['user_id'] == 4)&(scales3['Date']>one_month_ago)].weight
X = a.values
train, test = a.values, b.values
# train autoregression
model = AR(train)
model_fit = model.fit()
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
c=np.concatenate((train, test))
c1=np.concatenate((train, predictions))
# plot results
plt.figure(figsize=(10, 8)) 
pyplot.plot(c1, color='red')
pyplot.plot(c, color='blue')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.gca().legend(('Predicted Weight','Actual Weight'))
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  
plt.xlabel("Time", fontsize=16)  
plt.ylabel("Weight", fontsize=16)  
plt.ylim(150,170)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.savefig("chess.png", bbox_inches="tight")
pyplot.show()
