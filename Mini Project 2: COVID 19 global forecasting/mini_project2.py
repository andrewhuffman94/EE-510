# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.express as px
from plotly.offline import plot
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score
from urllib.request import urlopen
import json
import matplotlib.dates as mdates


# Read data
data_training = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
y_dates = data_test["Date"]


############################## EDA ######################################
#data_cases = data_training[data_training["Target"] == "ConfirmedCases"]
#data_deaths = data_training[data_training["Target"] == "Fatalities"]
#
## Plot country pie chart for confirmed cases
#data_Country = data_cases[data_cases["Province_State"].isnull()]
#data_Country_totals_cases = data_Country.groupby(['Country_Region',"Population",'Target'], as_index=False)['TargetValue'].sum()
#data_Country_totals_20 = data_Country_totals_cases.nlargest(20,'TargetValue')
#fig = px.pie(data_Country_totals_20, values='TargetValue', names='Country_Region')
#fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
##fig.write_image("PieChart_Countries_Cases.eps")
##plot(fig)
## Plot country pie chart for deaths
#data_Country = data_deaths[data_deaths["Province_State"].isnull()]
#data_Country_totals_deaths = data_Country.groupby(['Country_Region',"Population",'Target'], as_index=False)['TargetValue'].sum()
#data_Country_totals_20 = data_Country_totals_deaths.nlargest(20,'TargetValue')
#fig = px.pie(data_Country_totals_20, values='TargetValue', names='Country_Region')
#fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
##fig.write_image("PieChart_Countries_Deaths.eps")
##plot(fig)
#
## Plot state pie chart for cases
#data_USA_cases = data_cases[data_cases["Country_Region"]=="US"].dropna(subset=["Province_State","County"])
#data_state_totals_cases = data_USA_cases.groupby(["Province_State"],as_index=False)["TargetValue"].sum()
#data_state_totals_20 = data_state_totals_cases.nlargest(20,'TargetValue')
#fig = px.pie(data_state_totals_20, values='TargetValue', names='Province_State')
#fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
##fig.write_image("PieChart_States_Cases.eps")
##plot(fig)
## Plot state pie chart for deaths
#data_USA_deaths = data_deaths[data_deaths["Country_Region"]=="US"].dropna(subset=["Province_State","County"])
#data_state_totals_deaths = data_USA_deaths.groupby(["Province_State","Population"],as_index=False)["TargetValue"].sum()
#data_state_totals_20 = data_state_totals_deaths.nlargest(20,'TargetValue')
#fig = px.pie(data_state_totals_20, values='TargetValue', names='Province_State')
#fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
##fig.write_image("PieChart_States_Deaths.eps")
##plot(fig)
#
#
## Plot Oregon counties pie chart for cases
#data_OR_cases = data_USA_cases[data_USA_cases["Province_State"] == "Oregon"]
#data_OR_totals_cases = data_OR_cases.groupby(["County","Population","Target"],as_index=False)["TargetValue"].sum()
#data_OR_totals_20 = data_OR_totals_cases.nlargest(20,'TargetValue')
#fig = px.pie(data_OR_totals_20, values='TargetValue', names='County')
#fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
##fig.write_image("PieChart_OR_Cases.eps")
##plot(fig)
## Plot Oregon counties pie chart for deaths
#data_OR_deaths = data_USA_deaths[data_USA_deaths["Province_State"] == "Oregon"]
#data_OR_totals_deaths = data_OR_deaths.groupby(["County","Population","Target"],as_index=False)["TargetValue"].sum()
#data_OR_totals_20 = data_OR_totals_deaths.nlargest(20,'TargetValue')
#fig = px.pie(data_OR_totals_20, values='TargetValue', names='County')
#fig.update_traces(textposition='inside')
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
#fig.write_image("PieChart_OR_Deaths.eps")
#plot(fig)
#
##################### Plot percentage infected and death rate globally #############################
#
#percentage_infected = (data_Country_totals_cases["TargetValue"].divide(data_Country_totals_cases["Population"])).multiply(100)
#mortality_rate = data_Country_totals_deaths["TargetValue"].divide(data_Country_totals_cases["TargetValue"]).multiply(100)
#data_Country_rates = pd.concat([data_Country_totals_cases["Country_Region"],percentage_infected,mortality_rate],axis=1)
#data_Country_rates.columns=["Country_Region","Percentage_Infected","Mortality_Rate"]
#
#fig = px.choropleth(data_Country_rates, locations="Country_Region", locationmode='country names', color=data_Country_rates["Percentage_Infected"],range_color=[0,0.5], hover_name="Country_Region", hover_data=['Percentage_Infected'], color_continuous_scale="blues", title='Percentage of Population Infected')
#fig.update(layout_coloraxis_showscale=True)
#fig.write_image("Percentage_Infected_Map.eps")
#plot(fig)
#
#fig = px.choropleth(data_Country_rates, locations="Country_Region", locationmode='country names', color=data_Country_rates["Mortality_Rate"],range_color=[0,15], hover_name="Country_Region", hover_data=['Mortality_Rate'], color_continuous_scale="reds", title='Mortality Rate (%)')
#fig.update(layout_coloraxis_showscale=True)
#fig.write_image("Mortality_Rate_Map.eps")
#plot(fig)
#
#
################# Plot percentage infected and mortality rate by state in US  #######################
#us_state_abbrev = {
#    'Alabama': 'AL',
#    'Alaska': 'AK',
#    'American Samoa': 'AS',
#    'Arizona': 'AZ',
#    'Arkansas': 'AR',
#    'California': 'CA',
#    'Colorado': 'CO',
#    'Connecticut': 'CT',
#    'Delaware': 'DE',
#    'District of Columbia': 'DC',
#    'Florida': 'FL',
#    'Georgia': 'GA',
#    'Guam': 'GU',
#    'Hawaii': 'HI',
#    'Idaho': 'ID',
#    'Illinois': 'IL',
#    'Indiana': 'IN',
#    'Iowa': 'IA',
#    'Kansas': 'KS',
#    'Kentucky': 'KY',
#    'Louisiana': 'LA',
#    'Maine': 'ME',
#    'Maryland': 'MD',
#    'Massachusetts': 'MA',
#    'Michigan': 'MI',
#    'Minnesota': 'MN',
#    'Mississippi': 'MS',
#    'Missouri': 'MO',
#    'Montana': 'MT',
#    'Nebraska': 'NE',
#    'Nevada': 'NV',
#    'New Hampshire': 'NH',
#    'New Jersey': 'NJ',
#    'New Mexico': 'NM',
#    'New York': 'NY',
#    'North Carolina': 'NC',
#    'North Dakota': 'ND',
#    'Northern Mariana Islands':'MP',
#    'Ohio': 'OH',
#    'Oklahoma': 'OK',
#    'Oregon': 'OR',
#    'Pennsylvania': 'PA',
#    'Puerto Rico': 'PR',
#    'Rhode Island': 'RI',
#    'South Carolina': 'SC',
#    'South Dakota': 'SD',
#    'Tennessee': 'TN',
#    'Texas': 'TX',
#    'Utah': 'UT',
#    'Vermont': 'VT',
#    'Virgin Islands': 'VI',
#    'Virginia': 'VA',
#    'Washington': 'WA',
#    'West Virginia': 'WV',
#    'Wisconsin': 'WI',
#    'Wyoming': 'WY'
#}
#
## thank you to @kinghelix and @trevormarburger for this idea
##abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
#data_state_totals_cases = data_USA_cases.groupby(["Province_State"],as_index=False)["TargetValue"].sum()
#data_state_totals_deaths = data_USA_deaths.groupby(["Province_State"],as_index=False)["TargetValue"].sum()
#population_county = data_USA_cases.groupby(["County","Province_State"],as_index=False)["Population"].mean()
#population_state = population_county.groupby(["Province_State"],as_index=False)["Population"].sum()
#percentage_infected_state = pd.concat([population_state["Province_State"],(data_state_totals_cases["TargetValue"].divide(population_state["Population"])).multiply(100)],axis=1)
#mortality_rate_state = pd.concat([population_state["Province_State"],(data_state_totals_deaths["TargetValue"].divide(data_state_totals_cases["TargetValue"])).multiply(100)],axis=1)
#percentage_infected_state = percentage_infected_state.replace({"Province_State":us_state_abbrev})
#mortality_rate_state = mortality_rate_state.replace({"Province_State":us_state_abbrev})
##fig = px.choropleth(percentage_infected_state, locations="Province_State", locationmode='USA-states', color=percentage_infected_state[0],range_color=[0,1.5], scope="usa",hover_name="Province_State", hover_data=[0], color_continuous_scale="blues", title='Percentage of Population Infected')
##fig.update(layout_coloraxis_showscale=True)
##fig.write_image("Perecentage_Infected_State_Map.eps")
##plot(fig)
##
#fig = px.choropleth(mortality_rate_state, locations="Province_State", locationmode='USA-states', color=mortality_rate_state["TargetValue"],range_color=[0,10], scope="usa",hover_name="Province_State", hover_data=["TargetValue"], color_continuous_scale="reds", title='Mortality Rate (%)')
#fig.update(layout_coloraxis_showscale=True)
#fig.write_image("Mortality_Rate_State_Map.eps")
#plot(fig)
#
## Convert features formats
#date_time = pd.to_datetime(data_training["Date"])
#weekdays = date_time.dt.dayofweek
#le = preprocessing.LabelEncoder()
#enc = OneHotEncoder()
#le.fit(data_training["Target"])
#target_training = le.transform(data_training["Target"])
#target_test = le.transform(data_test["Target"])
#le.fit(data_training["Country_Region"])
#country_training = le.transform(data_training["Country_Region"])
#country_test = le.transform(data_test["Country_Region"])
#dates = pd.concat([data_training["Date"],data_test["Date"]]).unique()
#le.fit(dates)
#time_training = le.transform(data_training["Date"])
#time_test = le.transform(data_test["Date"])
#states = pd.concat([data_training["Province_State"],data_test["Province_State"]]).unique().astype(str)
#le.fit(states)
#states_training = le.transform(data_training["Province_State"].astype(str))
#states_test = le.transform(data_test["Province_State"].astype(str))
#counties = pd.concat([data_training["County"],data_test["County"]]).unique().astype(str)
#le.fit(counties)
#county_training = le.transform(data_training["County"].astype(str))
#county_test = le.transform(data_test["County"].astype(str))
#
##states = data_training["Province_State"].astype(str).to_numpy().reshape(states.shape[0],1)
##enc.fit(states)
##states_training2 = enc.transform(states).toarray()
#
#features_training = data_training
#features_training["County"] = county_training
#features_training["Province_State"] = states_training
#features_training["Country_Region"] = country_training
#features_training["Date"] = time_training
#features_training["Target"] = target_training
#features_training.insert(7,"Day",weekdays)
#
#features_test = data_test.drop(columns=["ForecastId"])
#features_test["County"] = county_test
#features_test["Province_State"] = states_test
#features_test["Country_Region"] = country_test
#features_test["Date"] = time_test
#features_test["Target"] = target_test
#
#training_data = features_training.loc[features_training["Date"]<95]
#test_data = features_training.loc[features_training["Date"]>=95]
#Id = training_data["Id"]
#ForecastId = test_data["Id"]
#X_train = training_data.loc[training_data["Date"]<81]
#X_val = training_data.loc[training_data["Date"]>80]
#y_train = X_train["TargetValue"]
#y_val = X_val["TargetValue"]
#X_train = X_train.drop(columns=["TargetValue","Id"])
#X_val = X_val.drop(columns=["TargetValue","Id"])
#y_true = test_data["TargetValue"]
#X_test = test_data.drop(columns=["Id","TargetValue"])
##
##
#X_train = X_train.to_numpy()
#X_train1 = X_train[::2]
#X_train2 = X_train[1::2]
#X_val = X_val.to_numpy()
#X_val1 = X_val[::2]
#X_val2 = X_val[1::2]
#X_test = X_test.to_numpy()
#X_test1 = X_test[::2]
#X_test2 = X_test[1::2]
#y_train = y_train.to_numpy()
#y_train1 = y_train[::2]
#y_train2 = y_train[1::2]
#y_val = y_val.to_numpy()
#y_val1 = y_val[::2]
#y_val2 = y_val[1::2]
#y_test = y_true.to_numpy()
#y_test1 = y_test[::2]
#y_test2 = y_test[1::2]
##
##ridge = Ridge()
##ridge.fit(X_train1,y_train1)
##pred_val1 = ridge.predict(X_val1)
##score_val1 = ridge.score(X_val1,y_val1)
##ridge.fit(X_train2,y_train2)
##pred_val2 = ridge.predict(X_val2)
##score_val2 = ridge.score(X_val2,y_val2)
#
# #############################################################################
## Fit  model
##dtrain1 = xgb.DMatrix(X_train1,label=y_train1)
##dval1 = xgb.DMatrix(X_val1,label=y_val1)
##dtest1 = xgb.DMatrix(X_test1)
##param = {'max_depth': 6, 'eta': 0.5, 'objective': 'reg:squarederror'}
##evallist = [(dval1,"eval"),(dtrain1,"train")]
##num_round = 1000
##bst = xgb.train(param,dtrain1,num_round,evallist)
##pred_bst_test1 = bst.predict(dtest1)
##
##dtrain2 = xgb.DMatrix(X_train2,label=y_train2)
##dval2 = xgb.DMatrix(X_val2,label=y_val2)
##dtest2 = xgb.DMatrix(X_test2)
##param = {'max_depth': 6, 'eta': 0.5, 'objective': 'reg:squarederror'}
##evallist = [(dval2,"eval"),(dtrain2,"train")]
##num_round = 1000
##bst = xgb.train(param,dtrain2,num_round,evallist)
##pred_bst_test2 = bst.predict(dtest2)
#
################ Extra Tress #########################
##reg_ET = ExtraTreesRegressor(n_estimators=2000,n_jobs=-1,verbose=1)
##reg_ET.fit(X_train1,y_train1)
##pred_ET_val1 = reg_ET.predict(X_val1)
##R21 = r2_score(y_val1,pred_ET_val1)
##pred_ET_test1 = reg_ET.predict(X_test1)
##
##reg_ET.fit(X_train2,y_train2)
##pred_ET_val2 = reg_ET.predict(X_val2)
##R22 = r2_score(y_val2,pred_ET_val2)
##pred_ET_test2 = reg_ET.predict(X_test2)
##
##L_1 = np.absolute(pred_ET_val1-y_val1)*0.5
##L_2 = np.absolute(pred_ET_val2-y_val2)*0.5
##score_1 = (1/y_val1.shape[0])*np.sum(np.multiply(X_val1[:,4],L_1),axis=0)
##score_2 = (1/y_val2.shape[0])*np.sum(np.multiply(X_val2[:,4],L_2),axis=0)
# #############################################################################
#
####################### Random Forest Regresssion #################################
##R2_val1 = np.zeros((4,3))
##R2_val2 = np.zeros((4,3))
##R2_test1 = np.zeros((4,3))
##R2_test2 = np.zeros((4,3))
##Loss_val1 = np.zeros((4,3))
##Loss_val2 = np.zeros((4,3))
##Loss_test1 = np.zeros((4,3))
##Loss_test2 = np.zeros((4,3))
##Estimators = np.zeros((4,1))
##Depth = np.zeros((10,1))
##row = -1
##column = -1
##for n in range(4):
##    row = row+1
##    num = 50+(n*50)
##    Estimators[row,0] = num
##    print("###############",n)
##    column = -1
##    for d in range(3):
##        column = column+1
##        depth = 4+(2*n)
##        Depth[column,0] = depth
##        reg_RF = RandomForestRegressor(n_estimators=num,max_depth=8,random_state=0,n_jobs=-1,verbose=1)
##        reg_RF.fit(X_train1,y_train1)
##        pred_RF_val1 = reg_RF.predict(X_val1)
##        r2_val1 = reg_RF.score(X_val1,y_val1)
##        
##        reg_RF.fit(X_train2,y_train2)
##        pred_RF_val2 = reg_RF.predict(X_val2)
##        r2_val2 = reg_RF.score(X_val2,y_val2)
##        
##        X_train_all1  = np.concatenate([X_train1,X_val1],axis=0)
##        y_train_all1 = np.concatenate([y_train1,y_val1])
##        X_train_all2 = np.concatenate([X_train2,X_val2],axis=0)
##        y_train_all2 = np.concatenate([y_train2,y_val2])
##        
##        reg_RF.fit(X_train_all1,y_train_all1)
##        pred_test1 = reg_RF.predict(X_test1)
##        r2_test1 = reg_RF.score(X_test1,y_test1)
##        
##        reg_RF.fit(X_train_all2,y_train_all2)
##        pred_test2 = reg_RF.predict(X_test2)
##        r2_test2 = reg_RF.score(X_test2,y_test2)
##        
##        L_1_val = np.absolute(pred_RF_val1-y_val1)*0.5
##        L_2_val = np.absolute(pred_RF_val2-y_val2)*0.5
##        L_1_test = np.absolute(pred_test1-y_test1)*0.5
##        L_2_test = np.absolute(pred_test2-y_test2)*0.5
##        score_val1 = (1/y_test1.shape[0])*np.sum(np.multiply(X_test1[:,4],L_1_test),axis=0)
##        score_val2 = (1/y_test2.shape[0])*np.sum(np.multiply(X_test2[:,4],L_2_test),axis=0)
##        score_test1 = (1/y_test1.shape[0])*np.sum(np.multiply(X_test1[:,4],L_1_test),axis=0)
##        score_test2 = (1/y_test2.shape[0])*np.sum(np.multiply(X_test2[:,4],L_2_test),axis=0)
##        
##        R2_val1[row][column] = r2_val1
##        R2_val2[row][column] = r2_val2
##        R2_test1[row][column] = r2_test1
##        R2_test2[row][column] = r2_test2
##        
##        Loss_val1[row][column] = score_val1
##        Loss_val2[row][column] = score_val2
##        Loss_test1[row][column] = score_test1
##        Loss_test2[row][column] = score_test2
##
##plt.figure()
##colors = ["lightskyblue","skyblue","lightsteelblue","steelblue","dodgerblue","cornflowerblue","royalblue","blue","mediumblue","darkblue"]
##labels = ["max depth = 4","max depth = 6","max depth = 8","max depth = 10","max depth = 12","max depth = 14","max depth = 16","max depth = 18","max_depth = 20","max_depth = 22"]
##for i in range(25):
##    for j in range(10):
##        print("yes")
##        plt.plot(Estimators[i],Loss_val1[i][j],c=colors[j],marker="*",label=labels[j])
##plt.xlabel("Number of Estimators")
##plt.ylabel("Loss")
##plt.legend()
##plt.show()
##
##plt.figure()
##colors = ["lightskyblue","skyblue","lightsteelblue","steelblue","dodgerblue","cornflowerblue","royalblue","blue","mediumblue","darkblue"]
##labels = ["max depth = 4","max depth = 6","max depth = 8","max depth = 10","max depth = 12","max depth = 14","max depth = 16","max depth = 18","max_depth = 20","max_depth = 22"]
##for i in range(25):
##    for j in range(10):
##        plt.plot(Estimators[i],Loss_val2[i][j],c=colors[j],marker="*",label=labels[j])
##plt.xlabel("Number of Estimators")
##plt.ylabel("Loss")
##plt.legend()
##plt.show()
#loss_RF = pd.read_csv("Loss_RF.csv")
#loss_ET = pd.read_csv("Loss_ET.csv")
#pred_RF = pd.read_csv("predictions_RF.csv")
#pred_ET = pd.read_csv("predictions_ET.csv")
#RF = loss_RF.to_numpy()
#ET = loss_ET.to_numpy()
#y_pred1 = pred_ET["pred_1"]
#y_pred2 = pred_ET["pred_2"]
#dates = y_dates[::2]
#y_dates = dates[0:14]
#y_dates = pd.concat([y_dates]*3463, ignore_index=True)
#df = features_training.loc[features_training["Date"]>94]
#cases = df.loc[df["Target"]==0]
#cases_date = cases.groupby(["Date"],as_index=False)["TargetValue"].sum()
#predicted_cases = pd.concat([y_dates,y_pred1],axis=1)
#predicted_cases_date = predicted_cases.groupby(["Date"], as_index=False)["pred_1"].sum()
#fatalities = df.loc[df["Target"]==1]
#fatalities_date = fatalities.groupby(["Date"],as_index=False)["TargetValue"].sum()
#predicted_fatalities = pd.concat([y_dates,y_pred2],axis=1)
#predicted_fatalities_date = predicted_fatalities.groupby(["Date"], as_index=False)["pred_2"].sum()
##DF_cases = DF_cases.assign(Date=y_dates)
##DF_fatalities = pd.concat([fatalities,y_dates,y_pred2],axis=1)
#plt.figure()
#plt.plot_date(x=predicted_fatalities_date["Date"], y=predicted_fatalities_date["pred_2"], xdate=True, fmt="r-",label="Predictions")
#plt.plot_date(x=predicted_fatalities_date["Date"], y=fatalities_date["TargetValue"], xdate=True, fmt="b-",label="Actual")
#plt.title("Comparison of predicted fatalities and actual fatalities")
#plt.legend()
#plt.ylabel("Fatalities")
#plt.grid(True)
#plt.gcf().autofmt_xdate()
#plt.show()
#plt.savefig("Fatalities_Comparison.eps")
#
#plt.figure()
#plt.plot_date(x=predicted_cases_date["Date"], y=predicted_cases_date["pred_1"], xdate=True, fmt="r-",label="Predictions")
#plt.plot_date(x=predicted_cases_date["Date"], y=cases_date["TargetValue"], xdate=True, fmt="b-",label="Actual")
#plt.title("Comparison of predicted cases and actual cases")
#plt.legend()
#plt.ylabel("Fatalities")
#plt.grid(True)
#plt.gcf().autofmt_xdate()
#plt.show()
#plt.savefig("Cases_Comparsion.eps")
#
##fig = px.line(x=y_dates, y=cases["TargetValue"])
##fig.write_image("Mortality_Rate_State_Map.eps")
##plot(fig)
#
##plt.figure()
##plt.plot(RF[0:3,0],RF[0:3,2],"b*",label="Confirmed Cases, max depth = 4")
##plt.plot(RF[4:7,0],RF[4:7,2],"r*",label="Confirmed Cases, max depth = 6")
##plt.plot(RF[8:11,0],RF[8:11,2],"g*",label="Confirmed Cases, max depth = 8")
##plt.plot(RF[12:15,0],RF[12:15,2],"c*",label="Confirmed Cases, max depth = 10")
##plt.xlabel("Number of Estimators")
##plt.ylabel("Validation Score")
##plt.legend()
##plt.show()
##plt.savefig("RF_validation1.eps")
##
##plt.figure()
##plt.plot(RF[0:3,0],RF[0:3,3],"bs",label="Fatalities, max depth = 4")
##plt.plot(RF[4:7,0],RF[4:7,3],"rs",label="Fatalities, max depth = 6")
##plt.plot(RF[8:11,0],RF[8:11,3],"gs",label="Fatalities, max depth = 8")
##plt.plot(RF[12:15,0],RF[12:15,3],"cs",label="Fatalities, max depth = 10")
##plt.xlabel("Number of Estimators")
##plt.ylabel("Validation Score")
##plt.legend()
##plt.show()
##plt.savefig("RF_validation2.eps")
##
##plt.figure()
##plt.plot(ET[:,0],ET[:,1],"b*",label="Confirmed Cases")
##plt.plot(ET[:,0],ET[:,2],"r*",label="Fatalities")
##plt.xlabel("Number of Estimators")
##plt.ylabel("Validation Score")
##plt.legend()
##plt.show()
##plt.savefig("ET_validation.eps")
#
