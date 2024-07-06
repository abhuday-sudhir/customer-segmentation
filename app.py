from flask import Flask,render_template,request,redirect,jsonify
import os
from joblib import load
import json

import pandas as pd
import seaborn as sns 
import matplotlib

matplotlib.use('Agg')#Use a  not interactive backend  coz it is causing a problem in macOS

import matplotlib.pyplot as plt
from datetime import datetime
import openpyxl

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

application=Flask(__name__) 

app=application

model =load('model.joblib')


def load_and_preprocess(filepath):
    data=pd.read_excel(filepath)
    data['CustomerID']=data['CustomerID'].astype(str)
    data['Amount']=data['Quantity']*data['UnitPrice']
    df1=data.groupby('CustomerID')['Amount'].sum().reset_index()
    df2=data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    df2.columns=['CustomerID','Frequency']
    df=pd.merge(df1,df2,on='CustomerID',how='inner')
    max_date=max(data['InvoiceDate'])
    data['Diff']=max_date-data['InvoiceDate']
    df3=data.groupby('CustomerID')['Diff'].min().reset_index()
    df3.columns=['CustomerID','Most_Recent_Engagement']
    df3['Most_Recent_Engagement']=df3['Most_Recent_Engagement'].dt.days
    df_merged=pd.merge(df,df3,on="CustomerID",how='inner')

    #Removing outliers for Amount
    q1=df_merged.Amount.quantile(0.05)
    q2=df_merged.Amount.quantile(0.95)
    iqr=q2-q1
    df_merged=df_merged[(df_merged.Amount>=q1-1.5*iqr) & (df_merged.Amount<=q2+1.5*iqr)]

    #Removing outliers for Frequency
    q1=df_merged.Frequency.quantile(0.05)
    q2=df_merged.Frequency.quantile(0.95)
    iqr=q2-q1
    df_merged=df_merged[(df_merged.Frequency>=q1-1.5*iqr) & (df_merged.Frequency<=q2+1.5*iqr)]

    #Removing outliers for Most_Recent_Engagement
    q1=df_merged.Most_Recent_Engagement.quantile(0.05)
    q2=df_merged.Most_Recent_Engagement.quantile(0.95)
    iqr=q2-q1
    df_merged=df_merged[(df_merged.Most_Recent_Engagement>=q1-1.5*iqr) & (df_merged.Most_Recent_Engagement<=q2+1.5*iqr)]

    df_merged=df_merged[['Amount','Frequency','Most_Recent_Engagement']]
    scaler=StandardScaler()
    df_merged_scaled=scaler.fit_transform(df_merged)
    df_merged_scaled=pd.DataFrame(df_merged_scaled)
    df_merged_scaled.columns=['Amount','Frequency','Most_Recent_Engagement']

    return  df_merged,df_merged_scaled

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'})
    
    
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    try:
        segmented_df,scaled_df=load_and_preprocess(file_path)
        segmented_df['Cluster']=model.predict(scaled_df)

        plt.ion()
        plt.figure(figsize=(10,10))
        sns.boxplot(x='Cluster',y='Amount', data= segmented_df,hue='Cluster')
        amount_img_path = 'static/Cluster_Amount.png'
        os.makedirs(os.path.dirname(amount_img_path), exist_ok=True)
        plt.savefig(amount_img_path)
        plt.clf()
        plt.ioff()

        plt.ion()
        plt.figure(figsize=(10,10))
        sns.boxplot(x='Cluster',y='Frequency', data= segmented_df,hue='Cluster')
        freq_img_path = 'static/Cluster_Frequency.png'
        os.makedirs(os.path.dirname(freq_img_path), exist_ok=True)
        plt.savefig(freq_img_path)
        plt.clf()
        plt.ioff()
        
        plt.ion()
        plt.figure(figsize=(10,10))
        sns.boxplot(x='Cluster',y='Most_Recent_Engagement', data= segmented_df,hue='Cluster')
        recency_img_path = 'static/Cluster_Recency.png'
        os.makedirs(os.path.dirname(recency_img_path), exist_ok=True)
        plt.savefig(recency_img_path)
        plt.clf()
        plt.ioff()

        response = {'amount_img': amount_img_path,
            'freq_img': freq_img_path,
            'recency_img': recency_img_path}
        return json.dumps(response)

    except Exception as e:
        raise jsonify({'error':str(e)})


    
if __name__=="__main__":
    app.run(debug=True,port=5000)