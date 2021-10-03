import os
import pdb
import pandas as pd
import shutil
test = pd.read_csv('covidData/test.txt', sep=' ', header=None)
test.columns = ['patient_id', 'filename', 'outcome', 'location']

#os.mkdir('dataset/')
#os.mkdir('dataset/covid19_positive')
#os.mkdir('dataset/covid19_negative')

for i, row in test.iterrows():
	shutil.copy('covidData/test/'+row['filename'], 'dataset/covid19_'+row['outcome']+'/'+row['filename'])


train = pd.read_csv('covidData/train.txt', sep=' ', header=None)
train.columns = ['patient_id', 'filename', 'outcome', 'location']

for i, row in train.iterrows():
	shutil.copy('covidData/train/'+row['filename'], 'dataset/covid19_'+row['outcome']+'/'+row['filename'])