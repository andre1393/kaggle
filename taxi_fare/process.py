import numpy as np
from sklearn.model_selection import train_test_split

def clear_data(df):
	# deleta as entidades com longitude fora do padrao
	df = df[(df['pickup_longitude'] > -80) & (df['pickup_longitude'] < -65)]
	df = df[(df['dropoff_longitude'] > -80) & (df['dropoff_longitude'] < -65)]
	
	# deleta as entidades com latitude fora do padrao
	df = df[(df['pickup_latitude'] < 50) & (df['pickup_latitude'] > 30)]
	df = df[(df['dropoff_latitude'] < 50) & (df['dropoff_latitude'] > 30)]
	
	#deleta as entidades com valores impossiveis
	df = df[df['fare_amount'] > 1]
	
	# deleta as entidades em que a distancia percorrida foi nula
	df = df[df['distance'] > 0]
	
	# deleta as instancias consideradoas outliers
	df = df[((df['fare_amount'] > 200) & (df['distance'] < 10)).apply(lambda x: not x)]
	
	# deleta as instancias com quantidade impossivel de passageiros
	df = df[df['passenger_count'] < 7]
	df = df[df['passenger_count'] > 0]
	
	return df

def add_fields(df, degree = 1, degree_categorical = 1):
	df['distance'] = (112)* np.sqrt((df['pickup_latitude'] - df['dropoff_latitude'])**2 + (df['pickup_longitude'] - df['dropoff_longitude'])**2)

	df['ano'] = df['pickup_datetime'].apply(lambda x: int(x.split('-')[0]))
	min_year = df['ano'].min()
	df['ano'] -= min_year
	df['hora'] = df['pickup_datetime'].apply(lambda x: x.split(' ')[1]).apply(lambda x: int(x.split(':')[0]))
	df['mes'] = df['pickup_datetime'].apply(lambda x: int(x.split('-')[1]))
	df['madrugada'] = df['hora'].apply(set_madrugada)
	df['pico'] = df['hora'].apply(set_pico)
	df['matine'] = df['hora'].apply(set_matine)
	df['ferias'] = df['mes'].apply(set_ferias)
	df['delta_longitude'] = (df['pickup_longitude'] - df['dropoff_longitude']).abs()
	df['delta_latitude'] = (df['pickup_latitude'] - df['dropoff_latitude']).abs()

	df = set_airport(df, degree_categorical)

	degree += 1
	columns = list(df.columns.values)
	for column in columns:
		if np.issubdtype(df[column].dtype, np.number):
			for d in range(2, degree):
				df[column + str(d)] = df[column]**d

	return df

def split_data(df, classe, atributos = None, test_size = 0.3, random_state = 101):
    if atributos == None:
        X = df
    else:
        X = df[atributos]
        
    y = df[classe]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def set_madrugada(x):
	x = int(x)
	if x >= 1 and x <= 6:
		return 1
	else:
		return 0

def set_pico(x):
	x = int (x)
	if x >= 13 and x <= 16:
		return 1
	else:
		return 0

def set_matine(x):
	x = int(x)
	if x >= 18 and x <= 21:
		return 1
	else:
		return 0 

def set_ferias(x):
    x = int(x)
    if (x >= 5 and x <= 6) or (x >= 9 and x <= 12):
        return 1
    else:
        return 0

def set_airport(df, degree_categorical):
	df['jfk'] = 0
	df.loc[(df['pickup_longitude'].between(-73.7841, -73.7721) & df['pickup_latitude'].between(40.6213, 40.6613)) | 
	(df['dropoff_longitude'].between(-73.7841, -73.7721) & df['dropoff_latitude'].between(40.6213, 40.6613)), 'jfk'] = degree_categorical

	df['lga'] = 0
	df.loc[(df['pickup_longitude'].between(-73.8870, -73.8580) & df['pickup_latitude'].between(40.7680, 40.7800)) | 
	(df['dropoff_longitude'].between(-73.8870, -738580) & df['dropoff_latitude'].between(40.7680, 40.7800)), 'lga'] = degree_categorical

	df['ewr'] = 0
	df.loc[(df['pickup_longitude'].between(-74.1920, -74.1720) & df['pickup_latitude'].between(40.6760, 40.70880)) | 
	(df['dropoff_longitude'].between(-74.1920, -74.1720) & df['dropoff_latitude'].between(40.6760, 40.7080)), 'ewr'] = degree_categorical

	return df