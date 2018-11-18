def pre_process_data (X, y):
    familes_died = get_family_died(X, y)
    families_survived = get_family_survived(X, y)
    

def pre_process (df, x, y = None, allow_add_column = True):
    
    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
    df['Cabin'] = df['Cabin'].dropna().apply(lambda x: x[0])
    df['Name'] = df['Name'].apply(lambda x: x.split(',')[0])
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Fare'] = df['Fare'].fillna(7)
    df_train = df[['Pclass', 'Age', 'SibSp', 'Fare']]
    
    cabins = pd.get_dummies(df['Cabin'], drop_first = True)
    male = pd.get_dummies(df['Sex'], drop_first = True)
    df_train = pd.concat([df_train, male, cabins], axis = 1)
    df_train['Pclass'] = df['Pclass'].apply(lambda x: x**2)
    df_train['Fare'] = df['Fare'].apply(lambda x: x**1.5)
    df_train['Minor'] = df[['Age', 'Name']].apply(def_minor, axis = 1)
    df_train['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])
    df_train['IsAlone'] = ((df['SibSp'] + df['Parch']) == 0).apply(lambda x: 1 if x else 0)
    df_train['Old'] = (((df['Age'] >= 70) & (df['Sex'] == 'male')) | ((df['Age'] > 70) & (df['Sex'] == 'female'))).apply(int)
    if allow_add_column:
        df_result = pd.concat([df_train[x], male, cabins], axis = 1)
    else:
        df_result = df_train[x]
        
    if y == None:
        return df_result, None
    else:
        return df_result, df[y]

def process_data(df, x, y = None, values = None):
    df_train, df_class = pre_process(df, x, y)
    if values == None:
        values = []
        values.append(get_family_died(df_train, df_class))
        values.append(get_family_survived(df_train, df_class))

    df_train = add_features(df_train, ['Surname', 'Surname'], ['SurnameAllDied', 'SurnameOneLived'], values)
    df_train.drop('Surname', inplace = True, axis = 1)
    
    df_train.drop(['SurnameAllDied'], inplace = True, axis = 1)
    df_train.drop(['SurnameOneLived'], inplace = True, axis = 1)
    return df_train, df_class, values

def classification_model_print(model, x_train, x_test, y_train, y_test):
    y_pred_test = model.predict(x_test)
    y_pred = model.predict(x_train)
    
    print("train set: \n")
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))
    print("\n<------------------------------>\ntest set: \n")
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

def classification_model_save(df, x, y, columns_save, file_name, model, file_extension = '.csv', set_name_timestamp = True, ensure_columns = [], values_new_features = [], scaler = None, values = []):
    
    df_test = pd.read_csv(file_name + file_extension)

    df_proc, df_class, values = process_data(df = df_test, x = x, values = values)
    
    for c in ensure_columns:
        if c not in df_proc.columns:
            df_proc[c] = 0
    
    if scaler != None:
        df_proc = scaler.transform(df_proc)
        
    if set_name_timestamp:
        comp = str(time.time()).split('.')[0]
    else:
        comp = ''

    y_pred = model.predict(df_proc)
    df_result = df_test[columns_save].copy()
    df_result[y] = y_pred
    df_result.to_csv(str(file_name + '_' + comp + file_extension), index = False)
    print(str("arquivo " + file_name + comp + file_extension + " salvo com sucesso"))

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

def scale (X, scaler = None):
    if scaler == None:
        from sklearn.preprocessing import StandardScaler  
        scaler = StandardScaler()
        scaler.fit(X)

    X = scaler.transform(X)  
    return X, scaler

def def_minor (cols):
    Age = cols[0]
    Name = cols[1]
    
    if "Master" in Name:
        return 0
    
    if Age < 16:
        return 1
    else:
        return 0

def get_family_died(X, y):
    df = pd.concat([X, y], axis = 1)

    grp = df.groupby('Surname').count()
    list_names = df['Surname'].value_counts().index
    values = []

    for item in list_names:
        count = df[df['Surname'] == item]['Survived'].value_counts()
        if (1 in count.index):
            if (int(count.loc[1] >= 3)):
                values.append(item)
    return values

def get_family_survived(X, y):
    count = X.groupby('Surname').count()
    lived = []
    for name in count.index:
        if (count['Pclass'].loc[name] > 3) & len(X[(X['Surname'] == name) & (y == 1)]) > 0:
            lived.append(name)
            
    return lived

def add_features(df, variables, features, values):
    for idx, value in enumerate(values):
        df[features[idx]] = 0
        for v in value:
            df.loc[df[variables[idx]] == v, features[idx]] = 1
    return df