import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder

# Read the data
data = pd.read_csv('train.csv')

# get number for rows and columns
print("Number of data instances and features/attributes", data.shape)

# Remove nan in 'Embarked' attribute # it has 2 nan values
data.dropna(inplace=True, subset=['Embarked'])

# Build a imputer to replace nan values
# strategy can be : 'mean' or 'median'
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Apply imputer on 'Age' attribute
data.loc[:, ['Age']] = imputer.fit_transform(data['Age'].values.reshape(-1, 1))

# Build encoder
enc = LabelEncoder()
# modify 'Sex' attribute i.e male and female to 0 and 1
enc.fit(data['Sex'])
data.loc[:, 'Sex'] = enc.transform(data['Sex'])
# modify 'Embarked' attribute i.e  S,C and Q to 0, 1 and 2
enc.fit(data['Embarked'])
data.loc[:,'Embarked'] = enc.transform(data['Embarked'])

# Remove unusable attributes
drop_columns = ['PassengerId','Cabin', 'Name', 'Ticket', 'Fare']
data.drop(drop_columns, axis=1, inplace=True)

# save data in csv
data.to_csv("preprocessed_data.csv", index=False)

