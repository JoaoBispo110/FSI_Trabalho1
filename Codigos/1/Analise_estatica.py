import pandas as pd
import numpy as np

dados = pd.read_csv('..\..\Dados\Treinamento\hepatitis.data.train.csv', index_col=0)
dados = dados.replace('?', np.NaN)
dados = dados.astype('float64')

medias = dados.iloc[:,0:18].mean(numeric_only = True, skipna = True)
desvios = dados.iloc[:,0:18].std(numeric_only = True, skipna = True)
print("Medias:")
print(medias)
print("\n\n")
print("Desvios:")
print(desvios)