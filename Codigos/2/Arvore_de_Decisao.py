import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

dados = pd.read_csv('..\..\Dados\Treinamento\hepatitis.data.train.csv', index_col=0)
dados = dados.replace('?', 0)
dados = dados.astype('float64')

X = dados.iloc[:,0:18].values
y = np.ravel(dados.iloc[:,[18]].values)




#código copiado e adaptado apartir da documentação do scikit learn,
#url: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html, Classification and ROC analysis:


#Run classifier with cross-validation and plot ROC curves
sss = StratifiedShuffleSplit(
    n_splits=10,
    test_size=0.1,
)
classifier = DecisionTreeClassifier() #Utiliza uma modificação do algoritmo CART

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax1 = plt.subplots()
ar = np.ndarray(
    shape = (2,2),
    dtype = int,
    buffer=np.array([[0,0],[0,0]])
)
for i, (train, test) in enumerate(sss.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax1,
    )
    cm = confusion_matrix(y[test], classifier.predict(X[test]))
    ar += cm
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax1.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

viz_m = ConfusionMatrixDisplay(
    confusion_matrix = ar,
    display_labels = classifier.classes_
    )
viz_m.plot()

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax1.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax1.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax1.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic curve",
)
ax1.legend(loc="lower right")
plt.show()