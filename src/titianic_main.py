import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing


path = '../resources/titanic/'
file_ext = '.csv'
file_train_ds = path + 'train' + file_ext
file_test_ds = path + 'test' + file_ext
file_ans_ds = path + 'gender_submission' + file_ext

df = pd.DataFrame(pd.read_csv(file_train_ds))
df_test = pd.DataFrame(pd.read_csv(file_test_ds))
df_ans = pd.DataFrame(pd.read_csv(file_ans_ds))

print(df.shape)
# print(tabulate(df.head(10), headers='keys'))
# df.info()
# print(df.isnull().sum())
# print(df_test.isnull().sum())


def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)


def show_fare_passenger_ratio():
    binwidth = 20
    df_class_list = [None] * 3
    for i in range(3):
        df_class_list[i] = dfClass = df.loc[df['Pclass'] == (i+1)]
        dfClass['Fare'].plot.hist(bins=range(0,540 + binwidth, binwidth), alpha=0.3)
    plt.legend(["First", "Second", "Third"])
    plt.xlabel("Fare Price")
    plt.ylabel("No. of Passengers")
    plt.show()


def show_class_survival_rate():
    print("Survival Rate:")
    survival_rate = [[None] * 3 for i in range(3)]
    for i in range(3):
        dfClass = df.loc[df['Pclass'] == (i + 1)]
        survived = dfClass['Survived'].value_counts()
        survival_rate[0][i] = survived[0] # Dead
        survival_rate[1][i] = survived[1] # Survived
        survival_rate[2][i] = survived[1] / (survived[1] + survived[0])
        print("Class {:d}: {:d}/{:d} \t{:.2f}%".format(i+1, survived[1], len(dfClass), round((survived[1]/len(dfClass)*100), 2)))
    df_s_rate = pd.DataFrame(survival_rate)
    df_s_rate.rename(columns={0: "First", 1: "Second", 2: "Third"}, inplace=True)
    print("pause")


def cal_pca():
    sns.set('talk')
    sns.set_style('ticks')

    # Cleaning Data
    # Delete unimportant columns
    dfc = df.copy()
    del dfc['PassengerId']
    del dfc['Name']
    del dfc['Ticket']
    del dfc['Embarked']
    del dfc['Cabin']
    # Male = 0, Female = 1
    dfc.loc[dfc["Sex"] == "male", "Sex"] = 0
    dfc.loc[dfc["Sex"] == "female", "Sex"] = 1
    nan_count = len(dfc.loc[dfc["Age"].isnull()])
    nan_death_count = len(dfc.loc[(dfc["Age"].isnull()) & (dfc["Survived"] == 0)])
    nan_death_ratio = round(nan_death_count / nan_count, 2)
    # Delete Null Rows
    dfc_nonull = dfc.loc[dfc["Age"]>=(-1)]

    pca = PCA()
    y = dfc_nonull['Survived']
    del dfc_nonull['Survived']
    X = dfc_nonull

    dfc_nonull = pca.fit_transform(X=X)
    dfc_nonull = pd.DataFrame(dfc_nonull)
    dfc_nonull.round(2)
    # print(tabulate(dfc_nonull.head(10), headers='keys'))

    dfc_nonull_loadings = pd.DataFrame(pca.components_)

    fig, axis = plt.subplots(figsize=(12,6))
    sns.heatmap(dfc_nonull_loadings, annot=True, ax=axis, cmap='RdBu')
    axis.set_xticklabels(X.columns)
    axis.set_yticklabels(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    plt.show()


def age_survival_rate():
    binwidth = 2
    df_dead = df.loc[df["Survived"] == 0]
    df_alive = df.loc[df["Survived"] == 1]
    df_dead['Age'].plot.hist(bins=range(0, 90 + binwidth, binwidth), label="Dead", alpha=0.5)
    df_alive['Age'].plot.hist(bins=range(0, 90 + binwidth, binwidth), label="Survived", alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Age")
    plt.ylabel("No. of Passengers")
    plt.show()


def sex_survival_rate():
    df_str = df.copy()
    df_str.Survived = df_str.Survived.astype(str)
    df_str.loc[df_str["Survived"] == '0'] = "Dead"
    df_str.loc[df_str["Survived"] == '1'] = "Survived"
    df_m = df_str.loc[df["Sex"] == "male"]
    df_f = df_str.loc[df["Sex"] == "female"]
    uni_vals = ["Dead", "Survived"]
    list_m = df_m.Sex.to_list()
    list_f = df_f.Sex.to_list()
    counts_m = [list_m.count(value) for value in uni_vals]
    counts_f = [list_f.count(value) for value in uni_vals]
    plt.figure(figsize=(5, 7))
    plt.ylabel("No. of Passengers")
    plt.bar(np.arange(len(uni_vals))-0.2, counts_m, 0.4)
    plt.bar(np.arange(len(uni_vals))+0.2, counts_f, 0.4)
    plt.xticks(np.arange(len(uni_vals)), uni_vals)
    plt.legend(["Male", "Female"], loc='upper right')
    plt.show()
    print("pause")


def add_ans_to_test_ds(df_t):
    df_ans_copy = df_ans.copy()
    df_ans_copy["Pclass"] = df_t["Pclass"]
    df_ans_copy["Name"] = df_t["Name"]
    df_ans_copy["Sex"] = df_t["Sex"]
    df_ans_copy["Age"] = df_t["Age"]
    df_ans_copy["SibSp"] = df_t["SibSp"]
    df_ans_copy["Parch"] = df_t["Parch"]
    df_ans_copy["Ticket"] = df_t["Ticket"]
    df_ans_copy["Fare"] = df_t["Fare"]
    df_ans_copy["Cabin"] = df_t["Cabin"]
    df_ans_copy["Embarked"] = df_t["Embarked"]
    df_ans_copy = df_ans_copy.loc[df_ans_copy["Fare"] >= -1]
    return df_ans_copy


def del_missing_age(df_c):
    df_d = df_c.copy()
    df_d = df_d.loc[df_d["Age"] >= (-1)]
    return df_d


def replace_categorical_to_num(df_c):
    df_r = df_c.copy()
    df_r.loc[df_r["Embarked"].isnull(), "Embarked"] = 'O'
    df_r['Embarked'] = df_r['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'O': 3}).astype(int)
    df_r["Sex"] = df_r["Sex"].map({'male': 0, 'female': 1}).astype(int)
    return df_r


def fill_missing_age():
    # train data
    dfn_train = df.copy()
    dfn_train = replace_categorical_to_num(dfn_train)
    del dfn_train['PassengerId']
    del dfn_train['Name']
    del dfn_train['Ticket']
    del dfn_train['Cabin']
    dfn_train_no_age = dfn_train.loc[dfn_train["Age"].isnull()]
    dfn_train = del_missing_age(dfn_train)

    X_dfn_train_no_age = dfn_train_no_age.drop("Age", axis="columns")
    X_train = dfn_train.drop("Age", axis="columns")
    y_train = dfn_train["Age"]


    # test data
    dfn_test = df_test.copy()
    dfn_test = add_ans_to_test_ds(dfn_test)
    dfn_test = replace_categorical_to_num(dfn_test)
    del dfn_test['PassengerId']
    del dfn_test['Name']
    del dfn_test['Ticket']
    del dfn_test['Cabin']
    dfn_test_no_age = dfn_test.loc[dfn_test["Age"].isnull()]
    dfn_test = del_missing_age(dfn_test)

    X_dfn_test_no_age = dfn_test_no_age.drop("Age", axis="columns")
    X_test = dfn_test.drop("Age", axis="columns")
    y_test = dfn_test["Age"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    predictions_train_set = model.predict(X_dfn_train_no_age)
    predictions_train_set = [x if x >= 0.0 else 0.0 for x in predictions_train_set]
    predictions_test_set = model.predict(X_dfn_test_no_age)
    predictions_test_set = [x if x >= 0.0 else 0.0 for x in predictions_test_set]

    dfn_train_no_age["Age"] = predictions_train_set
    dfn_test_no_age["Age"] = predictions_test_set

    dfn_train_predict_age = pd.concat([dfn_train, dfn_train_no_age])
    dfn_test_predict_age = pd.concat([dfn_test, dfn_test_no_age])

    return [dfn_train_predict_age, dfn_test_predict_age]


def logistic_regression():
    df_both = fill_missing_age()
    df_log_train = df_both[0].copy()
    X_df_log_train = df_log_train.drop("Survived", axis=1)
    y_df_log_train = df_log_train["Survived"]

    df_log_test = df_both[1].copy()
    X_df_log_test = df_log_test.drop("Survived", axis=1)
    y_df_log_test = df_log_train["Survived"]

    logreg = LogisticRegression()
    logreg.fit(X_df_log_train, y_df_log_train)
    y_prediction = logreg.predict(X_df_log_test)
    accuracy_logreg = round(logreg.score(X_df_log_train, y_df_log_train), 4)

    print("Accuracy:", accuracy_logreg)

    df_coeff = pd.DataFrame(df_log_train.columns.delete(0))
    df_coeff.columns = ['Feature']
    df_coeff["Correlation"] = pd.Series(logreg.coef_[0])
    df_coeff.sort_values(by='Correlation', ascending=True)
    print(df_coeff)

def knn():
    df_both = fill_missing_age()
    df_log_train = df_both[0].copy()
    X_df_log_train = df_log_train.drop("Survived", axis=1)
    y_df_log_train = df_log_train["Survived"]

    df_log_test = df_both[1].copy()
    X_df_log_test = df_log_test.drop("Survived", axis=1)
    y_df_log_test = df_log_train["Survived"]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_df_log_train, y_df_log_train)
    y_prediction = knn.predict(X_df_log_test)
    accuracy_knn = round(knn.score(X_df_log_train, y_df_log_train), 4)
    print("Accuracy:", accuracy_knn)

show_fare_passenger_ratio()
show_class_survival_rate()
cal_pca()
age_survival_rate()
sex_survival_rate()
logistic_regression()
knn()