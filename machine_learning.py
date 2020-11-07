import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score,  mean_squared_error, r2_score
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, BaggingClassifier, BaggingRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet, SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


class MLAlgorithmsComparison:
    def __init__(self, x, y, test_size: float, dimensionReduction=False):
        if not (type(x) == np.ndarray):
            self.x = x.values
        else:
            self.x = x
        if not (type(y) == np.ndarray):
            self.y = y.values
        else:
            self.y = y
        self.testSize = test_size

        if dimensionReduction == True:
            n_components = int(input('How many dimension?: '))
            lda = LDA(n_components=n_components)
            self.x = lda.fit_transform(x, y)

    def classificationAlgorithms(self):
        algs = [GaussianProcessClassifier,
                QuadraticDiscriminantAnalysis,
                BaggingClassifier,
                ExtraTreesClassifier,
                PassiveAggressiveClassifier,
                ExtraTreeClassifier,
                SGDClassifier,
                GaussianNB,
                BernoulliNB,
                LogisticRegression,
                CatBoostClassifier,
                LGBMClassifier,
                XGBClassifier,
                GradientBoostingClassifier,
                RandomForestClassifier,
                AdaBoostClassifier,
                DecisionTreeClassifier,
                MLPClassifier,
                KNeighborsClassifier,
                SVC]

        results = {}
        acc_trains = []
        acc_tests = []
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        # Algorithm - Data set mismatch control block
        for alg in algs:
            model_name = alg.__name__
            try:
                if model_name == 'CatBoostClassifier':
                    model = alg().fit(X_train, y_train, verbose=False)

                    y_pred_train = model.predict(X_train)
                    acc_train = accuracy_score(y_train, y_pred_train)
                    acc_trains.append(acc_train)

                    y_pred_test = model.predict(X_test)
                    acc_test = accuracy_score(y_test, y_pred_test)
                    acc_tests.append(acc_test)
                    results.__setitem__(model_name, acc_test)
                    continue
                model = alg().fit(X_train, y_train)
            except:
                print(
                    f'Error!! {model_name} Algorithm did not work properly with those given!')
                algs.remove(alg)
                continue

        for alg in algs:
            model_name = alg.__name__
            if model_name == 'MLPClassifier' or model_name == 'SVC':
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)
            if model_name == 'CatBoostClassifier':
                model = alg().fit(X_train, y_train, verbose=False)

                y_pred_train = model.predict(X_train)
                acc_train = accuracy_score(y_train, y_pred_train)
                acc_trains.append(acc_train)

                y_pred_test = model.predict(X_test)
                acc_test = accuracy_score(y_test, y_pred_test)
                acc_tests.append(acc_test)
                results.__setitem__(model_name, acc_test)
                continue

            model = alg().fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_trains.append(acc_train)

            y_pred_test = model.predict(X_test)
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_tests.append(acc_test)

            results.__setitem__(model_name, acc_test)

        print('\n')
        print('\n')
        print('**********************Graphs**********************')
        for i in range(len(algs)):
            SomeMethods.trainTestPlot(
                acc_train=acc_trains[i], acc_test=acc_tests[i], model=algs[i])
            print('Train Accuracy ---->', acc_trains[i])
            print('Test Accuracy ----->', acc_tests[i], '\n')

        print('==================================================')
        for i in results.items():
            print(i)
        print('==================================================')

    def regressionAlgorithms(self):
        algs = [BaggingRegressor,
                ExtraTreesRegressor,
                PassiveAggressiveRegressor,
                ExtraTreeRegressor,
                SGDRegressor,
                CatBoostRegressor,
                LGBMRegressor,
                XGBRegressor,
                GradientBoostingRegressor,
                AdaBoostRegressor,
                RandomForestRegressor,
                DecisionTreeRegressor,
                MLPRegressor,
                KNeighborsRegressor,
                SVR]

        results = {}
        acc_trains = []
        acc_tests = []
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        # Algorithm - Data set mismatch control block
        for alg in algs:
            model_name = alg.__name__
            try:
                if model_name == 'CatBoostRegressor':
                    model = alg().fit(X_train, y_train, verbose=False)

                    y_pred_train = model.predict(X_train)
                    acc_train = r2_score(y_train, y_pred_train)
                    acc_trains.append(acc_train)

                    y_pred_test = model.predict(X_test)
                    acc_test = r2_score(y_test, y_pred_test)
                    acc_tests.append(acc_test)
                    results.__setitem__(model_name, acc_test)
                    continue
                model = alg().fit(X_train, y_train)
            except:
                print(
                    f'Error!! {model_name} Algorithm did not work properly with those given!')
                algs.remove(alg)
                continue

        for alg in algs:

            # modelleme
            model_name = alg.__name__
            if model_name == 'MLPRegressor' or model_name == 'SVR':
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)
            if model_name == 'CatBoostRegressor':
                model = alg().fit(X_train, y_train, verbose=False)

                y_pred_train = model.predict(X_train)
                acc_train = r2_score(y_train, y_pred_train)
                acc_trains.append(acc_train)

                y_pred_test = model.predict(X_test)
                acc_test = r2_score(y_test, y_pred_test)
                acc_tests.append(acc_test)
                results.__setitem__(model_name, acc_test)
                continue

            model = alg().fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            acc_train = r2_score(y_train, y_pred_train)
            acc_trains.append(acc_train)

            y_pred_test = model.predict(X_test)
            acc_test = r2_score(y_test, y_pred_test)
            acc_tests.append(acc_test)

            results.__setitem__(model_name, acc_test)

        print('\n')
        print('\n')
        print('**********************Graphs**********************')
        for i in range(len(algs)):
            SomeMethods.trainTestPlot(
                acc_train=acc_trains[i], acc_test=acc_tests[i], model=algs[i])
            print('Train Accuracy ---->', acc_trains[i])
            print('Test Accuracy ----->', acc_tests[i], '\n')

        print('==================================================')
        for i in results.items():
            print(i)
        print('==================================================')

    def otherRegressionAlgorithms(self):
        algs = [LinearRegression,
                Ridge,
                Lasso,
                ElasticNet]

        results = {}

        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        # Algorithm - Data set mismatch control block
        for alg in algs:
            model_name = alg.__name__
            try:
                model = alg().fit(X_train, y_train)
            except:
                print(
                    f'Error!! {model_name} Algorithm did not work properly with those given!')
                algs.remove(alg)
                continue

        for alg in algs:

            # modelleme
            model_name = alg.__name__

            if model_name == 'LinearRegression':
                model = alg().fit(self.x, self.y)
                y_pred = model.predict(self.x)
                RMSE = np.sqrt(mean_squared_error(self.y, y_pred))
                print(model_name, "Modeli Test Hatasi:", RMSE)
                continue

            model = alg().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
            acc = r2_score(y_test, y_pred)
            results.__setitem__(model_name, acc)

        print('\n')
        print('\n')
        print('==================================================')
        for i in results.items():
            print(i)
        print('==================================================')


class ClassificationAlgorithmsOptimizer:
    def __init__(self, x: pd.DataFrame, y, test_size: float, cv: int, n_jobs: int, dimensionReduction=False):
        self.x = x
        self.y = y

        if dimensionReduction == True:
            n_components = int(input('How many dimension?: '))
            lda = LDA(n_components=n_components)
            self.x = lda.fit_transform(x, y)

        self.dimensionReduction = dimensionReduction
        self.testSize = test_size
        self.cv = cv
        self.n_jobs = n_jobs

    def modelPrediction(self, model_tuned, X_test, y_test, X_train, y_train):
        print(
            f'\n\nTuned Model Train Accuracy ----> {model_tuned.__class__.__name__}')
        y_pred_train = model_tuned.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        print(acc_train)

        y_pred_test = model_tuned.predict(X_test)
        print(
            f'Tuned Model Test Accuracy ----> {model_tuned.__class__.__name__}')

        acc_test = accuracy_score(y_test, y_pred_test)
        print(acc_test)

        return [acc_train, acc_test]

    def extraTree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = ExtraTreeClassifier()
        p = {"criterion": ['gini', 'entropy'],
             "splitter": ['random', 'best'],
             "min_samples_split": [1, 2, 3, 4, 5],
             "min_samples_leaf": [0.5, 1, 2, 3, 5],
             "max_features": ['auto', 'sqrt', 'log2']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = ExtraTreeClassifier(criterion=cv.best_params_['criterion'],
                                          splitter=cv.best_params_['splitter'],
                                          min_samples_split=cv.best_params_[
                                              'min_samples_split'],
                                          min_samples_leaf=cv.best_params_[
                                              'min_samples_leaf'],
                                          max_features=cv.best_params_['max_features']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def adaBoost(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = AdaBoostClassifier()
        p = {"n_estimators": [25, 50, 100, 150, 200, 300],
             "learning_rate": [0.1, 0.5, 1, 2, 3],
             "algorithm": ['SAMME', 'SAMME.R']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = AdaBoostClassifier(n_estimators=cv.best_params_['n_estimators'],
                                         learning_rate=cv.best_params_[
                                             'learning_rate'],
                                         algorithm=cv.best_params_['algorithm']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def sgd(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = SGDClassifier()
        p = {"penalty": ['l2', 'l1', 'elasticnet'],
             "alpha": [0.0001, 0.001, 0.01, 0.1],
             "max_iter": [1000, 2000, 3000]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = SGDClassifier(penalty=cv.best_params_['penalty'],
                                    alpha=cv.best_params_['alpha'],
                                    max_iter=cv.best_params_['max_iter']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def passiveAgg(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = PassiveAggressiveClassifier()
        p = {"C": [1, 2, 3, 5],
             "max_iter": [1000, 2000, 3000]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = PassiveAggressiveClassifier(C=cv.best_params_['C'],
                                                  max_iter=cv.best_params_['max_iter']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def extraTrees(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = ExtraTreesClassifier()
        p = {"n_estimators": [100, 200, 300, 500, 1000],
             "criterion": ['gini', 'entropy'],
             "min_samples_split": [1, 2, 3, 4, 5],
             "min_samples_leaf": [0.5, 1, 2, 3, 5]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = ExtraTreesClassifier(n_estimators=cv.best_params_['n_estimators'],
                                           criterion=cv.best_params_[
                                               'criterion'],
                                           min_samples_split=cv.best_params_[
                                               'min_samples_split'],
                                           min_samples_leaf=cv.best_params_['min_samples_leaf']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def bagging(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = BaggingClassifier()
        p = {"n_estimators": [5, 10, 20, 30, 50, 100, 200],
             "max_samples": [0.5, 1, 2, 3, 4, 5],
             "max_features": [0.5, 1, 2, 3, 4, 5]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = BaggingClassifier(n_estimators=cv.best_params_['n_estimators'],
                                        max_samples=cv.best_params_[
                                            'max_samples'],
                                        max_features=cv.best_params_['max_features']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def logisticRegression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = LogisticRegression()
        p = {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = LogisticRegression(
            solver=cv.best_params_['solver']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def knn(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = KNeighborsClassifier()
        p = {"n_neighbors": np.arange(1, 100)}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = KNeighborsClassifier(n_neighbors=cv.best_params_[
                                           'n_neighbors']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def svm(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        model = SVC()
        p = {"C": np.arange(1, 10),
             "kernel": ['linear', 'rbf', 'sigmoid']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = SVC(C=cv.best_params_['C'],
                          kernel=cv.best_params_['kernel']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def mlpc(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        model = MLPClassifier()
        p = {"alpha": [1, 5, 0.1, 0.01, 0.03, 0.005, 0.0001],
             "hidden_layer_sizes": [(10, 10), (100, 100, 100), (100, 100), (3, 5)]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = MLPClassifier(alpha=cv.best_params_['alpha'],
                                    hidden_layer_sizes=cv.best_params_['hidden_layer_sizes']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def cart(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = DecisionTreeClassifier()
        p = {"max_depth": [1, 3, 5, 8, 10],
             "min_samples_split": [2, 3, 5, 10, 20, 50]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = DecisionTreeClassifier(max_depth=cv.best_params_['max_depth'],
                                             min_samples_split=cv.best_params_['min_samples_split']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def randomForests(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = RandomForestClassifier()
        p = {"max_depth": [5, 8, 10],
             "n_estimators": [200, 500, 1000, 2000],
             "min_samples_split": [2, 5, 10, 20]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = RandomForestClassifier(max_depth=cv.best_params_['max_depth'],
                                             n_estimators=cv.best_params_[
                                                 'n_estimators'],
                                             min_samples_split=cv.best_params_['min_samples_split']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def gbm(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = GradientBoostingClassifier()
        p = {"learning_rate": [0.1, 0.01, 0.001, 0.05],
             "n_estimators": [100, 300, 500, 1000],
             "max_depth": [2, 3, 5, 8],
             "subsample": [1, 0.5, 0.8]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = GradientBoostingClassifier(learning_rate=cv.best_params_['learning_rate'],
                                                 n_estimators=cv.best_params_[
                                                     'n_estimators'],
                                                 max_depth=cv.best_params_[
                                                     'max_depth'],
                                                 subsample=cv.best_params_[
                                                     'subsample']
                                                 ).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def xgboost(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = XGBClassifier()
        p = {"n_estimators": [100, 500, 1000],
             "subsample": [0.6, 0.8, 1],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.1, 0.001, 0.01],
             "colsample_bytree": [0.4, 0.7, 1]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = XGBClassifier(n_estimators=cv.best_params_['n_estimators'],
                                    subsample=cv.best_params_['subsample'],
                                    max_depth=cv.best_params_['max_depth'],
                                    learning_rate=cv.best_params_[
                                        'learning_rate'],
                                    colsample_bytree=cv.best_params_['colsample_bytree']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def lgbm(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = LGBMClassifier()
        model = LGBMRegressor()
        p = {"learning_rate": [0.001, 0.01, 0.1, 0.5, 1],
             "n_estimators": [20, 40, 100, 200, 500, 1000],
             "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = LGBMClassifier(learning_rate=cv.best_params_['learning_rate'],
                                     n_estimators=cv.best_params_[
                                         'n_estimators'],
                                     max_depth=cv.best_params_['max_depth']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def catb(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = CatBoostClassifier()
        p = {"iterations": [200, 500, 1000],
             "learning_rate": [0.01, 0.1],
             "depth": [3, 6, 8]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = CatBoostClassifier(iterations=cv.best_params_['iterations'],
                                         learning_rate=cv.best_params_[
                                             'learning_rate'],
                                         depth=cv.best_params_['depth']).fit(X_train, y_train, verbose=False)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)


class RegressionAlgorithmsOptimizer:
    def __init__(self, x: pd.DataFrame, y, test_size: float, cv: int, n_jobs: int, dimensionReduction=False):
        self.x = x
        self.y = y

        if dimensionReduction == True:
            n_components = int(input('How many dimension?: '))
            lda = LDA(n_components=n_components)
            self.x = lda.fit_transform(x, y)

        self.dimensionReduction = dimensionReduction
        self.testSize = test_size
        self.cv = cv
        self.n_jobs = n_jobs

    def modelPrediction(self, model_tuned, X_test, y_test, X_train, y_train):
        print(
            f'\n\nTuned Model Train Accuracy ----> {model_tuned.__class__.__name__}')
        y_pred_train = model_tuned.predict(X_train)
        acc_train = r2_score(y_train, y_pred_train)
        print(acc_train)

        y_pred_test = model_tuned.predict(X_test)
        print(
            f'Tuned Model Test Accuracy ----> {model_tuned.__class__.__name__}')

        acc_test = r2_score(y_test, y_pred_test)
        print(acc_test)

        return [acc_train, acc_test]

    def extraTree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = ExtraTreeRegressor()
        p = {"criterion": ['mse', 'friedman_mse', 'mae'],
             "splitter": ['random', 'best'],
             "min_samples_split": [1, 2, 3, 4, 5],
             "min_samples_leaf": [0.5, 1, 2, 3, 5],
             "max_features": ['auto', 'sqrt', 'log2']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = ExtraTreeRegressor(criterion=cv.best_params_['criterion'],
                                         splitter=cv.best_params_['splitter'],
                                         min_samples_split=cv.best_params_[
                                             'min_samples_split'],
                                         min_samples_leaf=cv.best_params_[
                                             'min_samples_leaf'],
                                         max_features=cv.best_params_['max_features']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def adaBoost(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = AdaBoostRegressor()
        p = {"n_estimators": [25, 50, 100, 150, 200, 300],
             "learning_rate": [0.1, 0.5, 1, 2, 3],
             "loss": ['linear', 'square', 'exponential']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = AdaBoostRegressor(n_estimators=cv.best_params_['n_estimators'],
                                        learning_rate=cv.best_params_[
                                            'learning_rate'],
                                        loss=cv.best_params_['loss']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def sgd(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = SGDRegressor()
        p = {"penalty": ['l2', 'l1', 'elasticnet'],
             "alpha": [0.0001, 0.001, 0.01, 0.1],
             "max_iter": [1000, 2000, 3000]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = SGDRegressor(penalty=cv.best_params_['penalty'],
                                   alpha=cv.best_params_['alpha'],
                                   max_iter=cv.best_params_['max_iter']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def passiveAgg(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = PassiveAggressiveRegressor()
        p = {"C": [1, 2, 3, 5],
             "max_iter": [1000, 2000, 3000]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = PassiveAggressiveRegressor(C=cv.best_params_['C'],
                                                 max_iter=cv.best_params_['max_iter']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def extraTrees(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = ExtraTreesRegressor()
        p = {"n_estimators": [100, 200, 300, 500, 1000],
             "criterion": ['mse', 'mae'],
             "min_samples_split": [1, 2, 3, 4, 5],
             "min_samples_leaf": [0.5, 1, 2, 3, 5]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = ExtraTreesRegressor(n_estimators=cv.best_params_['n_estimators'],
                                          criterion=cv.best_params_[
                                              'criterion'],
                                          min_samples_split=cv.best_params_[
                                              'min_samples_split'],
                                          min_samples_leaf=cv.best_params_['min_samples_leaf']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def bagging(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = BaggingRegressor()
        p = {"n_estimators": [5, 10, 20, 30, 50, 100, 200],
             "max_samples": [0.5, 1, 2, 3, 4, 5],
             "max_features": [0.5, 1, 2, 3, 4, 5]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = BaggingRegressor(n_estimators=cv.best_params_['n_estimators'],
                                       max_samples=cv.best_params_[
                                           'max_samples'],
                                       max_features=cv.best_params_['max_features']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def knn(self):

        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = KNeighborsRegressor()
        p = {"n_neighbors": np.arange(1, 100)}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = KNeighborsRegressor(n_neighbors=cv.best_params_[
                                          'n_neighbors']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def svm(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        model = SVR()
        p = {"C": np.arange(1, 10),
             "kernel": ['linear', 'rbf', 'sigmoid']}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = SVR(C=cv.best_params_['C'],
                          kernel=cv.best_params_['kernel']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def mlpr(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        model = MLPRegressor()
        p = {"alpha": [1, 5, 0.1, 0.01, 0.03, 0.005, 0.0001],
             "hidden_layer_sizes": [(10, 10), (100, 100, 100), (100, 100), (3, 5)]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = MLPRegressor(alpha=cv.best_params_['alpha'],
                                   hidden_layer_sizes=cv.best_params_['hidden_layer_sizes']).fit(X_train, y_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def cart(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = DecisionTreeRegressor()
        p = {"max_depth": [1, 3, 5, 8, 10],
             "min_samples_split": [2, 3, 5, 10, 20, 50]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = DecisionTreeRegressor(max_depth=cv.best_params_['max_depth'],
                                            min_samples_split=cv.best_params_['min_samples_split']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def randomForests(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = RandomForestRegressor()
        p = {"max_depth": [5, 8, 10],
             "n_estimators": [200, 500, 1000, 2000],
             "min_samples_split": [2, 5, 10, 20]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = RandomForestRegressor(max_depth=cv.best_params_['max_depth'],
                                            n_estimators=cv.best_params_[
                                                'n_estimators'],
                                            min_samples_split=cv.best_params_['min_samples_split']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def gbm(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = GradientBoostingRegressor()
        p = {"learning_rate": [0.1, 0.01, 0.001, 0.05],
             "n_estimators": [100, 300, 500, 1000],
             "max_depth": [2, 3, 5, 8],
             "subsample": [1, 0.5, 0.8]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = GradientBoostingRegressor(learning_rate=cv.best_params_['learning_rate'],
                                                n_estimators=cv.best_params_[
                                                    'n_estimators'],
                                                max_depth=cv.best_params_[
                                                    'max_depth'],
                                                subsample=cv.best_params_[
                                                    'subsample']
                                                ).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def xgboost(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = XGBRegressor()
        p = {"n_estimators": [100, 500, 1000],
             "subsample": [0.6, 0.8, 1],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.1, 0.001, 0.01],
             "colsample_bytree": [0.4, 0.7, 1]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = XGBRegressor(n_estimators=cv.best_params_['n_estimators'],
                                   subsample=cv.best_params_['subsample'],
                                   max_depth=cv.best_params_['max_depth'],
                                   learning_rate=cv.best_params_[
                                       'learning_rate'],
                                   colsample_bytree=cv.best_params_['colsample_bytree']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def lgbm(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = LGBMRegressor()
        p = {"learning_rate": [0.001, 0.01, 0.1, 0.5, 1],
             "n_estimators": [20, 40, 100, 200, 500, 1000],
             "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = LGBMRegressor(learning_rate=cv.best_params_['learning_rate'],
                                    n_estimators=cv.best_params_[
                                        'n_estimators'],
                                    max_depth=cv.best_params_['max_depth']).fit(X_train, y_train)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)

    def catb(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = CatBoostRegressor()
        p = {"iterations": [200, 500, 1000],
             "learning_rate": [0.01, 0.1],
             "depth": [3, 6, 8]}
        cv = GridSearchCV(model, p, cv=self.cv,
                          n_jobs=self.n_jobs, verbose=2).fit(X_train, y_train)

        model_tuned = CatBoostRegressor(iterations=cv.best_params_['iterations'],
                                        learning_rate=cv.best_params_[
                                            'learning_rate'],
                                        depth=cv.best_params_['depth']).fit(X_train, y_train, verbose=False)

        if not self.dimensionReduction == True:
            SomeMethods.featureImportances(model_tuned, X_train)

        plot = self.modelPrediction(
            model_tuned, X_test, y_test, X_train, y_train)

        SomeMethods.trainTestPlot(
            acc_train=plot[0], acc_test=plot[1], model=model_tuned)

        SomeMethods.save(model_tuned)


class OtherRegressionAlgorithmsOptimizer:
    def __init__(self, x: pd.DataFrame, y, test_size: float, cv: int, n_jobs: int):
        self.x = x
        self.y = y
        self.testSize = test_size
        self.cv = cv
        self.n_jobs = n_jobs

    def modelPrediction(self, model_tuned: object, X_test, y_test):
        y_pred = model_tuned.predict(X_test)
        print(f'Tuned Model ----> {model_tuned.__class__.__name__}')
        print(np.sqrt(mean_squared_error(y_test, y_pred)))

    def linear(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        model = LinearRegression()
        from sklearn.model_selection import cross_val_score
        print('Tuned Model')
        print(np.sqrt(np.mean(-cross_val_score(model, X_train,
                                               y_train, cv=10, scoring="neg_mean_squared_error"))))

    def ridge(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)
        lambdas = 10**np.linspace(10, -2, 100)* 0.5

        ridgeCv = RidgeCV(
            alphas=lambdas, scoring="neg_mean_squared_error", cv=10, normalize=True)
        ridgeCv.fit(X_train, y_train)

        model_tuned = Ridge(alpha=ridgeCv.alpha_).fit(X_train, y_train)

        self.modelPrediction(model_tuned, X_test, y_test)

        SomeMethods.save(model_tuned)

    def lasso(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)


        alphas = 10**np.linspace(10, -2, 100) * 0.5

        lasso_cv_model = LassoCV(alphas=alphas, cv=10,
                                 max_iter=100000).fit(X_train, y_train)

        model_tuned = Lasso().set_params(alpha=lasso_cv_model.alpha_).fit(X_train, y_train)

        self.modelPrediction(model_tuned, X_test, y_test)

        SomeMethods.save(model_tuned)

    def elasticNet(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=self.testSize,
                                                            random_state=0)

        alphas = 10**np.linspace(10, -2, 100) * 0.5

        enet_cv_model = ElasticNetCV(
            alphas=alphas, cv=10).fit(X_train, y_train)

        model_tuned = ElasticNet(
            alpha=enet_cv_model.alpha_).fit(X_train, y_train)

        self.modelPrediction(model_tuned, X_test, y_test)

        SomeMethods.save(model_tuned)

class SomeMethods:
    @staticmethod
    def featureImportances(model_tuned: object, X_train):
        feature_imp = pd.Series(model_tuned.feature_importances_,
                                index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(16, 8))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Scores')
        plt.ylabel('Features')
        plt.title(
            f'Feature Importances ----> {model_tuned.__class__.__name__}')
        plt.show()
        print("\n")

    @staticmethod
    def save(model_tuned):
        save = int(input(f"""
        ------------------------------------------------------------------
        {model_tuned.__class__.__name__}
        Do you want the model you optimized to be saved?\n
        Save       ----> 1\n
        Don't Save ----> 0\n
        """))
        if save == 1:
            import pickle
            dosya = "model.save"
            pickle.dump(model_tuned, open(dosya, 'wb'))

    @staticmethod
    def trainTestPlot(acc_train, acc_test, model):
        acc_train, acc_test = float(acc_train * 100), float(acc_test * 100)
        x = ['Train Accuracy', 'Test Accuracy']
        hue = [acc_train, acc_test]
        plt.bar(x, hue, label='Percent', width=.4,
                color=['c', 'g'], edgecolor=['g', 'c'])
        plt.legend()
        plt.ylabel("Accuracy Rate")
        try:
            plt.title(
                f"Comparison of Train and Test Performance ----> {model.__name__}")
            plt.show()
        except:
            plt.title(
                f"Comparison of Train and Test Performance ----> {model.__class__.__name__}")
            plt.show()
