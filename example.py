from sklearn.model_selection import train_test_split
from pyglmnet import GLM, GLMCV, datasets
import numpy as np # noqa
from sklearn.model_selection import GridSearchCV # noqa
from sklearn.model_selection import KFold # noqa

X, y = datasets.fetch_community_crime_data()
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

cv = KFold(3)

reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10,
                         base=np.exp(1))
param_grid = [{'reg_lambda': reg_lambda}]

glm = GLM(distr='binomial', alpha=0.05, score_metric='pseudo_R2',
          learning_rate=0.1, tol=1e-4, verbose=True)
glmcv = GridSearchCV(glm, param_grid, cv=cv)
glmcv.fit(X_train, y_train)

print("test set pseudo $R^2$ = %f" % glmcv.score(X_test, y_test))
