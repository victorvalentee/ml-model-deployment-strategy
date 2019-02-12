"""
The simplest way to deploy a ML model and predict unseen samples.

In order to train a ML model you need two things: a dataset to train you model with and the model itself.
To predict unseen samples you need three things: the unseen samples, the trained model and the dataset schema used to train the model.

The training dataset schema is necessary for you to arrange unseen sample features in the same order as the training samples.
If you neglect this, it is very possible you will end up with incorrect predictions without even noticing it.

So let's get to work.
"""

# Training a model.
from sklearn.ensemble import GradientBoostingClassifier as gb
clf = gb()
clf.fit(X_train, y_train)

# The training dataset schema.
schema = X_train.columns

# By now you have all you need to deploy the training model in a persistent manner.
# Check out this link: https://scikit-learn.org/stable/modules/model_persistence.html
persistent_model = {
	'model':clf,
	'schema':schema
}

from joblib import dump, load
dump(clf, './persistent_model.joblib')

# Later you can load back the pickled model (possibly in another Python process).
persistent_model = load('./persistent_model.joblib')
model = persistent_model['model']
schema = persistent_model['schema']

# Reorder unseen samples using the training schema.
X_test = X_test[schema]

# Predict.
clf.predict(X_test)

# Ok, there you have it. Now let's DRY it.
from joblib import dump, load

def save_model(clf, schema, path):
	persistent_model = {'model':clf, 'schema':schema}
	return dump(clf, path)

def load_model(path):
	persistent_model = load(path)
	model = persistent_model['model']
	schema = persistent_model['schema']
	return model, schema
