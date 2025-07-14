# active_learning.py
from modAL.models import ActiveLearner
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/cleaned.csv")
X = TfidfVectorizer().fit_transform(df["cleaned"])
y = df["class"]

initial_idx = [0, 1, 2, 3, 4]  # label a few samples manually
learner = ActiveLearner(estimator=LogisticRegression(), X_training=X[initial_idx], y_training=y[initial_idx])

for idx in range(5, len(y), 10):
    query_idx = learner.query(X)[0]
    learner.teach(X[query_idx], y[query_idx])  # simulate manual labeling
