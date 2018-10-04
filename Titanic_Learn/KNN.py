from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def init():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    return X_train, X_test, y_train, y_test, iris


def KNN(X_train, X_test, y_train, y_test, iris):
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    print(knc.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=iris.target_names))


def main():
    X_train, X_test, y_train, y_test, iris = init()
    # KNN(X_train, X_test, y_train, y_test, iris)
    print(iris.DESCR)

if __name__ == "__main__":
    main()
