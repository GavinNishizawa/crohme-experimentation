#!/usr/bin/env python3
import warnings
import pandas as pd
import argparse
from sklearn import linear_model, svm, neighbors, ensemble, metrics, decomposition

from preprocess import preprocess_dir
from split_data import get_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Process a directory of inkml files using a ground truth reference file.")
    parser.add_argument("fdir", type=str, help="directory containing .inkml files")
    parser.add_argument("gt", type=str, help="ground truth file")

    return parser.parse_args()


def get_data(gt_fn, dir_fn):
    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    data = preprocess_dir(dir_fn, df)
    #print(data.head(1))
    return data


def print_results(name, predict, ys):
    print(name)
    #print(metrics.classification_report(prs, test_y))
    print("\tPrecision: %f, Recall: %f, f1-score: %f" % metrics.precision_recall_fscore_support(predict, ys, average="weighted")[:3])
    #print(metrics.accuracy_score(prs, test_y))


def test_model(name, model, splits):
    train_x, train_y, test_x, test_y = splits
    model.fit(train_x, train_y)
    prs = model.predict(test_x)
    print_results(name, prs, test_y)
    return prs


def main():
    print("Running classification tests...")
    args = parse_args()

    data = get_data(args.gt, args.fdir)

    splits = get_splits(data, 0.7)
    train_x, train_y, test_x, test_y = splits

    print(train_x.shape)
    warnings.filterwarnings('ignore')
    # test kd-tree model before PCA
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    test_model("Before PCA: K=5 Neighbors", knn, splits)
    svmm = svm.SVC(C=50.0, class_weight="balanced",  gamma='auto', tol=0.1)
    test_model("Before PCA: SVM", svmm, splits)
    # test Random Forest model
    rfc = ensemble.RandomForestClassifier()
    test_model("Before PCA: Random Forest", rfc, splits)

    # apply PCA
    n_samples, n_features = train_x.shape
    #pca = decomposition.PCA(n_components=int(n_features*0.5), svd_solver='arpack', whiten=False)
    pca = decomposition.FactorAnalysis(n_components=int(n_features*0.5))

    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print(train_x.shape)
    splits = (train_x, train_y, test_x, test_y)

    # test kd-tree model
    knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
    test_model("K=3 Neighbors", knn, splits)

    # test kd-tree model
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    test_model("K=5 Neighbors", knn, splits)

    # test kd-tree model
    knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')
    test_model("K=7 Neighbors", knn, splits)

    # test kd-tree model
    #knn = neighbors.KNeighborsClassifier(n_neighbors=13, weights='distance')
    #test_model("K=13 Neighbors", knn, splits)

    # test Random Forest model
    rfc = ensemble.RandomForestClassifier()
    test_model("Random Forest", rfc, splits)

    '''
    # test bag model
    bagc = ensemble.BaggingClassifier(rfc, max_samples=1.0, max_features=0.5, n_estimators=5, n_jobs=-1)
    test_model("Bagged RF", bagc, splits)

    # test Ada Boost model
    abc = ensemble.AdaBoostClassifier(rfc, n_estimators=5)
    test_model("Ada Boosted RF", abc, splits)

    # test bag Ada Boost model
    bagc = ensemble.BaggingClassifier(abc, max_samples=1.0, max_features=0.5, n_estimators=5, n_jobs=-1)
    test_model("Bagged Boosted RF", bagc, splits)

    # test Ada Boost bag model
    abc = ensemble.AdaBoostClassifier(bagc, n_estimators=5)
    test_model("Boosted Bagged RF", abc, splits)

    # test SVM model
    #svmm = svm.SVC(C=1, class_weight="balanced", gamma=0.5, probability=True)
    svmm = svm.SVC(C=5.0, class_weight="balanced",  gamma=0.5, tol=0.1, probability=False)
    test_model("SVM C=5.0, gamma=0.5", svmm, splits)

    svmm = svm.SVC(C=5.0, class_weight="balanced",  gamma='auto', tol=0.1, probability=False)
    test_model("SVM C=5.0, gamma=auto", svmm, splits)
    '''

    svmm = svm.SVC(C=50.0, class_weight="balanced",  gamma='auto', tol=0.1, probability=True)
    test_model("SVM C=50.0, gamma=auto", svmm, splits)

    # test Ada Boost model
    abc = ensemble.AdaBoostClassifier(svmm, learning_rate=1, n_estimators=3)
    prs = test_model("Ada Boosted SVM", abc, splits)
    print(metrics.classification_report(prs, test_y))


if __name__=="__main__":
    main()

