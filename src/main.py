#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
import argparse
from sklearn import linear_model, svm, neighbors, ensemble, metrics, decomposition, cluster, preprocessing, pipeline, random_projection

from preprocess import preprocess_dir
from split_data import split_data, get_splits


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
    #print(metrics.classification_report(predict, ys))
    print("\tPrecision: %f, Recall: %f, f1-score: %f" % metrics.precision_recall_fscore_support(predict, ys, average="weighted")[:3])
    #print(metrics.accuracy_score(predict, ys))


def test_model(name, model, splits):
    train_x, train_y, test_x, test_y = splits
    model.fit(train_x, train_y)
    prs = model.predict(test_x)
    print_results(name, prs, test_y)
    return prs


def get_counts(splits):
    train_x, train_y, test_x, test_y = splits

    n_samples, n_features = train_x.shape
    n_classes = len(set(train_y))

    return n_samples, n_features, n_classes


def get_cluster_distances(df, func):
    norm_df = lambda f: (f-f.min())/(f.max()-f.min()+1)

    ndf = df.apply(func, axis=1)
    ndf = ndf.apply(pd.Series)
    #ndf = norm_df(ndf)
    #ndf = ndf.apply(pd.Series)
    return ndf


def apply_kmeans(splits):
    print("Adding cluster class distances as features...")
    train_x, train_y, test_x, test_y = splits
    n_samples, n_features, n_classes = get_counts(splits)

    # only look at a subset for clustering
    s_train_x = train_x.sample(frac=0.1)
    s_test_x = test_x.sample(frac=0.1)

    kmeans = cluster.MiniBatchKMeans(n_clusters=n_classes)
    kmeans.fit(s_train_x)
    apply_kmeans = lambda d: tuple(kmeans.transform([d]).ravel())

    agglo = cluster.AgglomerativeClustering(n_clusters=n_classes)
    agglo.fit(s_train_x)
    apply_agglo = lambda d: tuple(kmeans.transform([d]).ravel())

    print("\tKMeans...")
    train_agg = get_cluster_distances(train_x, apply_agglo)
    test_agg = get_cluster_distances(test_x, apply_agglo)

    print("\tAgglomerative...")
    train_km = get_cluster_distances(train_x, apply_kmeans)
    test_km = get_cluster_distances(test_x, apply_kmeans)

    train_x = pd.concat([train_x, train_km, train_agg], axis=1)
    test_x = pd.concat([test_x, test_km, test_agg], axis=1)

    splits = (train_x, train_y, test_x, test_y)
    return splits


def apply_reduction(splits, eps):
    print("Applying dimensionality reduction...")
    train_x, train_y, test_x, test_y = splits
    n_samples, n_features, n_classes = get_counts(splits)
    print("Before:",n_samples, n_features, n_classes)

    min_comp = random_projection.johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
    min_comp = min(min_comp//2, n_features)
    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.QuantileTransformer()
    feat_agg = cluster.FeatureAgglomeration(n_clusters=min_comp)
    pca_pipe = pipeline.Pipeline([('scaler',scaler),('feat_agg',feat_agg)])
    train_x = pca_pipe.fit_transform(train_x)
    test_x = pca_pipe.transform(test_x)

    splits = (train_x, train_y, test_x, test_y)
    n_samples, n_features, n_classes = get_counts(splits)
    print("After:",n_samples, n_features, n_classes)
    return splits


def main():
    args = parse_args()

    print("Preprocessing inkml files...")
    data = get_data(args.gt, args.fdir)

    print("Splitting data into folds...")
    splits = get_splits(data, 0.7, train_p=0.5, total_p=0.1)
    train_x, train_y, test_x, test_y = splits

    n_samples, n_features, n_classes = get_counts(splits)
    print(n_samples, n_features, n_classes)

    print("Running classification tests...")
    warnings.filterwarnings('ignore')
    # test kd-tree model before PCA
    #ncc = neighbors.NearestCentroid()
    #test_model("Before PCA: Nearest Centroid", ncc, splits)
    #knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    #test_model("Before PCA: K=5 Neighbors", knn, splits)
    #svmm = svm.SVC(C=100.0, class_weight="balanced",  gamma='auto', tol=0.1)
    #test_model("Before PCA: SVM", svmm, splits)
    # test Random Forest model
    rfc = ensemble.RandomForestClassifier()
    test_model("Before PCA: Random Forest", rfc, splits)
    rfc = ensemble.RandomForestClassifier(criterion="entropy")
    test_model("Before PCA: Random Forest (entropy)", rfc, splits)

    '''
    # scale
    scaler = preprocessing.QuantileTransformer()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    splits = (train_x, train_y, test_x, test_y)
    '''

    # add k means data as features
    splits = apply_kmeans(splits)

    # apply reduction
    splits = apply_reduction(splits, eps=0.99)

    train_x, train_y, test_x, test_y = splits
    n_samples, n_features, n_classes = get_counts(splits)

    #ncc = neighbors.NearestCentroid()
    #test_model("Nearest Centroid", ncc, splits)

    # test Random Forest model
    rfc = ensemble.RandomForestClassifier()
    test_model("Random Forest", rfc, splits)
    rfc = ensemble.RandomForestClassifier(criterion="entropy")
    test_model("Random Forest (entropy)", rfc, splits)

    # test bag model
    bagc = ensemble.BaggingClassifier(rfc, max_samples=1.0, max_features=0.5, n_estimators=5, n_jobs=-1)
    test_model("Bagged RF", bagc, splits)

    # test Ada Boost model
    abc = ensemble.AdaBoostClassifier(rfc, n_estimators=5)
    test_model("Ada Boosted RF", abc, splits)

    # test bag Ada Boost model
    #bagc = ensemble.BaggingClassifier(abc, max_samples=1.0, max_features=0.5, n_estimators=5, n_jobs=-1)
    #test_model("Bagged Boosted RF", bagc, splits)

    # test Ada Boost bag model
    #abc = ensemble.AdaBoostClassifier(bagc, n_estimators=5)
    #test_model("Boosted Bagged RF", abc, splits)

    # test kd-tree model
    #knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
    #test_model("K=3 Neighbors", knn, splits)

    # test kd-tree model
    #knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    #test_model("K=5 Neighbors", knn, splits)

    # test kd-tree model
    #knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')
    #test_model("K=7 Neighbors", knn, splits)

    # test kd-tree model
    #knn = neighbors.KNeighborsClassifier(n_neighbors=13, weights='distance')
    #test_model("K=13 Neighbors", knn, splits)

    '''

    # test SVM model
    #svmm = svm.SVC(C=1, class_weight="balanced", gamma=0.5, probability=True)
    svmm = svm.SVC(C=5.0, class_weight="balanced",  gamma=0.5, tol=0.1, probability=False)
    test_model("SVM C=5.0, gamma=0.5", svmm, splits)

    svmm = svm.SVC(C=5.0, class_weight="balanced",  gamma='auto', tol=0.1, probability=False)
    test_model("SVM C=5.0, gamma=auto", svmm, splits)
    '''

    svmm = svm.SVC(C=100.0, class_weight="balanced",  gamma='auto', tol=0.1, probability=False)
    prs = test_model("SVM C=100.0, gamma=auto", svmm, splits)

    svmm = svm.SVC(C=100.0, gamma='auto', tol=0.1, probability=True)
    prs = test_model("SVM C=100.0, gamma=auto, uniform weights", svmm, splits)


    # test Ada Boost model
    #abc = ensemble.AdaBoostClassifier(svmm, learning_rate=1, n_estimators=3)
    #prs = test_model("Ada Boosted SVM", abc, splits)
    print(metrics.classification_report(prs, test_y))


if __name__=="__main__":
    main()

