#!/usr/bin/env python3
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import argparse
from sklearn import linear_model, svm, neighbors, ensemble, metrics, decomposition, cluster, preprocessing, pipeline, random_projection, feature_selection

from preprocess import preprocess_dir
from split_data import split_data, get_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Process a directory of inkml files using a ground truth reference file.")
    parser.add_argument("fdir", type=str, help="directory containing .inkml files")
    parser.add_argument("gt", type=str, help="ground truth file")
    parser.add_argument("ratio", type=float, help="ratio of train:test (0.0,1.0)")
    parser.add_argument("train_p", type=float, help="amount of train used to train (0.0,1.0]")
    parser.add_argument("total_p", type=float, help="amount of data used (0.0,1.0]")

    return parser.parse_args()


def get_data(gt_fn, dir_fn):
    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    data = preprocess_dir(dir_fn, df)
    #print(data.head(1))
    return data


def print_results(name, predict, ys):
    #print(name)
    warnings.filterwarnings('ignore')
    print("\tPrecision: %f, Recall: %f, f1-score: %f" % \
            metrics.precision_recall_fscore_support( \
            predict, ys, average="weighted")[:3])
    warnings.resetwarnings()
    #print(metrics.accuracy_score(predict, ys))
    #print(metrics.classification_report(predict, ys))


def get_cluster_distances(df, func):
    norm_df = lambda f: (f-f.min())/(f.max()-f.min()+1)

    ndf = df.apply(func, axis=1)
    ndf = ndf.apply(pd.Series)
    #ndf = norm_df(ndf)
    #ndf = ndf.apply(pd.Series)
    return ndf


def apply_kmeans(splits):
    print("Adding cluster distances as features...")

    # cluster splits
    c_splits = dict(splits)
    # feature selection for clusters
    c_splits = apply_feature_select(c_splits, "mean")

    n_samples, n_features, n_classes = get_counts(c_splits)

    kmeans = cluster.MiniBatchKMeans(n_clusters=n_classes)
    kmeans.fit(c_splits['train_x'])
    apply_kmeans = lambda d: tuple(kmeans.transform([d]).ravel())

    print("\tKMeans...")
    train_km = get_cluster_distances(c_splits['train_x'], apply_kmeans)
    test_km = get_cluster_distances(c_splits['test_x'], apply_kmeans)

    train_km.index = splits['train_x'].index
    splits['train_x'] = pd.concat([splits['train_x'], train_km], axis=1)

    test_km.index = splits['test_x'].index
    splits['test_x'] = pd.concat([c_splits['test_x'], test_km], axis=1)

    n_samples, n_features, n_classes = get_counts(splits)
    print("Added cluster distances:",n_samples, n_features, n_classes)
    return splits


def apply_reduction(splits, eps):
    print("Applying dimensionality reduction...")
    n_samples, n_features, n_classes = get_counts(splits)
    print("Before:",n_samples, n_features, n_classes)

    min_comp = random_projection.johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
    min_comp = min(min_comp, n_features)
    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.QuantileTransformer()
    feat_agg = cluster.FeatureAgglomeration(n_clusters=min_comp)
    pca_pipe = pipeline.Pipeline([('scaler',scaler),('feat_agg',feat_agg)])
    splits['train_x'] = pd.DataFrame(pca_pipe.fit_transform( \
            splits['train_x']))
    splits['test_x'] = pd.DataFrame(pca_pipe.transform( \
            splits['test_x']))

    n_samples, n_features, n_classes = get_counts(splits)
    print("After:",n_samples, n_features, n_classes)
    return splits


def get_save_fn(fn):
    return os.path.join("pickled",fn+".pkl")


def save_obj(fn, obj):
    sfn = get_save_fn(fn)
    pickle.dump(obj, open(sfn, 'wb'), \
            protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(fn):
    sfn = get_save_fn(fn)
    if os.path.isfile(sfn):
        return pickle.load(open(sfn, 'rb'))


def apply_feature_select(splits, threshold="median"):
    xtc = ensemble.ExtraTreesClassifier(n_estimators=50, \
            n_jobs=-1)
    my_train_model(xtc, splits['train_x'], splits['train_y'])
    print("Max Feature importance:",max(xtc.feature_importances_))
    print("Min Feature importance:",min(xtc.feature_importances_))

    feat_sel = feature_selection.SelectFromModel( \
            xtc, prefit=True, threshold=threshold)
    print("Before feature selection:",splits['train_x'].shape)

    splits['train_x'] = pd.DataFrame( \
            feat_sel.transform(splits['train_x']))
    splits['test_x'] = pd.DataFrame( \
            feat_sel.transform(splits['test_x']))
    print("After feature selection:",splits['train_x'].shape)
    return splits


def my_train_model(model, train_x, train_y):
    model.fit(train_x, train_y)


def my_test_model(name, model, test_x, test_y):
    prs = model.predict(test_x)
    print_results(name, prs, test_y)
    return prs


def train_test(name, model, splits):
    print("\nTraining:",name,"..")
    start_train = time.time()
    my_train_model(model, splits['train_x'], splits['train_y'])
    print_time(start_train, time.time())

    print("Testing:",name,"..")
    start_test = time.time()
    results = my_test_model(name, model, splits['test_x'], splits['test_y'])
    print_time(start_test, time.time())

    return results


def test_on_train(name, model, splits):
    print("Testing on training data:",name,"..")
    start_test = time.time()
    results = my_test_model(name, model, splits['train_x'], splits['train_y'])
    print_time(start_test, time.time())

    return results


def get_counts(splits):
    return get_counts_tt(splits['train_x'], splits['train_y'])


def get_counts_tt(train_x, train_y):
    n_samples, n_features = train_x.shape
    n_classes = len(set(train_y))

    return n_samples, n_features, n_classes


# train the dim reduction and feature selection pipeline
def train_drfs(train_x, train_y, eps=0.5, threshold="median"):
    n_samples, n_features, n_classes = \
            get_counts_tt(train_x, train_y)

    # pick number of components
    min_comp = random_projection.johnson_lindenstrauss_min_dim( \
            n_samples=n_samples, eps=eps)
    min_comp = min(min_comp, n_features)

    # scale and agglomerate to min_comp
    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.QuantileTransformer()
    feat_agg = cluster.FeatureAgglomeration( \
            n_clusters=min_comp)
    xtc = ensemble.ExtraTreesClassifier(n_estimators=50, n_jobs=-1)

    # train the model pipeline
    dr_pipe = pipeline.Pipeline(\
            [('scaler', scaler), ('feat_agg', feat_agg)])
    dr_pipe.fit(train_x)

    # transform train_x to train xtc
    train_x = dr_pipe.transform(train_x)
    # train the xtc
    xtc.fit(train_x, train_y)

    print("Feature importances:")
    print("\tMax:",max(xtc.feature_importances_))
    print("\tMin:",min(xtc.feature_importances_))

    # create the feature selection model from the xtc
    feat_sel = feature_selection.SelectFromModel( \
            xtc, prefit=True, threshold=threshold)

    # create the pipeline to reduce dim then feature select
    drfs_pipe = pipeline.Pipeline(\
            [('dr_pipe', dr_pipe), ('feat_sel', feat_sel)])

    return drfs_pipe


def apply_drfs_sample(test_data, drfs_model):
    '''Reduce components on one or more test samples.
    '''
    return pd.DataFrame(drfs_model.transform(test_data))


def apply_drfs(splits, typ='drfs_model_train', load=True, eps=0.5, threshold="median"):
    drfs_pipe = None
    if load:
        drfs_pipe = load_obj(typ)
    if drfs_pipe is None:
        drfs_pipe = train_drfs(splits['train_x'],  \
                splits['train_y'], eps, threshold)
        if load:
            # pickle model for the future
            save_obj(typ, drfs_pipe)

    n_samples, n_features, n_classes = get_counts(splits)
    print("Before:",n_samples, n_features, n_classes)

    splits['train_x'] = apply_drfs_sample( \
            splits['train_x'], drfs_pipe)

    n_samples, n_features, n_classes = get_counts(splits)
    print("After:",n_samples, n_features, n_classes)

    if splits['test_y'] is not None and len(splits['test_x']) > 0:
        splits['test_x'] = apply_drfs_sample( \
                splits['test_x'], drfs_pipe)
    return splits


def print_time(start,end):
    elapsed = end-start
    print('\tTime: %d m %2d s' % (elapsed/60, elapsed%60))


def main():
    args = parse_args()
    ratio = args.ratio
    train_p = args.train_p
    total_p = args.total_p

    start_time = time.time()

    print("Preprocessing inkml files...")
    data = get_data(args.gt, args.fdir)

    print("Splitting data into folds...")

    # save/load data split
    split_fn = "split"+str(ratio)+str(train_p)+str(total_p) + os.path.basename(args.fdir)

    load = False
    splits = None
    if load:
        splits = load_obj(split_fn)
    if splits == None:
        splits = get_splits(data, ratio, train_p, total_p)
        if load:
            save_obj(split_fn, splits)

    n_samples, n_features, n_classes = get_counts(splits)
    print(n_samples, n_features, n_classes)

    print("Running classification tests...")
    '''
    name = "Nearest Centroid"
    ncc = neighbors.NearestCentroid()
    train_test(name, ncc, splits)
    test_on_train(name, ncc, splits)
    # test Random Forest model
    rfc = ensemble.RandomForestClassifier()
    train_test("Initial: Random Forest", rfc, splits)
    rfc = ensemble.RandomForestClassifier(criterion="entropy")
    train_test("Initial: Random Forest (entropy)", rfc, splits)
    # scale
    scaler = preprocessing.QuantileTransformer()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    splits = (train_x, train_y, test_x, test_y)

    # add k means data as features (seems to reduce accuracy)
    #splits = apply_kmeans(splits)
    '''

    name = "Extra Trees (50)"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=50, n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)

    '''
    name = "Extra Trees (50) max depth 30"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=50, max_depth=30, n_jobs=-1)
    train_test(name, xtc, splits)
    test_on_train(name, xtc, splits)
    '''

    # apply dim reduction and feature selection
    splits = apply_drfs(splits, load=False, eps=0.999, threshold="0.9*median")#0.004)

    n_samples, n_features, n_classes = get_counts(splits)

    '''
    name = "Nearest Centroid"
    ncc = neighbors.NearestCentroid()
    train_test(name, ncc, splits)
    test_on_train(name, ncc, splits)

    # test Random Forest model
    rfc = ensemble.RandomForestClassifier()
    train_test("Random Forest", rfc, splits)
    rfc = ensemble.RandomForestClassifier(criterion="entropy")
    train_test("Random Forest (entropy)", rfc, splits)

    # test bag model
    bagc = ensemble.BaggingClassifier(rfc, max_samples=1.0, max_features=0.5, n_estimators=5, n_jobs=-1)
    train_test("Bagged RF", bagc, splits)

    # test Ada Boost model
    abc = ensemble.AdaBoostClassifier(rfc, n_estimators=5)
    train_test("Ada Boosted RF", abc, splits)
    '''

    name = "Extra Trees"
    xtc = ensemble.ExtraTreesClassifier(n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)

    name = "Extra Trees (50)"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=50, n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)

    '''
    name = "Extra Trees (50) max depth 30"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=50, max_depth=30, n_jobs=-1)
    train_test(name, xtc, splits)
    test_on_train(name, xtc, splits)

    # test Ada Boost model
    name = "Ada Boosted Extra Trees (50) max depth 21"
    abc = ensemble.AdaBoostClassifier(xtc, n_estimators=3)
    train_test(name, abc, splits)
    #test_on_train(name, abc, splits)
    '''

    name = "Extra Trees (100)"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=100, n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)

    name = "SVM C=50.0, gamma=auto, uniform weights"
    svmm = svm.SVC(C=50.0, gamma='auto', tol=0.00001, probability=True)
    prs = train_test(name, svmm, splits)
    #test_on_train(name, svmm, splits)
    print(metrics.classification_report(prs, splits['test_y']))

    '''
    name = "SVM C=100.0, gamma=auto, uniform weights"
    svmm = svm.SVC(C=100.0, gamma='auto', tol=0.000001, probability=True)
    train_test(name, svmm, splits)
    test_on_train(name, svmm, splits)

    # test SVM models
    m_name = "SVM C=100.0, gamma=auto"
    svmm = load_obj(m_name)
    if True or svmm == None:
        svmm = svm.SVC(C=100.0, class_weight="balanced",  gamma='auto', tol=0.1, probability=False)
        train_model(svmm, train_x, train_y)
        save_obj("model-"+m_name, svmm)
    prs = test_model(m_name, svmm, test_x, test_y)

    m_name = "SVM C=100.0, gamma=auto, uniform weights"
    svmm = load_obj(m_name)
    if True or svmm == None:
        svmm = svm.SVC(C=100.0, gamma='auto', tol=0.1, probability=False)
        train_model(svmm, train_x, train_y)
        save_obj("model-"+m_name, svmm)
    prs = test_model(m_name, svmm, test_x, test_y)

    svmm = svm.LinearSVC(dual=False)
    train_test("LinearSVC", svmm, splits)
    '''

    print("\nTests complete!")
    print_time(start_time, time.time())


if __name__=="__main__":
    main()

