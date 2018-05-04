#!/usr/bin/env python3
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import argparse
from sklearn import linear_model, svm, neighbors, ensemble, metrics, decomposition, cluster, preprocessing, pipeline, random_projection, feature_selection, neural_network

from preprocess import preprocess_dir, get_process_iso, \
        get_get_symbols_iso, get_symbols_inkml, process_inkml
from split_data import get_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Process a directory of inkml files.")
    parser.add_argument("fdir", type=str, help="directory containing .inkml files")
    parser.add_argument("gt", nargs="?", type=str, help="ground truth file")
    parser.add_argument("-r", nargs="?", type=float, default=0.7, help="ratio of train:test (0.0,1.0)")
    #parser.add_argument("train_p", nargs="?", type=float, default=1, help="amount of train used to train (0.0,1.0]")
    parser.add_argument("-p", nargs="?", type=float, default=1, help="amount of data used (0.0,1.0]")

    return parser.parse_args()


def get_iso_data(gt_fn, dir_fn, ratio):
    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    process_iso = get_process_iso(df)
    get_symbols = get_get_symbols_iso(df)
    data = preprocess_dir(dir_fn, process_iso, get_symbols, ratio)
    #print(data.head(1))
    return data


def my_get_data(dir_fn, gt_fn, ratio):
    if gt_fn != None:
        print("Getting iso data...")
        data = get_iso_data(gt_fn, dir_fn, ratio)
    else:
        print("Getting data...")
        data = preprocess_dir(dir_fn, process_inkml, get_symbols_inkml, ratio)
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
    xtc = ensemble.ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    scaler2 = preprocessing.RobustScaler()
    #poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)

    # train the model pipeline
    dr_pipe = pipeline.Pipeline([('scaler', scaler), \
            ('feat_agg', feat_agg),('scaler2', scaler2)])

    dr_pipe.fit(train_x)

    # transform train_x to train xtc
    train_x = dr_pipe.transform(train_x)
    # train the xtc
    xtc.fit(train_x, train_y)

    print("Feature importances:")
    print("\tMax:",max(xtc.feature_importances_))
    print("\tMin:",min(xtc.feature_importances_))
    #print(xtc.feature_importances_)

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
    start = time.time()
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
    print_time(start, time.time())

    if splits['test_y'] is not None and len(splits['test_x']) > 0:
        splits['test_x'] = apply_drfs_sample( \
                splits['test_x'], drfs_pipe)
    return splits


def print_time(start,end):
    elapsed = end-start
    print('\tTime: %d m %2d s' % (elapsed/60, elapsed%60))


def main():
    args = parse_args()
    ratio = args.r
    total_p = args.p

    start_time = time.time()

    print("Preprocessing inkml files...")
    data = my_get_data(args.fdir, args.gt, ratio)

    # save/load data split
    split_fn = "split"+str(ratio)+str(total_p) + os.path.basename(args.fdir)

    load = False
    splits = None
    if load:
        splits = load_obj(split_fn)
    if splits == None:
        splits = get_splits(data, total_p)
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
    '''

    name = "Extra Trees (50)"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=50, n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)


    # apply dim reduction and feature selection
    splits = apply_drfs(splits, load=False, eps=0.001, threshold="0.9*mean")#0.004)

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

    name = "KNN (5)"
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    train_test(name, knn, splits)

    name = "Extra Trees"
    xtc = ensemble.ExtraTreesClassifier(n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)
    '''

    name = "Extra Trees (50)"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=50, n_jobs=-1)
    train_test(name, xtc, splits)
    #test_on_train(name, xtc, splits)

    '''
    name = "Extra Trees (100) max depth 20"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=100, max_depth=20, n_jobs=-1)
    train_test(name, xtc, splits)

    name = "Extra Trees (100) max depth 30"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=100, max_depth=30, n_jobs=-1)
    train_test(name, xtc, splits)
    '''

    name = "Extra Trees (100)"
    xtc = ensemble.ExtraTreesClassifier( \
            n_estimators=100, n_jobs=-1)
    train_test(name, xtc, splits)
    test_on_train(name, xtc, splits)


    name = "MLP, (2*n) sgd adaptive"
    nns = int(2*n_classes)
    mlp = neural_network.MLPClassifier( \
            solver='sgd', \
            learning_rate='adaptive', \
            learning_rate_init=0.7, \
            hidden_layer_sizes=(nns), max_iter=1000)
    train_test(name, mlp, splits)
    test_on_train(name, mlp, splits)

    name = "MLP, (3*n) sgd adaptive"
    nns = int(3*n_classes)
    mlp = neural_network.MLPClassifier( \
            solver='sgd', \
            learning_rate='adaptive', \
            learning_rate_init=0.7, \
            hidden_layer_sizes=(nns), max_iter=1000)
    train_test(name, mlp, splits)
    test_on_train(name, mlp, splits)


    name = "SVM C=25.0, gamma=auto, uniform weights"
    svmm = svm.SVC(C=25.0, gamma='auto', tol=0.00001, probability=True)
    train_test(name, svmm, splits)

    votc = ensemble.VotingClassifier(estimators=[ \
            ('xt100',xtc),('svm',svmm),('mlp',mlp)], \
            voting='hard')
    prs = train_test("Voting [ExtraTrees(100), SVM(C=25), MLP] (hard)", votc, splits)
    test_on_train(name, svmm, splits)

    print(metrics.classification_report(prs, splits['test_y']))

    '''
    votc = ensemble.VotingClassifier(estimators=[ \
            ('xt100',xtc),('svm',svmm),('mlp',mlp)], \
            voting='soft', weights=[2,7,3])
    prs = train_test("Voting [ExtraTrees(100), SVM(C=25), MLP] (soft)", votc, splits)

    bagc = ensemble.BaggingClassifier(svmm, max_samples=1.0, bootstrap=False, max_features=0.7, n_estimators=3, n_jobs=-1)
    train_test("Bagged (3) 0.7 SVM", bagc, splits)

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

