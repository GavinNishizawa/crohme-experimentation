import pandas as pd


def split_data(data, ratio):
    shuffled = data.sample(frac=1)
    s_ind = int(len(shuffled)*ratio)

    train = shuffled[:s_ind]
    test = shuffled[s_ind:]

    return train, test


def get_splits(data, total_p):
    train = data['train']
    test  = data['test']

    if total_p < 1:
        train, ignore = split_data(train, total_p)
        test, ignore = split_data(test, total_p)

    col_list = train.columns.tolist()
    to_rm = ['symbol','fn']
    for c in to_rm:
        col_list.remove(c)

    train_x = pd.DataFrame(train[col_list])
    train_y = train['symbol']

    test_x = pd.DataFrame(test[col_list])
    test_y = test['symbol']

    return {'train_x': train_x, 'train_y': train_y, \
            'test_x': test_x, 'test_y': test_y}


