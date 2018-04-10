import numpy as np
import random as r


def split_data(data, ratio):
    shuffled = data.sample(frac=1)
    s_ind = int(len(shuffled)*ratio)

    train = shuffled[:s_ind]
    test = shuffled[s_ind:]

    return train, test


def get_splits(data, ratio, train_p, total_p):
    data, ignore = split_data(data, total_p)
    train, test = split_data(data, ratio)
    train, ignore = split_data(train, train_p)

    col_list = train.columns.tolist()
    to_rm = ['symbol','image','fn','symbol_fn']
    for c in to_rm:
        col_list.remove(c)

    train_x = train[col_list]
    train_y = np.squeeze(train['symbol'].values.tolist())

    test_x = test[col_list]
    test_y = np.squeeze(test['symbol'].values.tolist())

    return train_x, train_y, test_x, test_y


def main():
    # Example usage
    td = pd.DataFrame()

    train, test = split_data(td, 0.7)
    print(train.shape, test.shape)


if __name__=='__main__':
    main()

