import pandas as pd


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
    to_rm = ['symbol','fn']
    for c in to_rm:
        col_list.remove(c)

    train_x = pd.DataFrame(train[col_list])
    train_y = train['symbol']

    test_x = pd.DataFrame(test[col_list])
    test_y = test['symbol']

    return {'train_x': train_x, 'train_y': train_y, \
            'test_x': test_x, 'test_y': test_y}


def main():
    # Example usage
    td = pd.DataFrame()

    train, test = split_data(td, 0.7)
    print(train.shape, test.shape)


if __name__=='__main__':
    main()

