from sklearn.model_selection import train_test_split



def temporal_train_split(X, test_size,
                         return_cases=True,
                         case_id_col='case:concept:name',
                         timestamp_col='time:timestamp'):
    X = X[X.event_nr <= X.pos]
    X["case_start"] = X.groupby(case_id_col)[timestamp_col].transform("first")
    X["case_end"] = X.groupby(case_id_col)[timestamp_col].transform("last")
    case_ends = X.groupby(case_id_col)["case_end"].last().reset_index().sort_values(by=["case_end"])
    del X
    case_list = case_ends[case_id_col].values.tolist()
    end_list = case_ends["case_end"].values.tolist()
    train_idx, test_idx, end_train, end_test = train_test_split(case_list, end_list, test_size=test_size, shuffle=False)
    if return_cases:
        return train_idx, test_idx
    else:
        return[i for i in range(len(train_idx))], [j for j in range(len(train_idx), len(train_idx)+len(test_idx))]


