from __future__ import print_function

import pandas as pd
import matplotlib.pylab as plt


def compute_uplift_core(df, col_name_of_interest, random=False):
    if not random:
        df = df.sort_values(by=col_name_of_interest[0], ascending=False).reset_index()
    else:
        df = df.sample(frac=1).reset_index()

    treat_df = df.copy()
    treat_df["modified_response"] = treat_df[col_name_of_interest[1]] * treat_df[col_name_of_interest[2]]

    treat_cumsum = treat_df.modified_response.cumsum() / (treat_df[col_name_of_interest[2]].cumsum() + 1e-6)
    ctrl_df = df.copy()
    ctrl_df["modified_response"] = ctrl_df[col_name_of_interest[1]] * (1 - ctrl_df[col_name_of_interest[2]])
    ctrl_cumsum = ctrl_df.modified_response.cumsum() / ((1 - ctrl_df[col_name_of_interest[2]]).cumsum() + 1e-6)
    te = treat_cumsum - ctrl_cumsum
    te = te.cumsum()
    te = te / te.iloc[-1]
    return te


def compute_area(values):
    assert isinstance(values, pd.Series)
    shifted_cumsum = pd.Series([0., 0.], index=[0, 1])
    values.reset_index()
    shifted_cumsum = shifted_cumsum.append(
        values.drop([len(values)-2, len(values)-1])).cumsum().reset_index(drop=True)
    values_cumsum = values.cumsum()

    return ((values_cumsum - shifted_cumsum).drop([0]) * 0.5).mean()


def compute_uplift(df, col_name_of_interest):
    """

    Args:
        df:
        col_name_of_interest: ['score_name', real_response_name', 'treatment_tag_name']

    Returns:

    """
    assert len(col_name_of_interest) == 3
    assert isinstance(df, pd.DataFrame)
    te = compute_uplift_core(df, col_name_of_interest, random=False)
    te_random = compute_uplift_core(df, col_name_of_interest, random=True)
    area = compute_area(te)
    random_area = compute_area(te_random)
    print("area:{:.5f}, random_area:{:.5f}, auuc:{:.5f}".format(area, random_area, area - random_area))

    te = pd.concat([te, te_random], axis=1)
    te.columns = ['auuc', 'random_auuc']
    te.plot()
    plt.show()

    # print(treat_df)
    # print(ctrl_df)


if __name__ == '__main__':
    score_list = [0.5, 0.2, 0.6]
    y_list = [1., 2., 3.]
    t_list = [1, 0, 1]
    metric_df = pd.DataFrame([score_list, y_list, t_list]).T
    metric_df.columns = ['model', 'y', 'w']
    print(metric_df)

    compute_uplift(df=metric_df, col_name_of_interest=["model", "y", "w"])
