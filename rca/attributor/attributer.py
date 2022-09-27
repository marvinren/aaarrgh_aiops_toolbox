import numpy as np
import pandas as pd


class AtributorMD:
    """
    Read:
    https://github.com/shaido987/multi-dim-baselines/blob/main/adtributor.py
    https://github.com/NetManAIOps/PSqueeze
    """
    def __init__(self, p_col_name="pv_pred", a_col_name="pv_actual", timestamp_col_name="timestamp", t_ep=0.67,
                 t_eep=0.1):
        self.p_col_name = p_col_name
        self.a_col_name = a_col_name
        self.timestamp_col_name = timestamp_col_name

        self.t_eep = t_eep
        self.t_eq = t_ep

    def search_root_cause(self, data_cube: pd.DataFrame):
        timestamps = data_cube[self.timestamp_col_name].unique()
        attribute_names = list(
            sorted(set(data_cube.columns) - {self.p_col_name, self.a_col_name, self.timestamp_col_name}))
        root_case = {}
        for t in timestamps:
            df_cube = data_cube[data_cube[self.timestamp_col_name] == t]
            ac = self.search_cols_by_atributor(attribute_names, df_cube)
            root_case[t] = ";".join(list(map(lambda x: x[0] + "=" + "&".join(x[1]), ac)))
        return root_case

    def search_cols_by_atributor(self, attribute_names, df_cube):
        candidacies = []
        surprises = []
        for attr in attribute_names:
            candi_attr_val, surprise = self.search_root_cause_in_the_attribute(df_cube, attr)
            candidacies.append((attr, candi_attr_val))
            surprises.append(surprise)
        df_ret = pd.DataFrame({"ac": candidacies, "surprise": surprises})
        ac_df = df_ret.sort_values(by="surprise", ascending=False)
        ss = ac_df["surprise"].sum()
        s = 0
        ret = []
        for idx, a in ac_df.iterrows():
            s += a["surprise"]
            ret.append(a['ac'])
            if s / ss >= 0.8 or idx > 3:
                break
        return ret

    def search_root_cause_in_the_attribute(self, data_cube: pd.DataFrame, dim_4_check):
        p_col_name = self.p_col_name
        a_col_name = self.a_col_name
        p_col_name_all_sum = self.p_col_name + "_all_sum"
        a_col_name_all_sum = self.a_col_name + "_all_sum"

        # copy the all data
        df = data_cube[[dim_4_check, p_col_name, a_col_name]].copy()
        # calculate sum_f and sum_v
        # TODO: update for mutli-metrics
        pv_sum = df.sum(numeric_only=True)

        # start to calculate [dim_4_check] dimension
        group_chk_dim = df.groupby(dim_4_check).sum()
        group_chk_dim[p_col_name_all_sum] = pv_sum[p_col_name]
        group_chk_dim[a_col_name_all_sum] = pv_sum[a_col_name]

        # calculate the Surprise score
        group_chk_dim['p'] = group_chk_dim[p_col_name] / group_chk_dim[p_col_name_all_sum]
        group_chk_dim['q'] = group_chk_dim[a_col_name] / group_chk_dim[a_col_name_all_sum]
        group_chk_dim['surprise'] = group_chk_dim[['p', 'q']].apply(lambda x: _calc_surprise_score(x['p'], x['q']),
                                                                    axis=1)

        # calculate the EP score
        group_chk_dim['EP'] = group_chk_dim[[p_col_name, a_col_name, p_col_name_all_sum, a_col_name_all_sum]].apply(
            lambda x: _calc_ep_score(x[p_col_name], x[a_col_name], x[p_col_name_all_sum], x[a_col_name_all_sum]), axis=1
        )
        # get this dimension's surprise by summing the all surprise score
        # group_surprise_sum = group_chk_dim['surprise'].sum()
        # filter by ep > t_eep and sort by surprise
        group_root_cause = group_chk_dim[group_chk_dim['EP'] > self.t_eep]
        group_root_cause = group_root_cause.sort_values(by=['surprise'], axis=0, ascending=False)
        # cusum ep < t_ep
        ep = 0
        candi_idx = 0
        for eq_s in group_root_cause["EP"]:
            ep += eq_s
            candi_idx += 1
            if ep > self.t_eq:
                break

        if 0 < candi_idx < group_root_cause.size:
            df = group_root_cause[:candi_idx]
            return df.index.tolist(), df['surprise'].sum()
        else:
            return [], 0


def _calc_ep_score(pv_pred, pv_actual, pred_sum, actual_sum):
    if actual_sum - pred_sum == 0:
        return pv_actual - pv_pred
    else:
        return (pv_actual - pv_pred) / (actual_sum - pred_sum)


def _calc_surprise_score(p, q):
    """calculate Surprise by using js"""
    p = np.array(p)
    q = np.array(q)
    m = (p + q) / 2
    # if p, q, m is zero, set to 1 for js calculation
    p = np.where(p == 0., 1, p)
    q = np.where(q == 0., 1, q)
    m = np.where(m == 0., 1, m)
    # calculate js distance
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return round(float(js), 6)
