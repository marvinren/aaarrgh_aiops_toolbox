from scipy import stats


class StatsDistributeDiff:

    def __init__(self, method="MW", **kwargs):
        funcs = {
            "MW": stats.mannwhitneyu,
            "KW": stats.kruskal,
            "KS": stats.ks_2samp,
        }
        self.method = funcs[method]
        self.ret = None

    def predict(self, x, y):
        self.ret = self.method(x, y)
        p = self.ret[1]
        if p < 0.05:
            return -1, p
        else:
            return 1, p


if __name__ == "__main__":
    group1 = [7, 14, 14, 13, 12, 9, 6, 14, 12, 8]
    group2 = [15, 17, 13, 15, 15, 13, 9, 12, 10, 8]
    group3 = [6, 8, 8, 9, 5, 14, 13, 8, 10, 9]

    ret = stats.kruskal(group1, group2, group3)
    print(ret)

    ret = stats.mannwhitneyu(group1, group2)
    print(ret, ret[0])

    model = StatsDistributeDiff(method="KS")
    print(model.predict(group1, group2))
