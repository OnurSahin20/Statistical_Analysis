import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


class DataProcessing:
    def __init__(self, file_name):
        self.file_name = file_name
        self.month = 12
        self.days = 31
        self.all_outputs = self.get_datas()
        self.stations = list(self.all_outputs[0].keys())
        self.datas = self.all_outputs[0]
        self.years = self.all_outputs[1]

    @staticmethod
    def str_to_float(l1):
        new_l1 = []
        for s in l1:
            if float(s) == -5.0:
                new_l1.append(np.nan)
            else:
                new_l1.append(float(s))
        return new_l1

    def get_datas(self):
        
        with open("prec_data.csv") as file:
            datas = {}
            years = {}
            d = np.array([])
            y = np.array([])
            key = int(next(file).strip("\n").split(",")[0][3:])
            for f in file:
                if np.shape(d)[0] == 0:
                    d = np.array([self.str_to_float(f.strip("\n").split(","))])[:, 1:]
                    y = np.array([self.str_to_float(f.strip("\n").split(","))])[:, 0]
                elif f.strip("\n").split(",")[0] == "":
                    datas[key] = d
                    years[key] = y
                    key = int(next(file).strip("\n").split(",")[0])
                    d = np.array([])
                    y = np.array([])
                else:
                    d = np.vstack((d, np.array([self.str_to_float(f.strip("\n").split(","))])[:, 1:]))
                    y = np.vstack((y, np.array([self.str_to_float(f.strip("\n").split(","))])[:, 0]))

        return datas, years

    def annual_summation(self, station):
        if np.sum(np.isnan(self.datas[station])) != 0:
            return np.sum(self.replace_missing_median(station), axis=1)
        else:
            return np.sum(self.datas[station], axis=1)

    def distribution_test(self, station):
        list_of_dists = ["norm", "gamma", "lognorm", "loggamma"]
        result = []
        shapes = np.shape(self.datas[station])
        data = self.replace_missing_median(station).reshape((shapes[0] * shapes[1]))
        for i in list_of_dists:
            dist = getattr(stats, i)
            param = dist.fit(data)
            a = stats.kstest(data, i, args=param)
            result.append((i, a[0], a[1]))
            result.sort(key=lambda x: float(x[2]), reverse=True)
        return "The most fitting distrubiton is {}".format(result[0][0]), result

    def general_info(self):
        print('General information about the given station !')
        print("Monthly flow or precipitation of given data set ")
        print("--------------------")
        if len(self.stations) > 10:
            print("Data set have {0} stations and 10 of them are {1}".format(len(self.stations), self.stations[0:11]))
        else:
            print("Data set have {0} stations and them are {1}".format(len(self.stations), self.stations))
        max_interspace = max([np.shape(self.years[key])[0] - 1 for key in self.datas])
        min_interspace = min([np.shape(self.years[key])[0] - 1 for key in self.datas])
        print("Maximum time  is {} year and minimum is {}".format(max_interspace, min_interspace))
        return "--------------------"

    def show_statistics(self, station):
        labels = []
        d = self.datas[station]
        index = ['Max', "Min", "average", "std_deviation"]
        c = 0
        df = pd.DataFrame(index=index)
        for _ in range(np.shape(d)[0]):
            labels.append(self.years[station][_, 0])
            df[c] = np.array([np.nanmax(d[_]), np.nanmin(d[_]), np.nanmean(d[_]), np.nanstd(d[_])])
            c += 1
        df.columns = labels
        return df

    def missing_values_percentage(self, station):
        a = np.sum(np.isnan(self.datas[station]))
        ll = np.shape(self.datas[station])
        return a / (ll[0] * ll[1]) * 100, "Station = {0}, {1} data is missing, % {2} all of datas!". \
            format(station, a, a / (ll[0] * ll[1]) * 100)

    def replace_missing_median(self, station):
        """Changes nan variable with median of variables for column which variable belongs to.
        Function changes global variable self.datas for given station!!"""
        shape = np.shape(self.datas[station])
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.isnan(self.datas[station][i, j]):
                    self.datas[station][i, j] = np.nanmedian(self.datas[station][:, j])
        return self.datas[station]

    def choose_stations(self, min_year, min_mis_val_per=50):
        """This function  choose stations criterion of minimum_year : min_year and
        min_mis_val_per : percentage of missing value """
        stations = []
        for station in [key for key in self.years if np.shape(self.datas[key])[0] >= min_year + 1]:
            missing_percentage = self.missing_values_percentage(station)[0]
            if missing_percentage < min_mis_val_per:
                stations.append(station)
        return len(stations), stations

    def stationarity_test(self, station):
        """This method test the stationary of given data set. Function divides data set to a 2 groups."""
        noy = np.shape(self.datas[station])[0]
        n1 = int(noy / 2)
        n2 = noy - n1
        flows = self.annual_summation(station)
        u = np.mean(flows)
        s = np.std(flows)
        v1 = (np.mean(flows[0:n1]) - u) / s * n1 ** 0.5
        v2 = (np.mean(flows[n1:]) - u) / s * n2 ** 0.5
        v = np.array([v1, v2])
        if np.all(v < 1.96) and np.all(v > -1.96):
            return "The values {0} inside of boundaries [1.96, -1.96] that's why " \
                   "given data set stationary for the %5 level significance level.".format(v)
        else:
            return "Given data set not stationary for the %5 level significance and the values {0} ".format(v)

    def ratio_test(self, station, plotting=False):
        """In this test (Alexandersson,1986)) whicch known ratio or Standard Normal Homogeneity test """
        annual_flow = self.annual_summation(station)
        mean, std = np.mean(annual_flow), np.std(annual_flow)
        t_values = []
        t0s = [9.56, 10.45, 11.01, 11.38, 11.89, 12.32]
        tsn = [20, 30, 40, 50, 70, 700]
        t0 = t0s[0]
        length = np.shape(self.datas[station])[0]
        for k in range(1, length):
            z1 = 1 / k * np.sum((annual_flow[0:k] - mean) / std)
            z2 = 1 / (length - k) * np.sum((annual_flow[k:] - mean) / std)
            t_values.append(k * z1 ** 2 + (length - k) * z2 ** 2)

        for t in range(len(tsn)):
            if length <= tsn[t]:
                if length == tsn[t]:
                    t0 = t0s[t]
                else:
                    t0 = (t0s[t] + t0s[t - 1]) / 2
                    break

        if plotting is True:
            axis = [self.years[station][0, 0] + i for i in range(length - 1)]
            plt.plot(axis, [t0] * len(axis))
            plt.plot(axis, t_values)
            plt.xlabel('Years')
            plt.ylabel('T(k)')
            plt.show()
        if max(t_values) <= t0:
            return 'Homogeneous'
        else:
            return "Not Homogeneous"

    def von_neumann_ratio(self, station):
        """Nr value lower than 2 and that's imply non - homogeneity"""
        anuual_flow = self.annual_summation(station)
        length = np.shape(self.datas[station])[0]
        mean = np.mean(anuual_flow)
        t1, t2 = 0, 0
        for m in range(length):
            if m < length - 1:
                t1 += (anuual_flow[m] - anuual_flow[m + 1]) ** 2
            t2 += (anuual_flow[m] - mean) ** 2
        nr = t1 / t2
        if nr <= 2:
            return 'Homogeneous'
        else:
            return "Not Homogeneous"

    def cumulative_deviation(self, station, plotting=False):
        """Buishand or cumulative_deviation test for homogeneity"""
        anuual_flow = self.annual_summation(station)
        length = np.shape(self.datas[station])[0]
        sk, skk = [0], []
        mean, std = np.mean(anuual_flow), np.std(anuual_flow)
        d_number = [10, 20, 30, 40, 50, 100, 200]
        qcs = [1.29, 1.42, 1.46, 1.5, 1.52, 1.55, 1.63]
        rcs = [1.38, 1.60, 1.70, 1.74, 1.78, 1.86, 2]
        rc, qc = rcs[2], qcs[2]
        for i in range(1, length + 1):
            total = 0
            for j in range(i):
                total += (anuual_flow[j] - mean)
            sk.append(total)
            skk.append(sk[i] / std)
        rq = (max(skk) - min(skk)) / np.sqrt(length)
        qq = np.max(np.abs(np.array(skk))) / np.sqrt(length)
        for t in range(len(d_number)):
            if length <= d_number[t]:
                if length == d_number[t]:
                    qc = qcs[t]
                    rc = rcs[t]
                else:
                    qc = (qcs[t] + qcs[t - 1]) / 2
                    rc = (rcs[t] + rcs[t - 1]) / 2
                    break

        if plotting is True:
            axis = self.years[station]
            plt.plot(axis, skk)
            plt.xlabel('Years')
            plt.ylabel('Cumulative Deviation')
            plt.show()
        if rq <= rc and qq <= qc:
            return 'Homogeneous'
        else:
            return "Not Homogeneous"

    def bayesian_statistic(self, station):
        anuual_flow = self.annual_summation(station)
        length = np.shape(self.datas[station])[0]
        mean, std = np.mean(anuual_flow), np.std(anuual_flow)
        sk, skk = [0], []
        d_number = [10, 20, 30, 40, 50, 100, 200]
        ucs = [0.575, 0.662, 0.691, 0.698, 0.718, 0.712, 0.743]
        acs = [3.14, 3.50, 3.70, 3.66, 3.78, 3.82, 3.86]
        uc, ac = ucs[2], acs[2]
        for i in range(1, length + 1):
            total = 0
            for j in range(i):
                total += (anuual_flow[j] - mean)
            sk.append(total)
            skk.append(sk[i] / std)
        for t in range(len(d_number)):
            if length <= d_number[t]:
                if length == d_number[t]:
                    uc = ucs[t]
                    ac = acs[t]
                else:
                    uc = (ucs[t] + ucs[t - 1]) / 2
                    ac = (acs[t] + acs[t - 1]) / 2
                    break
        u = 1 / (length * (length + 1)) * sum([skk[i] ** 2 for i in range(length - 1)])
        a = sum([skk[i] ** 2 / ((i + 1) * (length - i - 1)) for i in range(length - 1)])
        if u <= uc and a <= ac:
            return 'Homogeneous'
        else:
            return "Not Homogeneous"

    def pettit_test(self, station, plotting=False):
        anuual_flow = self.annual_summation(station)
        length = np.shape(self.datas[station])[0]
        ranks = stats.rankdata(anuual_flow)
        sk = []
        for k in range(1, length + 1):
            total = 0
            for i in range(k):
                total += ranks[i]
            sk.append(2 * total - k * (length + 1))

        d_number = [10, 20, 30, 40, 50, 100]
        scl = [71, 133, 208, 273, 488, 841]
        sc = scl[2]
        for t in range(len(d_number)):
            if length <= d_number[t]:
                if length == d_number[t]:
                    sc = scl[t]
                else:
                    sc = (scl[t] + scl[t - 1]) / 2
                    break
        max_sk = max([abs(i) for i in sk])
        if plotting is True:
            axis = self.years[station]
            plt.plot(axis, sk)
            plt.plot(axis, [max_sk] * len(axis))
            plt.xlabel('Years')
            plt.ylabel('Pettit Static')
            plt.show()

        
        if max_sk <= sc:
            return 'Homogeneous'
        else:
            return "Not Homogeneous"


# if __name__ == '__main__':
#     s2 = DataProcessing("prec_data.csv")
#     print(s2.choose_stations(40))
