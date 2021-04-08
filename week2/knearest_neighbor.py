from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


class K_NearestNeighbor:
    def __init__(self):
        pass

    # Calculate the distance between x1 and x2
    def distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        result = np.zeros(x2.shape[0])
        for i in range(len(x2)):
            for j in range(x2.shape[1]):
                result[i] += (x1[j] - x2[i][j]) ** 2
        return result ** 0.5

    def predict_target(self, data, k):
        """
        :param data: data should store a dataframe with all features and classifications
        :return data: data should store a dataframe with all features, classification and prediction values
        """
        target = []
        for i in range(data.shape[0]):
            x2 = data.assign(distance=self.distance(data.iloc[i, :-1], data.iloc[:, :-1]))
            x2 = x2.sort_values(by='distance')
            target.append(self.compute_nearest(x2, k))

        target = list(map(int, target))
        x = data.assign(pre_target=target)
        return (x)


    # Calculate precision and recall values
    def confusion_matrix(self, data):
        matrix = {'T0': 0, 'T1': 0, 'T2': 0, 'N0': 0, 'N1': 0, 'N2': 0}
        # num = data[(data.target == 1) & (data.pre_target ==1)].shape[0]
        for i in range(data.shape[0]):
            if data.iloc[i,-2] == 0 :
                if data.iloc[i,-1] == 0:
                    matrix['T0'] += 1
                elif data.iloc[i,-1] == 1:
                    matrix['N1'] += 1
                elif data.iloc[i, -1] == 2:
                    matrix['N2'] += 1

            elif data.iloc[i,-2] == 1 :
                if data.iloc[i,-1] == 0:
                    matrix['N0'] += 1
                elif data.iloc[i,-1] == 1:
                    matrix['T1'] += 1
                elif data.iloc[i, -1] == 2:
                    matrix['N2'] += 1

            elif data.iloc[i,-2] == 2 :
                if data.iloc[i,-1] == 0:
                    matrix['N0'] += 1
                elif data.iloc[i,-1] == 1:
                    matrix['N1'] += 1
                elif data.iloc[i, -1] == 2:
                    matrix['T2'] += 1

        # print(matrix)

    # Calculate the predicted target
    def compute_nearest(self, neiber, k):
        nearest = neiber.iloc[1:k + 1]
        result = {'0': 0, '1': 0, '2': 0}
        for i in range(nearest.shape[0]):
            result[str(int(nearest.iloc[i][-2]))] += 1 - nearest.iloc[i][-1] / np.sum(nearest['distance'])

        return max(result, key=result.get)

    # Output Accuracy Rate
    def accuracy(self, x):
        count = 0
        for i in range(x.shape[0]):
            if x.iloc[i, -1] == x.iloc[i, -2]:
                count += 1
        print('The accuracy rate is {}%'.format(count / x.shape[0] * 100))


if __name__ == '__main__':
    data = load_iris()
    x = pd.DataFrame(data.data)
    x.columns = data.feature_names
    y = data.target
    x = x.assign(target=y)

    knn = K_NearestNeighbor()
    model = knn.predict_target(x, 3)
    knn.accuracy(model)
    knn.confusion_matrix(model)