import numpy as np
import matplotlib.pyplot as plt


def loadDatas(filename, delimiter, a=0, b=2):
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        line_data = line.strip().split(delimiter)
        data.append(line_data)
    datas = np.array(data).astype(np.float64)[:, a:b]
    return datas


class Agens:
    def __init__(self, datas, k, method):
        self.datas = datas
        self.k = k
        self.method = method

        N, D = np.shape(datas)
        self.cluster_set = []
        self.cluster_index = []
        self.N = N

        tile_x = np.tile(np.expand_dims(self.datas, 1), [1, N, 1])
        tile_y = np.tile(np.expand_dims(self.datas, 0), [N, 1, 1])
        self.dims_matrix_datas = np.linalg.norm(tile_x - tile_y, axis=-1)

        for i in range(N):
            self.cluster_set.append(np.expand_dims(self.datas[i], 0))
        self.cluster_index = [[i] for i in range(N)]
        self.dis_matrix_cluster = self.dims_matrix_datas.copy()
        for i in range(N):
            self.dis_matrix_cluster[i, i] = np.inf

    def dis_bw2cluster(self, inds_x, inds_y):
        dis = [self.dims_matrix_datas[x, y] for x in inds_x for y in inds_y]
        if self.method == 'avg':
            return np.mean(dis)
        elif self.method == 'min':
            return np.min(dis)
        elif self.method == 'max':
            return np.max(dis)

    def update_dis_matrix_cluster(self, ind_x, ind_y):
        self.dis_matrix_cluster = np.delete(self.dis_matrix_cluster, ind_y, axis=0)
        self.dis_matrix_cluster = np.delete(self.dis_matrix_cluster, ind_y, axis=1)

        N_cluster = len(self.cluster_set)
        for i in range(N_cluster):
            if i == ind_x:
                self.dis_matrix_cluster[i, i] = np.inf
            else:
                dis = self.dis_bw2cluster(self.cluster_index[i], self.cluster_index[ind_x])
                self.dis_matrix_cluster[i, ind_x] = dis
                self.dis_matrix_cluster[ind_x, i] = dis

    def find_min(self):
        ind_x, ind_y = np.where(self.dis_matrix_cluster == np.min(self.dis_matrix_cluster))
        return ind_x[0], ind_y[0]

    def fit(self, display=True):
        # 开始时簇的数目
        q = len(self.cluster_set)
        n_round = 0
        while q > self.k:
            ind_x, ind_y = self.find_min()
            datas_x = self.cluster_set[ind_x]
            datas_y = self.cluster_set[ind_y]
            self.cluster_set[ind_x] = np.concatenate((datas_x, datas_y), axis=0)
            # 2
            self.cluster_index[ind_x] = self.cluster_index[ind_x] + self.cluster_index[ind_y]
            # 3
            self.cluster_set.pop(ind_y)
            self.cluster_index.pop(ind_y)
            # 更新
            self.update_dis_matrix_cluster(ind_x, ind_y)

            q = len(self.cluster_set)
            n_round += 1
            print('n_round = %d n_cluster = %d' % (n_round, q))
        if display:
            plt.ion()
            draw(self.cluster_set, self.cluster_index, self.N,
                 str_title='n_round=%d n_cluster=%d linkage=%s' % (n_round, q, self.method))
            plt.pause(0.1)
            plt.ioff()
        return self.cluster_set, self.cluster_index


def draw(cluster_set, cluster_index, N, str_title=''):
    plt.cla()
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1.1, N)]
    for i in range(len(cluster_set)):
        datas = cluster_set[i]
        # s = int(datas.shape[0] * 5)
        s = 10
        color_index = cluster_index[i][0]
        if color_index > N:
            color_index = i
        color = colors[color_index]
        plt.scatter(datas[:, 0], datas[:, 1], s=s, color=color)

    plt.title(str_title)
    plt.show()


if __name__ == '__main__':
    datas = loadDatas('D:\\聚类\\数据集\\数据集\\数据集\\人工数据集\\Spiral.txt', ',', 0, 2)
    k = 3
    m_agens = Agens(datas, k, method='avg')
    cluster_set, cluster_index = m_agens.fit()
    draw(cluster_set, cluster_index, k)
