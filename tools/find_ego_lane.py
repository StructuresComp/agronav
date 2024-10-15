import numpy as np

def find_ego_idx(self, key_points):
       dist_x = []
       idxs = []
       for idx, key_point in enumerate(key_points):
           p_y = np.polyfit(key_point['y'], key_point['x'], self.N_DEGREE)
           x_min = np.polyval(p_y, 0)
           dist_x.append(x_min-0.5)
           idxs.append(idx)
       dist_x = np.array(dist_x)
       idxs = np.array(idxs)
       idxs_l = idxs[dist_x < 0]
       xs_l = dist_x[dist_x < 0]
       
       idxs_r = idxs[dist_x >= 0]
       xs_r = dist_x[dist_x >= 0]
       return idxs_l[np.argmax(xs_l)], idxs_r[np.argmin(xs_r)]