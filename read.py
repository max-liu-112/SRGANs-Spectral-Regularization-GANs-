import numpy as np
import os


preview_dir = './dis_sigmas'
dis_dir = './dis_sigmas/dis'
if not os.path.exists(dis_dir):
    os.makedirs(dis_dir)
for i in range(10):
    arrys = []
    for iters in range(1000, 50000,5000):
        preview_path = preview_dir + '/sigmas_{:0>8}_{}.txt'.format(iters, i)
        a = np.loadtxt(preview_path)
        arrys.append(a)
    arrys = np.asarray(arrys).transpose([1, 0])
    print(arrys.shape)
    dis_path = dis_dir + '/{}.txt'.format(i)
    np.savetxt(dis_path, arrys)


