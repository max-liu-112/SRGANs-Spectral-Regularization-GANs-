import numpy as np
import copy
import six
import chainer
import chainer.functions as F
from chainer import Variable
from source.miscs.random_samples import sample_continuous, sample_categorical
from chainer.backends import cuda
from chainer import function
from source.links.sn_convolution_2d import SNConvolution2D

def SR(dis):
    links = [[name, link] for name, link in sorted(dis.namedlinks())]
    for name, link in links:
        if isinstance(link, SNConvolution2D):
            W_mat = link.W.reshape(link.W.shape[0], -1)
            xp = cuda.get_array_module(W_mat.data)
            W_cpu = chainer.cuda.to_cpu(W_mat.data)
            U, s, V = np.linalg.svd(W_cpu, full_matrices=False)
            s = s / max(s)
            s[:s.shape[0] // 2] = 1
            S = np.diag(s)
            W_cpu = np.dot(U, np.dot(S, V))
            W_mat.data = xp.asarray(W_cpu)
            link.W.copydata(W_mat.reshape(link.W.shape))

        # Classic Adversarial Loss
# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss

def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss

def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss

class Updater(chainer.training.ParallelUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        self.devices = kwargs.pop('devices')
        self.n_accumulation = kwargs.pop('num_accumulation')
        self.n_SR = 5
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError

        super(Updater, self).__init__(devices = self.devices,models = self.models, iterator = kwargs.pop('iterator'), optimizer = kwargs.pop('optimizer'))

    def _generete_samples(self, gen,  n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples
        if self.conditional:
            y = sample_categorical(gen.n_classes, n_gen_samples, xp=gen.xp)
        else:
            y = None
        x_fake = gen(n_gen_samples, y=y)
        return x_fake, y



    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real


    def connect_trainer(self, trainer):
        # Add observers for all (other) models.
        pass

    def update_core(self):
        names = list(six.iterkeys(self.devices))
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        for i in range(self.n_dis):
            # clear the gradients first
            for model in six.itervalues(self.models):
                model['gen'].cleargrads()
                model['dis'].cleargrads()
            # update D
            # first calculate the gradients
            for accumulation_index in range(self.n_accumulation):
                for name in names:
                    with function.force_backprop_mode():
                        dev_id = self.devices[name]
                        dev_id = dev_id if 0 <= dev_id else None
                        with cuda.get_device_from_id(dev_id):
                            gen = self.models[name]['gen']
                            dis = self.models[name]['dis']
                            xp = gen.xp
                            x_real, y_real = self.get_batch(xp)
                            batchsize = len(x_real)
                            dis_real = dis(x_real, y=y_real)
                            x_fake, y_fake = self._generete_samples(gen=gen, n_gen_samples=batchsize)
                            dis_fake = dis(x_fake, y=y_fake)
                            x_fake.unchain_backward()
                            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)/ float(self.n_accumulation)
                            chainer.reporter.report({'loss_dis': loss_dis})
                            loss_dis.backward()

            for name in names:
                if name != 'main':
                    self.models['main']['dis'].addgrads(self.models[name]['dis'])
            dis_optimizer.update()
            if self.iteration % self.n_SR == 0:
                SR(self.models['main']['dis'])

            for name in names:
                if name != 'main':
                    self.models[name]['dis'].copyparams(self.models['main']['dis'])
            # update G

            if i == 0:
                for model in six.itervalues(self.models):
                    model['gen'].cleargrads()
                    model['dis'].cleargrads()
                for accumulation_index in range(self.n_accumulation):
                    for name in names:
                        with function.force_backprop_mode():
                            dev_id = self.devices[name]
                            dev_id = dev_id if 0 <= dev_id else None
                            with cuda.get_device_from_id(dev_id):
                                gen = self.models[name]['gen']
                                dis = self.models[name]['dis']
                                x_fake, y_fake = self._generete_samples(gen=gen)
                                dis_fake = dis(x_fake, y=y_fake)
                                loss_gen = self.loss_gen(dis_fake=dis_fake)/ float(self.n_accumulation)
                                chainer.reporter.report({'loss_gen': loss_gen})
                                loss_gen.backward()

                for name in names:
                    if name != 'main':
                        self.models['main']['gen'].addgrads(self.models[name]['gen'])
                gen_optimizer.update()
                for name in names:
                    if name != 'main':
                        self.models[name]['gen'].copyparams(self.models['main']['gen'])

