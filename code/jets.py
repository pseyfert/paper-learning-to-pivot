import numpy as np
import pickle
import sys


import h5py

# Get data from http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/
import os
if not os.path.isdir("hepjets"):
    try:
        os.makedirs("hepjets")
    except OSError as e:
        print e.strerror
        sys.exit(e.errno)
    import urllib
    urllib.urlretrieve("http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/test_no_pile_5000000.h5",filename="hepjets/test_no_pile_5000000.h5")
    urllib.urlretrieve("http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/test_pile_5000000.h5",filename="hepjets/test_pile_5000000.h5")
    urllib.urlretrieve("http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/train_no_pile_10000000.h5",filename="hepjets/train_no_pile_10000000.h5")
    urllib.urlretrieve("http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/train_pile_10000000.h5",filename="hepjets/train_pile_10000000.h5")

if not os.path.exists("jets-pile.pickle"):

    f = h5py.File("hepjets/test_no_pile_5000000.h5", "r")
    X_no_pile = f["features"].value
    y_no_pile = f["targets"].value.ravel()

    f = h5py.File("hepjets/test_pile_5000000.h5", "r")
    X_pile = f["features"].value
    y_pile = f["targets"].value.ravel()


    from sklearn.cross_validation import train_test_split

    X = np.vstack((X_no_pile, X_pile))
    y = np.concatenate((y_no_pile, y_pile)).ravel()
    z = np.zeros(len(X))
    z[len(X_no_pile):] = 1

    strates = np.zeros(len(X))
    strates[(y==0)&(z==0)]=0
    strates[(y==0)&(z==1)]=1
    strates[(y==1)&(z==0)]=2
    strates[(y==1)&(z==1)]=3

    from keras.utils import np_utils
    z = np_utils.to_categorical(z.astype(np.int))

    from sklearn.preprocessing import StandardScaler
    tf = StandardScaler()
    X = tf.fit_transform(X)

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=25000, random_state=1, stratify=strates)
    pickle.dump((X_train, X_test, y_train, y_test, z_train, z_test),open("jets-pile.pickle","wb"))
else:
    X_train, X_test, y_train, y_test, z_train, z_test = pickle.load(open("jets-pile.pickle","rb"))

X_train = X_train[:150000]
y_train = y_train[:150000]
z_train = z_train[:150000]


import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam

import argparse
parser = argparse.ArgumentParser(
        description = "Run training"
        )
parser.add_argument(dest="Lambda",type=float,help="lambda parameter")
parser.add_argument(dest="random_seed",type=int,help="random seed")
options = parser.parse_args()
lam = options.Lambda
seed = options.random_seed
np.random.seed = seed

# Prepare data
X = X_train
y = y_train
z = z_train

# mask = (z_train[:, 0] == 1)
# X_train = X_train[mask]
# y_train = y_train[mask]
# z_train = z_train[mask]

# Set network architectures
inputs = Input(shape=(X.shape[1],))
Dx = Dense(64, activation="tanh")(inputs)
Dx = Dense(64, activation="relu")(Dx)
Dx = Dense(64, activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)
D = Model(input=[inputs], output=[Dx])

Rx = D(inputs)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(z.shape[1], activation="softmax")(Rx)
R = Model(input=[inputs], output=[Rx])


def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * K.binary_crossentropy(y_pred, y_true)
    return loss_D


def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * K.categorical_crossentropy(z_pred, z_true)
    return loss_R


opt_D = Adam()
D.compile(loss=[make_loss_D(c=1.0)], optimizer=opt_D)

opt_DRf = SGD(momentum=0)
DRf = Model(input=[inputs], output=[D(inputs), R(inputs)])
DRf.compile(loss=[make_loss_D(c=1.0),
                  make_loss_R(c=-lam)],
            optimizer=opt_DRf)

opt_DfR = SGD(momentum=0)
DfR = Model(input=[inputs], output=[R(inputs)])
DfR.compile(loss=[make_loss_R(c=1.0)],
            optimizer=opt_DfR)

# Pretraining of D
D.trainable = True
R.trainable = False
D.fit(X_train, y_train, nb_epoch=20)

# Pretraining of R
if lam > 0.0:
    D.trainable = False
    R.trainable = True
    DfR.fit(X_train[y_train == 0], z_train[y_train == 0], nb_epoch=20)
    # DfR.fit(X_train, z_train, nb_epoch=20)

# Adversarial training
batch_size = 128

for i in range(1001):
    print(i)

    # Fit D
    D.trainable = True
    R.trainable = False
    indices = np.random.permutation(len(X_train))[:batch_size]
    DRf.train_on_batch(X_train[indices], [y_train[indices], z_train[indices]])

    # Fit R
    if lam > 0.0:
        D.trainable = False
        R.trainable = True

        DfR.fit(X_train[y_train == 0], z_train[y_train == 0],
                batch_size=batch_size, nb_epoch=1, verbose=0)
        # DfR.fit(X_train, z_train,
        #         batch_size=batch_size, nb_epoch=1, verbose=0)

D.save_weights("D-%.4f-%d-z=0.h5" % (lam, seed))
