import argparse
import logging
import dataset
from model import ListNet

logging.basicConfig(level=logging.INFO)
"""
    entrance
"""


def run(args):
    logging.info("Loading data...")
    id, x_train, x_test, y_train, y_test = dataset.load_random_data()
    logging.info("loading {} id".format(len(id)))
    logging.info("loading {} items data for training, dims {}".format(x_train.shape[0], x_train.shape[1]))
    logging.info("loading {} items data as training label, dims {}".format(y_train.shape[0], y_train.shape[1]))
    logging.info("loading {} items data for testing, dims {}".format(x_test.shape[0], x_test.shape[1]))
    logging.info("loading {} items data as testing result, dims {}".format(y_test.shape[0], y_test.shape[1]))
    listnet = ListNet(h1_input=x_train.shape[1],
                      h1_output=10,
                      h2_input=10,
                      h2_output=1)
    listnet.train(x_train, y_train, 100, 0.001, optmz='adam')
    listnet.predict(x_test)
    perms=listnet.permutation(id)
    print(perms)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train',
                   type=str,
                   help='input train json file',
                   default='random')
    p.add_argument('--env',
                   required=False,
                   help='input dev/prod to alter environment',
                   default='dev')
    p.print_help()
    args = p.parse_args()
    print(args)
    run(args)