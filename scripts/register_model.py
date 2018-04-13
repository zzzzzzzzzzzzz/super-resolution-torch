# coding: utf-8

# TODO: hope someday it will do some useful work. Myabe implementing general runner is the good idea?
import sys
sys.path.append('..')
import argparse
from init import Infrastructure

infra = Infrastructure()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name of the model', required=True)
    parser.add_argument('--path', type=str, help='path to model class file', required=True)
    parser.add_argument('--description', type=str, help="Some words about this model")
    opt = parser.parse_args()
    print(opt)

    c = infra.conn.cursor()
    c.execute('''
                 INSERT INTO models (name, path, description) VALUES (?, ?, ?)
              ''', (opt.name, opt.path, opt.description))
    infra.conn.commit()