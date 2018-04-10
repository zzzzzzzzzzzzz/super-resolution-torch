# coding: utf-8

import argparse
from init import Infrastructure

infra = Infrastructure()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name of the metric', required=True)
    parser.add_argument('--description', type=str, help="Some words about this metric")
    opt = parser.parse_args()
    print(opt)

    c = infra.conn.cursor()
    c.execute('''
                 INSERT INTO metrics (name, description) VALUES (?, ?)
              ''', (opt.name, opt.description))
    infra.conn.commit()