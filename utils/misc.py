# coding=utf-8
from init import Infrastructure


def finish_experiment(infra: Infrastructure, experiment_id: str):
    c = infra.conn.cursor()
    c.execute('''
                 UPDATE experiments SET ended=1 WHERE idmd5=?
              ''', (experiment_id,))
    infra.conn.commit()
    if c.rowcount > 0:
        return True
    else:
        return False
