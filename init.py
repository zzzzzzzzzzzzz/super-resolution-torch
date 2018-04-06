# coding: utf-8
import os
import sqlite3


class Infrastructure(object):
    """
    The object of such class will initialize all folders, dbs, environment variables...
    It's also decompress all archives to corresponding folders (in future)
    """
    GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, *args, **kwargs):
        self.snapshots_path = self.create_folder_or_pass('snapshots')
        self.logs_path = self.create_folder_or_pass('logs')
        self.models_path = self.create_folder_or_pass('models')
        self.datasets_path = self.create_folder_or_pass('datasets')
        self.scripts_path = self.create_folder_or_pass('scripts')
        self.transforms_path = self.create_folder_or_pass('transforms')
        self.metrics_path = self.create_folder_or_pass('metrics')
        self.create_file_or_pass(os.path.join(self.models_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.scripts_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.metrics_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.transforms_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.datasets_path, '__init__.py'))
        self.create_db_or_pass('db.sqlite3')

    def create_folder_or_pass(self, which):
        path = os.path.join(self.GLOBAL_PATH, which)
        if os.path.exists(path):
            if os.path.isdir(path):
                return path
            else:
                raise Exception("expected folder, got file or something else {}".format(path))
        else:
            os.makedirs(path)
            return path

    def create_file_or_pass(self, which):
        path = os.path.join(self.GLOBAL_PATH, which)
        if os.path.exists(path):
            if os.path.isfile(path):
                return path
            else:
                raise Exception("expected file, got folder or something else {}".format(path))
        else:
            with open(path, 'w') as f:
                pass
            return path

    def create_db_or_pass(self, dbname):
        path = os.path.join(self.GLOBAL_PATH, dbname)
        if os.path.exists(path):
            return path
        else:
            conn = sqlite3.connect(path)
            c = conn.cursor()

            c.execute('''
                          CREATE TABLE 
                            datasets
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT UNIQUE, 
                          path TEXT, 
                          description TEXT, 
                          train_size INTEGER, 
                          test_size INTEGER)
                      ''')

            c.execute('''
                          CREATE TABLE 
                            models
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT UNIQUE, 
                          path TEXT, 
                          description TEXT, 
                          params TEXT)
                      ''')

            c.execute('''
                          CREATE TABLE 
                            metrics
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT UNIQUE, 
                          description TEXT)
                      ''')

            c.execute('''
                          CREATE TABLE 
                            experiments
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          model_id INTEGER,
                          dataset_id INTEGER,
                          train_params TEXT, 
                          description TEXT,
                          dt DATETIME)
                      ''')

            c.execute('''
                          CREATE TABLE 
                            metric_values
                         (experiment_id INTEGER,
                          metric_id INTEGER,
                          value REAL, 
                          PRIMARY KEY (experiment_id, metric_id),
                          FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                          FOREIGN KEY (metric_id) REFERENCES  metrics(id))
                      ''')

            # Save (commit) the changes
            conn.commit()

            # We can also close the connection if we are done with it.
            # Just be sure any changes have been committed or they will be lost.
            conn.close()
            return path


if __name__ == '__main__':
    infra = Infrastructure()
