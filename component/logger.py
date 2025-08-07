import traceback
from typing import Union
from time import time
from threading import Condition
import sys
from io import TextIOWrapper
from pathlib import Path


class Logger(object):
    DEBUG = 1
    INFO = 2
    WARNING = 3

    def __init__(self, start: int = 0, indentation: int = 0, file: Union[TextIOWrapper, str, Path] = sys.stdout, con: Condition = None, level: int = 2):
        """
        @TODO 记得写注释
        :param start:
        :param indentation:
        :param file:
        :param con:
        :param level:
        """
        self.close_file = isinstance(file, (str, Path))
        self.output_file = file
        self.con = con or Condition()
        self.start = start or time()
        self.indentation = indentation
        self.level = level
        with self.con:
            if isinstance(self.output_file, (str, Path)):
                self.output_file = open(self.output_file, 'a')

    def close(self):
        with self.con:
            if self.close_file:
                self.output_file.close()

    def track(self, message: str, prefix: str = ''):
        with self.con:
            self.output_file.write('# %s%s %s -> at time %.2f\n' % (prefix, '\t' * self.indentation, message, time() - self.start))
            self.output_file.flush()

    def debug(self, message: str):
        if self.level <= Logger.DEBUG: self.track(message, prefix='DEBUG')

    def info(self, message: str):
        if self.level <= Logger.INFO: self.track(message, prefix='INFO')

    def warn(self, message: str):
        if self.level <= Logger.WARNING: self.track(message, prefix='WARNING')

    def tab(self):
        with self.con:
            return Logger(start=self.start, indentation=self.indentation+1, file=self.output_file, con=self.con)

    def __enter__(self):
        self.enter = time()
        with self.con:
            self.output_file.write('#%s enter at %.2f seconds\n' % ('\t' * self.indentation, time() - self.start))
            self.output_file.flush()
        return self.tab()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            sys.stderr.write('#%s -------------------------------------- Error Message ---------------------------------------------- \n')
            sys.stderr.write(f'#%s Error: exc_type - {exc_type} \n')
            sys.stderr.write(f'#%s Error: exc_val - {exc_val} \n')
            sys.stderr.write(f'#%s Error: exc_tb - {exc_tb} \n')
            sys.stderr.write('#%s --------------------------------------------------------------------------------------------------- \n')
            sys.stderr.flush()
            traceback.print_exc()
        else:
            self.output_file.write('#%s exit at %.2f seconds -- costing %.2f seconds\n' % ('\t' * self.indentation, time() - self.start, time() - self.enter))
            self.output_file.flush()
        return False


# L = Logger()
# with L:
#     raise Exception
