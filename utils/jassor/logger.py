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
        self.enter = start
        self.indentation = indentation
        self.level = level

    def open(self):
        with self.con:
            if isinstance(self.output_file, (str, Path)):
                self.output_file = open(self.output_file, 'a')
            self.output_file.write('#%s enter at %.2f seconds\n' % ('\t' * self.indentation, time() - self.start))
            self.output_file.flush()
            self.enter = time()
        return self.tab()

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

    def close(self):
        with self.con:
            self.output_file.write('#%s exit at %.2f seconds -- costing %.2f seconds\n' % ('\t' * self.indentation, time() - self.start, time() - self.enter))
            self.output_file.flush()
            if self.close_file:
                self.output_file.close()


# L = Logger()
# with L:
#     raise Exception
