# 以下两行可切换进程、线程选用
from .base_thread import Process, Queue
# from .base_process import Process, Queue
from .logger import Logger
from .timer import Timer
from .merger import *


__all__ = ['Process', 'Queue', 'Logger', 'Timer', 'MergerInterface', 'MemoryMerger', 'RowMerger', 'ColMerger', 'DictMerger']
