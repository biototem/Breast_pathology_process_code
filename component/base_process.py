from typing import TypeVar
import queue
import threading


V = TypeVar('V')

# 本类存在的意义是自由切换线程、进程写法，屏蔽无关差异，并提供一个相对友善的队列封装
# base_thread 是线程写法封装


class Process(threading.Thread):
    def __init__(self, ):
        super(Process, self).__init__()

    def start(self) -> None:
        super(Process, self).start()

    def run(self) -> None:
        raise NotImplemented


class Queue(object):
    """
    这是一个简易的队列封装设计，原版队列好头大，没有 top，没有 has_next，用起来好诡异的感觉
    经过封装后，队列获得了明确的状态标识：
    无元素时，可根据 has_next 判断队列是否结束
    有元素时，可调用 top 获取下一个元素但不弹出
    声明结束时，只需 push 两个 None 即可
    """
    def __init__(self, maxsize: int = 0):
        self._top = True
        self.con = threading.Condition()
        self.que = queue.Queue(maxsize=maxsize)

    def ready(self) -> bool:
        return self.que.qsize() > 0 and self.top() is not None

    def push(self, item: V) -> None:
        self.que.put(item)

    def has_next(self) -> bool:
        return self._top is True or self.top() is not None

    def top(self) -> V:
        with self.con:
            if self._top is True:
                self._top = self.que.get()
        return self._top

    def pop(self) -> V:
        with self.con:
            item = self.top()
            if item is not None:
                self._top = True
        return item

    def end(self, flag: bool = True) -> None:
        """
        结束队列，含两种结束方式，正常结束或异常结束
        flag == True: 正常结束，即，当前队列用完后不再接受新数据，正常退出
        flag == False： 异常结束，即，立即结束队列的正常读写，向上下游任务发送异常报告
        目前只做正常结束的情况：
        # 约定：push 两个 None 作为队列结束标志          ---- 如果需要队列允许传递 None 的话，可以在这个文件中自己定一一个 EMPTY 对象代替 None
        # 这里选择 push 两个是出于代码设计安全考虑
        # 理论上一个就够了，但万一存在某种极端情况，两个进程 / 线程同时 pop
        # 就可能造成队列弹出两次
        # 而由于队列已经结束，第二次弹出永远不会完成
        # 从而进程 / 线程直接脱离控制，永不结束
        # 这是在设计阶段就要严格规避的问题
        """
        self.que.put(None)
        self.que.put(None)
