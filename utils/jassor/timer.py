import traceback
import time
import sys


class Timer(object):
    def __init__(self):
        """
        这是改版后的计时工具库，它可以构成一个树结构，因此可以从 root 节点找到一切子计时器，如果有必要的话
        对每个计时器，使用 with 语法块以自动调用 enter、exit 函数实现计时功能
        例：
        T = Timer()
        for i in range(1000):
            x = np.array([[i * j for j in range(1000)]])
            y = x.T
            x = x ** 2
            y = y - i*i
            with T:
                z = np.matmul(y, x)
        print(T.time())
        在上面这个假想的例子中，我们希望生成一千组数据并计算它们的矩阵乘，但我们只想统计矩阵乘的用时，而对生成数据的用时不感兴趣
        此时，我们的 with 块只包含 matmul，因此我们只统计了矩阵乘用时，而将数据的预处理和生成工作用时排除在外
        当我们希望统计多段代码用时时，我们不必创建多个 Timer，而可以这样做：
        T = Timer()
        with T['block1']:
            do something ...
        with T['block2']:
            do something ...
        print(T['block1'])
        print(T['block2'])
        值得注意的是，该计时器的实现本身效率比较低，无法用于统计高频代码耗时（每运行一万次，耗时百分之一秒，请自行评估合适的使用频率）
        """
        self.cost = 0
        self.sub_timers = {}

    def __getitem__(self, item: str):
        if item not in self.sub_timers:
            self.sub_timers[item] = Timer()
        return self.sub_timers[item]

    def __enter__(self):
        self.enter = time.time()

    def time(self) -> float:
        return self.cost

    def stamp(self) -> str:
        t = self.time()
        r = f'{round(t * 1000 % 1000)}ms'
        t = int(t)
        for n, s in zip(
                [60, 60, 24, 1024],
                ['s', 'm', 'h', 'd'],
        ):
            x = t % n
            t = t // n
            r = f'{x}{s} ' + r
            if t == 0: break
        return r

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
            self.cost += time.time() - self.enter
        return False


# import numpy as np
# t1 = time.time()
# T = Timer()
# for i in range(100000):
#     # 使用 with T
#     with T:
#         x = np.random.randn(16)
#         x = np.expand_dims(x, axis=0)
#         y = x.T
#         z = np.matmul(y, x).sum()
#         print('rst', z)
#     # 或者不使用 with T
#     # x = np.random.randn(16)
#     # x = np.expand_dims(x, axis=0)
#     # y = x.T
#     # z = np.matmul(y, x).sum()
#     # print('rst', z)
# t2 = time.time()
# print('T', T.time())
# print('t', t2 - t1)
