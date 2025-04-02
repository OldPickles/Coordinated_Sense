import numpy as np


def demo_function(**kwargs):
    print(type(kwargs))
    for key, value in kwargs.items():
        print(key, value)
if __name__ == '__main__':
    demo_function(a=1, b=2, c=3)