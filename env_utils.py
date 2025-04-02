"""
This file contains utility functions for the environment.
1. 环境的元素对象
"""

class EnvObject:
    flag_content = 1  # 表示旗子的数据内容
    uav_capacity = 5  # 表示无人机的数据容量
    server_capacity = 100  # 表示服务器的数据容量

    ont_host_base = [0, 0, 0, 0, 0]
    phone_one_hot = [1, 0, 0, 0, 0]  # 电话的one-hot编码
    uav_one_hot = [0, 1, 0, 0, 0]  # 无人机的one-hot编码
    flag_one_hot = [0, 0, 1, 0, 0]  # 旗子的one-hot编码
    obstacle_one_hot = [0, 0, 0, 1, 0]  # 障碍物的one-hot编码
    server_one_hot = [0, 0, 0, 0, 1]  # 服务器的one-hot编码

    def __init__(self, name=None, position=None, content=0, capacity=0, tk_id=None):
        """
        初始化环境元素对象
        :param name: 元素名称："phone", "uav", "flag", "obstacle", "server"
        :param position: 元素位置
        :param content: 元素数据内容,只有 flag， uav 和 server 有
        :param capacity: 元素数据容量,只有 uav 和 server 有
        :param tk_id: 元素的tkinter id,用于绘制元素
        """
        if position is None:
            position = [0, 0]
        self.name = name
        self.position = position
        self.content = content
        self.capacity = capacity
        self.tk_id = tk_id

        self.one_hot = None
        self.init_one_hot()
        self.init_capacity()
        self.init_content()

    def init_one_hot(self):
        """
        初始化one-hot编码
        :return:
        """
        if self.name == "phone":
            self.one_hot = EnvObject.phone_one_hot
        elif self.name == "uav":
            self.one_hot = EnvObject.uav_one_hot
        elif self.name == "flag":
            self.one_hot = EnvObject.flag_one_hot
        elif self.name == "obstacle":
            self.one_hot = EnvObject.obstacle_one_hot
        elif self.name == "server":
            self.one_hot = EnvObject.server_one_hot
        else:
            raise ValueError("Invalid object name: {}".format(self.name))
        pass

    def init_capacity(self):
        """
        初始化元素数据容量
        uav 的数据容量为 5， server 的数据容量为 100
        :return:
        """
        if self.name == "uav":
            self.capacity = EnvObject.uav_capacity
        elif self.name == "server":
            self.capacity = EnvObject.server_capacity
        else:
            pass
        pass

    def init_content(self):
        """
        初始化元素数据内容
        flag 的数据内容为 1，其他元素的数据内容为 0
        :return:
        """
        if self.name == "flag":
            self.content = 1
        else:
            self.content = 0
        pass

    def set_tk_id(self, tk_id):
        """
        设置元素的tkinter id
        :param tk_id:
        :return:
        """
        self.tk_id = tk_id
        pass

    def set_content(self, content):
        """
        设置元素数据内容
        :param content:
        :return:
        """
        self.content = content
        pass

    def set_position(self, position):
        """
        设置元素位置
        :param position:
        :return:
        """
        self.position = position
        pass

    def set_capacity(self, capacity):
        """
        设置元素数据容量
        :param capacity:
        :return:
        """
        self.capacity = capacity
        pass

    def get_name(self):
        """
        获取元素名称
        :return:
        """
        return self.name

    def get_position(self):
        """
        获取元素位置
        :return:
        """
        return self.position

    def get_content(self):
        """
        获取元素数据内容
        :return:
        """
        return self.content

    def get_capacity(self):
        """
        获取元素数据容量
        :return:
        """
        return self.capacity

    def get_tk_id(self):
        """
        获取元素的tkinter id
        :return:
        """
        return self.tk_id

    def get_one_hot(self):
        """
        获取元素的one-hot编码
        :return:
        """
        return self.one_hot