import time
import os
import shutil


class Logger:
    def __init__(self):
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.finishtime = "Not Finished"
        self.data = ""
        self.dirname = self.create_log_dir()
        self.plot_dirname = self.create_plot_dir()

        self.t_start = 0
        self.t_finish = 0
        self.e_start = 0
        self.e_finish = 0

    @staticmethod
    def copy_folder(source_folder, destination_folder):
        try:
            shutil.copytree(source_folder, destination_folder)
            print(f"文件夹 {source_folder} 已成功复制到 {destination_folder}")
        except shutil.Error as e:
            print(f"发生错误: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")

    def finish(self, config):
        self.config_to_log(config=config)

    def create_log_dir(self, timestamp=None):
        # 创建日志文件夹
        if not os.path.exists("log_compar5"):
            os.makedirs("log_compar5")
        if timestamp is None:
            timestamp = self.timestamp
            if not os.path.exists(f"log_compar5/{timestamp}"):
                os.makedirs(f"log_compar5/{timestamp}")

        if os.path.exists(f"log_compar5/{timestamp}"):
            return f"log_compar5/{timestamp}"
        else:
            raise Exception(f"Failed to create log directory: log_compar5/{timestamp}")

    def log_runtime(self):
        with open(f"log_compar5/{self.timestamp}/runtime.txt", "w") as file:
            file.write(f"start_time: {self.t_start}\n")
            file.write(f"finish_time: {self.t_finish}\n")
            file.write(f"duration: {self.t_finish - self.t_start}\n")
            file.write(f"evaluation_start_time: {self.e_start}\n")
            file.write(f"evaluation_finish_time: {self.e_finish}\n")
            file.write(f"evaluation_duration: {self.e_finish - self.e_start}\n")

    def create_plot_dir(self, timestamp=None):
        # 创建日志文件夹
        if timestamp is None:
            timestamp = self.timestamp

        if not os.path.exists(f"log_compar5/{timestamp}/plot"):
            os.makedirs(f"log_compar5/{timestamp}/plot")

        if os.path.exists(f"log_compar5/{timestamp}/plot"):
            return f"log_compar5/{timestamp}/plot"
        else:
            raise Exception(f"Failed to create log directory: log_compar5/{timestamp}/plot")

    @staticmethod
    def get_timestamp():
        return time.strftime("%Y-%m-%d_%H-%M-%S")

    def config_to_log(self, config, timestamp=None):
        # 获取当前时间的格式化字符串

        if timestamp is None:
            timestamp = self.timestamp

        # 构造文件名
        filename = f"config_{timestamp}.txt"
        pathname = self.dirname + "/" + filename

        data = ""
        for key, value in config.items():
            data += f"{key}:{value}\n"

        # 保存文件
        with open(pathname, "w") as file:
            file.write(data)

    def plot_to_log_name(self, timestamp=None, file_format="pdf"):
        if timestamp is None:
            timestamp = self.timestamp
        filename = f"plot_{timestamp}" + "." + file_format
        pathname = self.dirname + "/" + filename
        return pathname

    def evaluate_to_log_name(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamp
        filename = f"evaluate_{timestamp}.csv"
        pathname = self.dirname + "/" + filename
        return pathname

    def save_to_disk(self, main_exp=True):
        dirname = self.dirname
        timestamp = self.timestamp
        if not os.path.exists("./.log_compar5"):
            os.makedirs("./.log_compar5")
        if main_exp:
            self.copy_folder(
                source_folder=dirname,
                destination_folder="./.log_compar5/" + self.timestamp,
            )
        else:
            self.copy_folder(
                source_folder=dirname,
                destination_folder="./.log_compar5/" + "(unsync)" + self.timestamp,
            )


def create_netplot_dir(timestamp=None):
    # 创建日志文件夹
    if timestamp is None:
        raise Exception("timestamp is None")

    if not os.path.exists(f"log_compar5/{timestamp}"):
        raise Exception(f"log_compar5/{timestamp} does not exist")

    if not os.path.exists(f"log_compar5/{timestamp}/netplot"):
        os.makedirs(f"log_compar5/{timestamp}/netplot")

    if os.path.exists(f"log_compar5/{timestamp}/netplot"):
        return f"log_compar5/{timestamp}/netplot"
    else:
        raise Exception(f"Failed to create log directory: log_compar5/{timestamp}/netplot")


def create_replot_dir(timestamp=None):
    # 创建日志文件夹
    if timestamp is None:
        raise Exception("timestamp is None")

    if not os.path.exists(f"log_compar5/{timestamp}"):
        raise Exception(f"log_compar5/{timestamp} does not exist")

    if not os.path.exists(f"log_compar5/{timestamp}/replot"):
        os.makedirs(f"log_compar5/{timestamp}/replot")

    if os.path.exists(f"log_compar5/{timestamp}/replot"):
        return f"log_compar5/{timestamp}/replot"
    else:
        raise Exception(f"Failed to create log directory: log_compar5/{timestamp}/replot")


class Experiment_Agent:
    def __init__(self, config, id=None):
        self.config = config
        self.exp_id = 1

        if not os.path.exists("experiment_compar"):
            os.makedirs("experiment_compar")

        if id is not None:
            self.exp_id = id
        else:
            i = 1
            while os.path.exists(f"experiment_compar/exp_{i}.txt"):
                i += 1
            self.exp_id = i

        if not os.path.exists(f"experiment_compar/exp_{self.exp_id}.txt"):
            self.init_new_exp()

    def init_new_exp(self):
        # 创建一个记录实验的txt文件

        with open(f"experiment_compar/exp_{self.exp_id}.txt", "w") as file:
            # 写入文字到文件中
            file.write("times:0\n")

            data = ""
            for key, value in self.config.items():
                data += f"{key}:{value}\n"

            file.write(data)

    def delete_last_exp_epoch(self):
        with open(f"experiment_compar/exp_{self.exp_id}.txt", "r") as file:
            # 读取文件中的内容
            data = file.readlines()

        times = int(data[0].split(":")[1])
        times -= 1
        data[0] = f"times:{times}\n"
        data.pop()

        with open(f"experiment_compar/exp_{self.exp_id}.txt", "w") as file:
            # 写入文字到文件中
            file.writelines(data)

    def run_exp(self, timestamp=None):
        with open(f"experiment_compar/exp_{self.exp_id}.txt", "r") as file:
            # 读取文件中的内容
            data = file.readlines()

        times = int(data[0].split(":")[1])
        times += 1
        data[0] = f"times:{times}\n"
        data.append(f"timestamp:{timestamp}\n")

        print(data)

        with open(f"experiment_compar/exp_{self.exp_id}.txt", "w") as file:
            # 写入文字到文件中
            file.writelines(data)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S")
