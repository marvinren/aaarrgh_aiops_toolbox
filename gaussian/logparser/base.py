import abc

import regex as re
import pandas as pd


class FileLogLoader:
    """
    日志加载&预处理：完成对日志文件的加载，并根据日志格式对日志进行预处理，最终将日志转换为DataFrame
    """

    def __init__(self, logfile: str, logformat: str = None):
        self.logfile = logfile
        self.logformat = logformat

    def load(self) -> pd.DataFrame:
        """
        加载数据
        :return: 日志的dataframe格式的
        """
        headers, regex = self.generate_log_format_regex()
        return self.log_to_dataframe(self.logfile, regex, headers)

    def generate_log_format_regex(self):
        """
        根据日志格式生成提取日志的正则表达式，日志格式中的每个字段用<>包裹，例如：
        <date> <time> <level> <message>
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', self.logformat)
        regex = ''
        for k in range(len(splitters)):
            # if splitters[k].strip() == '':
            #     continue
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(self, log_file, regex, headers):
        """
        将加载日志文件转换为DataFrame，其中包括利用日志格式logformat对日志进行预处理
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


class BaseLogParser(abc.ABC):

    @abc.abstractmethod
    def parse(self, log_df: pd.DataFrame) -> pd.DataFrame:
        pass