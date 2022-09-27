import unittest

from gaussian.logparser.base import FileLogLoader
from gaussian.logparser.drain import DrainLogParser


class TestDrainParser(unittest.TestCase):

    def test_drain_parser_with_parameters(self):
        input_log_file = "/Users/renzhiqiang/Workspace/aiops/loghub/HDFS/HDFS_2k.log"
        output_dir = "output/"
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

        log_df = FileLogLoader(logfile=input_log_file,
                               logformat="<Date> <Time> <Pid> <Level> <Component>: <Content>").load()

        regex = [
            r'(blk_(|-)[0-9]+ ?)+',  # block id and block id list
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]

        model = DrainLogParser(outdir=output_dir, log_format=log_format, rex=regex)
        log_df = model.parse(log_df)
        self.assertEqual(len(log_df), 2000)
        self.assertTrue('EventId' in log_df.columns)
        self.assertTrue('EventTemplate' in log_df.columns)
        self.assertEqual("PacketResponder <*> for block <*>terminating", log_df['EventTemplate'][0])
        self.assertEqual(['1', 'blk_38865049064139660 '], log_df['ParameterList'][0])
