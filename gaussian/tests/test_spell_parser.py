import unittest

from gaussian.logparser.base import FileLogLoader
from gaussian.logparser.spell import SpellLogParser


class TestSpellParser(unittest.TestCase):

    def test_spell_parser_with_parameters(self):
        input_dir = "/Users/renzhiqiang/Workspace/aiops/loghub/HDFS/HDFS_2k.log"
        output_dir = "output/"
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

        log_df = FileLogLoader(logfile="/Users/renzhiqiang/Workspace/aiops/loghub/HDFS/HDFS_2k.log",
                               logformat="<Date> <Time> <Pid> <Level> <Component>: <Content>").load()
        tau = 0.5
        regex = [
            r'(blk_(|-)[0-9]+ ?)+',  # block id and block id list
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        model = SpellLogParser(outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=True)
        log_df = model.parse(log_df)
        # model.printTree(model.rootNode, 0)

        self.assertEqual(len(log_df), 2000)
        self.assertTrue('EventId' in log_df.columns)
        self.assertTrue('EventTemplate' in log_df.columns)
        self.assertEqual("PacketResponder <*> for block <*>terminating", log_df['EventTemplate'][0])
        self.assertEqual(['1', 'blk_38865049064139660'], log_df['ParameterList'][0])
