import unittest

from gaussian.logparser.base import FileLogLoader


class TestLogLoader(unittest.TestCase):

    def testLoadData(self):
        df = FileLogLoader(logfile="/Users/renzhiqiang/Workspace/aiops/loghub/HDFS/HDFS_2k.log", logformat="<Date> <Time> <Pid> <Level> <Component>: <Content>").load()
        self.assertEqual(len(df), 2000)
        self.assertListEqual(df.columns.values.tolist(), ['LineId', 'Date', 'Time', 'Pid', 'Level', 'Component', 'Content'])