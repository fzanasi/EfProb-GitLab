import unittest
import csl

class TestCsl(unittest.TestCase):

    def setUp(self):
        self.my_cslInterpreter = csl.cslInterpreter()

    def drive(self,filename):
        try:
            programCsl = open(filename)
            sourceCode = programCsl.read()
            programCsl.close()
            self.my_cslInterpreter.programName = programCsl.name
            self.my_cslInterpreter.run(sourceCode)
        except Exception as e:
            self.assertRaises(e)

    def test_basic(self):
        """ Programma basic.csl """
        self.drive("test/basic.csl")

    def test_basic(self):
        """ Programma pythonEscape.csl """
        self.drive("test/pythonEscape.csl")

if __name__ == '__main__':
    unittest.main()
