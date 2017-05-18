import efprob_dc as efp
from cslParser import cslParser

###### Compiler: Code generation ######
class cslCompiler(object):
    def __init__(self,single=False):
        self.single = single
        self.parser = cslParser(single=single)

    def compile(self, code):
        self.syntaxTree = self.parser.parse(code)
        if self.single:
            mode = 'single'
        else:
            mode = 'exec'
        return compile(self.syntaxTree,filename="<ast>", mode=mode)

###### Compiled Code Execution ######

class cslInterpreter(object):
    def __init__(self, single=False, debug=False):
        self.debug = debug
        self.single = single
        self.compiler = cslCompiler(single=self.single)

    def run(self, sourceCode):
        compiled = self.compiler.compile(sourceCode)
        exec(compiled,globals())
        if self.debug:
            import ast
            if self.programName:
                sys.stdout = open(self.programName + ".ast", "w")
            print(ast.dump(self.compiler.syntaxTree))

import argparse
import sys

if __name__ == "__main__":
    optionParser = argparse.ArgumentParser(description='Effectus Probability Language interpreter.')
    optionParser.add_argument('cslProgram', metavar='prg', nargs='?',
                    type=argparse.FileType('r'),
                    default=sys.stdin,
                    help='csl program path file (e.g. test/0.csl)')
    optionParser.add_argument('--debug', help='show debug', action="store_true")
    args = optionParser.parse_args()

    if args.debug:
        print("debug is on")

    if (args.cslProgram != sys.stdin):
        sourceCode = args.cslProgram.read()
        my_cslInterpreter = cslInterpreter(debug=args.debug)
        my_cslInterpreter.programName = args.cslProgram.name
        my_cslInterpreter.run(sourceCode)
    else:
        my_cslInterpreter = cslInterpreter(single=True,debug=args.debug)
        my_cslInterpreter.programName = None
        while 1:
            try:
                sourceCode = input('csl > ')
            except EOFError:
                break
            if not sourceCode:
                continue
            try:
                my_cslInterpreter.run(sourceCode)
            except Exception as error:
                print(error)
