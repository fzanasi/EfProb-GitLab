import ast
import re
import cslAst
import ply.yacc as yacc
import ply.lex as lex
import efprob_dc as efp # only for python escape

reserved = { # reserved keywords
    'if': 'IF',
    'then': 'THEN',
    'else': 'ELSE',
    'discard': 'DISCARD',
    'observe': 'OBSERVE',
    'True' : "TRUE",
    'False' : "FALSE",
    'flip' : 'FLIP', # flip function
    'uniform' : 'UNIFORM',
    'binomial' : 'BINOMIAL'
}

tokens = [ # any input get split into tokens
   'PYTHON_ESCAPE',
   'COMMA', 'ARROW',
   'NAME', 'STRING', 'PROBABILITY_VALUE','BIT', # bit is 0 or 1
   'NUMBER','INTEGER',
   'L_PAREN', 'R_PAREN', # brackets ( )
   'L_S_PAREN','R_S_PAREN', # square brackets []
   'L_B_PAREN','R_B_PAREN', # brace brackets {}
   'EQUALS','GE','LE',
   'SEMICOL', 'COLON',
   'PLUS', 'PIPE', 'KET',  # ket = >
   'CROSS','AND','NEGATE' # cross = @, negate = ~, marginal = %
] +  list(reserved.values())

# Tokens
t_EQUALS = r'\='
t_CROSS = r'\@'
t_AND = r'\&'
t_NEGATE = r'\~'
t_SEMICOL = r'\;'
t_L_B_PAREN = r'\{'
t_R_B_PAREN = r'\}'
t_COMMA = r'\,'
t_KET = r'\>'
t_PIPE = r'\|'
t_PLUS = r'\+'
t_COLON = r'\:'
t_L_S_PAREN = r'\['
t_R_S_PAREN = r'\]'
t_L_PAREN = r'\('
t_R_PAREN = r'\)'

def t_PYTHON_ESCAPE(t):
    r'\$\$\$\n'
    start = stop = t.lexer.lexpos
    while (t.lexer.lexdata[stop:stop+3] != '$$$'):
        stop += 1
    pythonCode = t.lexer.lexdata[start:stop]
    skip = (stop+3) - start
    t.lexer.lexpos += skip # senza newLine
    t.value = pythonCode
    return t

def t_ARROW(t):
    r'\<\-'
    t.type = 'ARROW'
    return t

def t_GE(t):
    r'\>\='
    t.type = 'GE'
    return t

def t_LE(t):
    r'\<\='
    t.type = 'GE'
    return t

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'NAME')    # Check for reserved words
    return t

def t_STRING(t):
    r'\"[^\"]+\"' # ^\" means any character but "
    t.type = 'STRING'    # Check for reserved words
    return t

def t_comment(t):
    r"[ ]*\043[^\n]*"  # \043 is '#'
    pass # ignore comments, just advance in the lexical analysis (nb: with return you would stop the lexical analysis)

def t_NUMBER(t):
    r'((\d+)?\.(\d+))|\d+' # I can write .5 for 0.5
    if re.match(r'^\d+$',t.value):
        t.type = 'INTEGER'
        t.value = int(t.value)
        if t.value in [0,1]:
            t.type = 'BIT'
    else:
        t.value = float(t.value)
        if t.value <= 1.0:
            t.type = 'PROBABILITY_VALUE'
    # note that token gets typed only if the value is (a) it is integer (in which case can be INTEGER or BIT) or (b) it is <=1 (in which case it is PROBABILITY_VALUE)
    return t

t_ignore = u" \t" # ignore tabs.

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

# Parsing rules (definining the grammar as a BNF)

precedence = (
    ('left', 'CROSS'),
    ('left', 'ARROW'),
    ('left', 'PIPE'),
    ('left', 'EQUALS'),
    ('left', 'AND'),
    ('left', 'PLUS'),
    ('right', 'UNEGATE')
)

def p_program(p):
    '''program : stm '''
    p[0] = []

def p_program_morestm(p):
    '''program : stm flow program'''
    p[0] = []

def p_stm(p):
    '''stm   : channel
            | PYTHON_ESCAPE '''


def p_channel_if(p):
    '''channel : IF pred THEN channel else'''

def p_else(p):
    '''else : ELSE channel
            | empty'''

def p_channel_discard(p):
    '''channel : DISCARD pred'''

def p_channel_observe(p):
    '''channel : OBSERVE pred'''

def p_channel_sample(p):
    '''channel : sample'''

def p_sample(p):
    '''sample : NAME ARROW state_or_channel'''

def p_state_or_channel_state(p):
    '''state_or_channel : state'''

def p_state_or_channel_channel(p):
    '''state_or_channel : channel'''

def p_state_flip(p):
   'state : FLIP L_PAREN PROBABILITY_VALUE R_PAREN'

def p_state_uniform(p):
   'state : UNIFORM L_PAREN PROBABILITY_VALUE R_PAREN'

def p_state_binomial(p):
   'state : BINOMIAL L_PAREN domain COMMA PROBABILITY_VALUE R_PAREN'

def p_state_par(p):
   'state : L_S_PAREN stateBody R_S_PAREN'

def p_stateBody(p):
   'stateBody : stateElem PLUS stateBody'

def p_stateBody_singleElem(p):
   'stateBody : stateElem'

def p_state_elem(p):
   'stateElem : PROBABILITY_VALUE PIPE nameOrBool optionKET'

def p_domain(p):
    'domain : L_S_PAREN elementList R_S_PAREN'

def p_elementList(p):
    'elementList : nameOrBool COMMA elementList'

def p_elementList_singleElem(p):
   'elementList : nameOrBool'

def p_nameOrBool(p):
    '''nameOrBool : name
                    | string
                    | boolean'''

def p_name(p):
    'name : NAME'

def p_string(p):
    'string : STRING'

def p_boolean(p):
    '''boolean : TRUE
                | FALSE'''

def p_optionKET(p):
   '''optionKET : KET
                | empty'''

def p_pred(p):
    '''pred : NAME
            | NAME EQUALS var_anyNumber
            | NAME GE anyNumber
            | NAME LE anyNumber
            | bracketDef
            | pred binOp pred
            | NEGATE pred %prec UNEGATE
            '''
def p_var_anyNumber(p):
    '''var_anyNumber : NAME
            | anyNumber
            '''
def p_anyNumber(p):
    '''anyNumber : PROBABILITY_VALUE
            | INTEGER
            | BIT
            | NUMBER
            '''
def p_binOp(p):
    '''binOp : CROSS
            | AND
            | NEGATE
            '''
def p_bracketDef(p):
   'bracketDef : L_B_PAREN predicateBody R_B_PAREN'

def p_predicateBody(p):
   'predicateBody : predicateElem COMMA predicateBody'

def p_predicateBody_singleElem(p):
   'predicateBody : predicateElem'

def p_predicate_elem(p):
   'predicateElem : nameOrBool COLON PROBABILITY_VALUE'

def p_flow(p):
    '''flow : SEMICOL'''

def p_empty(p):
    '''empty : '''

def p_error(p):
    if p:
      print("Syntax error at '%s'" % p.value)
    else:
      print("Syntax error at EOF")
    raise TypeError("unknown/unexpected text at %r" % (p.value,))


# ====== Syntax Parsing ==================
class cslParser(object):
    def __init__(self, single=False):
        self.single = single
        self.lexer = lex.lex()
        self.parser = yacc.yacc(start="program")
    def parse(self, code):
        self.lexer.input(code) # let code be the input of the lexer
        parsedListStatements = self.parser.parse(lexer=self.lexer) # this list is p[0], constructed with the production rules
        if self.single:
            syntaxTree = ast.Interactive(body=parsedListStatements)
        else:
            syntaxTree = ast.Module(body=parsedListStatements)
        ast.fix_missing_locations(syntaxTree) # Adds line numbers to the generated program. If the program gets stuck, this tells where the error was.
        return syntaxTree
