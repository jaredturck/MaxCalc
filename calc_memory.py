''' Module memory '''

from calc_number import e, pi, imag_i, Number
from calc_functions import Function, FuncComposition
import calc_op as op
from calc_settings import Settings

st = Settings()

class Memory:
    ''' Memory class for storing variables and functions. '''

    # Base class intended for use by Functions.

    globalMem = None

    baseList = {
        'e': e,
        'pi': pi,
        'i': imag_i,
        'P': op.permutation,   # These are here so that
        'C': op.combination,   # they are overrideable.
    }

    topList = {
        'abs': op.absolute,   # These override user vars
        'arg': op.argument,
        'conj': op.conjugate,
        'Im': op.imPart,
        'Re': op.realPart,
        'sgn': op.signum,
        'sin': op.sin,
        'cosec': op.csc,
        'csc': op.csc,
        'cos': op.cos,
        'sec': op.sec,
        'tan': op.tan,
        'cot': op.cot,
        'sinh': op.sinh,
        'cosh': op.cosh,
        'tanh': op.tanh,
        'asin': op.arcsin,
        'arcsin': op.arcsin,
        'acos': op.arccos,
        'arccos': op.arccos,
        'atan': op.arctan,
        'arctan': op.arctan,
        'sqrt': op.sqrt,
        'ln': op.ln,
        'lg': op.lg,
        'normcdf': op.normcdf,
        'normpdf': op.normpdf,
        'invnorm': op.invnorm,
    }

    def __init__(self):
        self.vars = {}

    def get(self, txt):
        ''' Get variable from memory. '''
        seq = [Memory.topList, self.vars if self is not Memory.globalMem else {}, Memory.globalMem.vars, Memory.baseList]
        for dct in seq:
            if txt in dct:
                return dct[txt]

    def add(self, txt, val):
        ''' Add variable to memory. '''
        if isinstance(val, Number):
            val = val.fast_continued_fraction(epsilon=st.epsilon)
        self.vars[txt] = val

    def delete(self, key):
        ''' Delete variable from memory. '''
        raise NotImplementedError # Functions should never have stuff in their memory deleted

    def copy(self):
        ''' Return a copy of this Memory object. '''
        cpy = Memory()
        cpy.__dict__.update(self.__dict__)
        cpy.vars = self.vars.copy()
        return cpy

    def __iter__(self):
        ''' Iterate over variables in memory. '''
        yield from self.vars

    def __str__(self):
        ''' Return a string representation of the memory. '''
        return "{{" + '; '.join([f"{v}" if isinstance(v, Function) and k == v.name else f"{k} = {v}" for k, v in self.vars.items()]) + "}}"

    def str_without(self, key_without):
        ''' Return a string representation of the memory without a specific key. '''
        return "{{" + '; '.join([f"{v}" if isinstance(v, Function) and k == v.name else f"{k} = {v}" for k, v in self.vars.items() if k != key_without]) + "}}"

    def full_dict(self):
        ''' Return a full dictionary of the memory, including base and global variables. '''
        return Memory.baseList | Memory.globalMem.vars | self.vars | Memory.topList

class GlobalMemory(Memory):
    ''' Global memory class for storing variables and functions. '''

    def __init__(self, filepath=None):
        if filepath is None:
            raise MemoryError("Must specify a memory file.")
        self.vars = {}
        self.trie = None
        self.filepath = filepath
        self.write_lock = True
        Memory.globalMem = self
        if filepath.exists():
            self.load()

    def add(self, string, val, save=True):
        ''' Add variable to memory. '''
        if string == 'ans':
            # if 'ans' in self.vars: self.vars.pop('ans')
            self.vars['ans'] = val
        else:
            if isinstance(val, Number):
                val = val.fast_continued_fraction(epsilon=st.epsilon)
            need_sort = string not in self.vars
            self.vars[string] = val
            if self.trie is not None:
                self.trie.insert(string)
            if need_sort:
                self.vars = {k: self.vars[k] for k in sorted(self.vars, key=lambda x: -isinstance(self.vars[x], Function))}
        if not self.write_lock and save:
            self.save()

    def delete(self, string):
        ''' Delete variable from memory. '''
        str_list = string.replace(',', ' ').split()
        deleted = []
        for s in str_list:
            if s in self.vars:
                del self.vars[s]
                deleted.append(s)
                if s not in self.baseList:
                    self.trie.delete(s)
        if not self.write_lock and deleted:
            self.save()
        return deleted

    def copy(self):
        return Memory()  # global returns a blank Memory object when copy is called

    def save(self):
        ''' Save memory to file. '''
        with open(self.filepath, 'w', encoding='utf-8') as f:
            for var, value in self.vars.items():
                if isinstance(value, FuncComposition):
                    f.write(f"{var} = {value.name}\n")
                elif isinstance(value, Function):
                    if var != value.name:
                        f.write(f"{var} = ")
                    f.write(f"{str(value)}\n")
                else:
                    f.write(f"{var} = {value.fromString if hasattr(value, 'fromString') else str(value)}\n")

    def load(self):
        ''' Load memory from file. '''
        import calc_parser
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                calc_parser.parse(line).value(mem=self)
        self.write_lock = False
        self.save()
