from settings import Settings
import vars as module_vars
from memory import GlobalMemory
import errors as module_errors
from parser import parse
from number import *
from pathlib import Path
import help as help_module
import sys, re
from UI import *

class Calculator:
    def __init__(self):
        self.basedir = Path(__file__).resolve().parent
        self.memPath = self.basedir/'mem.txt'
        self.settingsPath = self.basedir/'settings.txt'
        self.historyPath = self.basedir/'history.txt'
        self.st = Settings(self.settingsPath)
        self.mainMem = GlobalMemory(self.memPath)
        self.mainMem.trie = self.trie = Trie.fromCollection(self.mainMem.fullDict())
        self.ui = UI(self.mainMem, self.historyPath)
        self.main_loop = True
        sys.setrecursionlimit(500000)

        self.modules = {
            re.compile(r'^\s*help\s*$'): self.module_help,
            re.compile(r'^\s*vars\s*$'): self.module_vars,
            re.compile(r'^\s*del\s(.*)$'): self.module_delete,
            re.compile(r'^\s*(?:quit|exit)\s*$'): self.module_quit,
            re.compile(r'^\s*frac(?:\s+(\d+))?$'): self.module_frac,
            re.compile(r'^\s*prec(?:ision)?(?:\s+(\d+))?$'): self.module_prec,
            re.compile(r'^\s*disp(?:lay)?(?:\s+(\d+))?$'): self.module_display,
            re.compile(r'^\s*debug(?:\s+(\w+))?$'): self.module_debug,
            re.compile(r'^\s*(?:kb|keyboard)(?:\s+(\w+))?$'): self.module_keyboard,
            re.compile(r'^\s*(?:quick_exp(?:onents)?)(?:\s+(\w+))?$'): self.module_quick_exp,
            re.compile(r'^\s*(?:=|sto(?:re)? |->)\s*([A-Za-z]\w*)\s*$'): self.module_display
        }
    
    def HandleErrors(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)

            except module_errors.CalculatorError as e:
                if len(e.args) > 1: 
                    self.ui.addText("display", (' ' * (len(self.ui.prompt) + (span := e.args[1])[0] - 1) + '↗' + '‾' * (span[1] - span[0]), UI.BRIGHT_RED_ON_BLACK))
                self.ui.addText("display", (f"{repr(e).split('(')[0]}: {e.args[0]}", UI.BRIGHT_RED_ON_BLACK))

            except ZeroDivisionError as e:
                self.ui.addText("display", (repr(e), UI.BRIGHT_RED_ON_BLACK))

            except RecursionError:
                self.ui.addText("display", ("RecursionError: Check for infinite recursion in functions.", UI.BRIGHT_RED_ON_BLACK))

            except (EOFError, KeyboardInterrupt):
                return None

        return wrapper

    def module_help(self):
        help_module.display()
    
    def module_vars(self):
        if len(self.ui.text["display"]) > 0: 
            self.ui.addText("display")
            
        self.ui.addText("display", ("User-defined Variables", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("──────────────────────", ))
        
        for k, v in self.mainMem.vars.items():
            if isinstance(v, Function) and k == v.name:
                self.ui.addText("display", (f"{v.value()}", UI.LIGHTBLUE_ON_BLACK))
            else:
                self.ui.addText("display", (k, UI.LIGHTBLUE_ON_BLACK), (' = ', ), (f"{v.value()}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_delete(self):
        deleted = self.mainMem.delete(m.group(1))
        if not deleted:
            self.ui.addText("display", ('Variable(s) not found!', UI.LIGHTBLUE_ON_BLACK))
        else:
            self.ui.addText("display", ("Deleted: ", ))
            for i, var in enumerate(deleted):
                if i > 0: self.ui.addText("display", (', ', ), startNewLine=False)
                self.ui.addText("display", (var, UI.LIGHTBLUE_ON_BLACK), startNewLine=False)
    
    def module_quit(self):
        self.main_loop = False
    
    def module_frac(self):
        if self.m.group(1) is not None: 
            st.set("frac_max_length", int(self.m.group(1)))
        self.ui.addText("display", ("frac_max_length", UI.LIGHTBLUE_ON_BLACK), (' -> ', ), (f"{st.get('frac_max_length')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_prec(self):
        if self.m.group(1) is not None: 
            st.set("working_precision", int(self.m.group(1)))
        self.ui.addText("display", ("working_precision", UI.LIGHTBLUE_ON_BLACK), (' -> ', ), (f"{st.get('working_precision')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_display(self):
        if self.m.group(1) is not None: 
            st.set("final_precision", int(self.m.group(1)))
        self.ui.addText("display", ("final_precision", UI.LIGHTBLUE_ON_BLACK), (' -> ', ), (f"{st.get('final_precision')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_debug(self):
        flag = {'on':True, 'off':False}.get(self.m.group(1) if self.m.group(1) is None else self.m.group(1).lower(), None)
        if flag is not None: 
            st.set("debug", flag)
        else: 
            self.ui.addText("display", ("Usage: ", ), ("debug [on/off]", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("debug", UI.LIGHTBLUE_ON_BLACK), (" -> ", ), (f"{st.get('debug')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_keyboard(self):
        flag = {'on':True, 'off':False}.get(self.m.group(1) if self.m.group(1) is None else self.m.group(1).lower(), None)
        if flag is not None:
            st.set("keyboard", flag)
        else:
            self.ui.addText("display", ("Usage: ", ), ("keyboard [on/off] (allows use of keyboard module)", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("keyboard", UI.LIGHTBLUE_ON_BLACK), (" -> ", ), (f"{st.get('keyboard')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_quick_exp(self):
        flag = {'on':True, 'off':False}.get(self.m.group(1) if self.m.group(1) is None else self.m.group(1).lower(), None)
        if flag is not None: 
            st.set("quick_exponents", flag)
        else: 
            self.ui.addText("display", ("Usage: ", ), ("quick_exp[onents] [on/off]", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("quick_exponents", UI.LIGHTBLUE_ON_BLACK), (" -> ", ), (f"{st.get('quick_exponents')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_display(self):
        if (ans := self.mainMem.get('ans')) is None:
            self.ui.addText("display", ("Variable '", ), ("ans", UI.LIGHTBLUE_ON_BLACK), ("' does not exist or has been deleted", ))
        else:
            self.mainMem.add(self.m.group(1), ans)
            self.ui.trie.insert(self.m.group(1))
            self.ui.addText("display", (f'{self.m.group(1)}', UI.LIGHTBLUE_ON_BLACK), (' = ', ), (f'{self.mainMem.get(self.m.group(1)).value()}', UI.LIGHTBLUE_ON_BLACK))

    @HandleErrors
    def main(self):
        while self.main_loop:
            inp = self.ui.getInput(trie=self.trie)

            # check for commands
            command = False
            for pattern, func in self.modules.items():
                self.m = re.match(pattern, inp)
                if self.m:
                    func()
                    command = True

            if not command:
                expr = parse(inp)
                if expr is None: 
                    continue
                self.mainMem.writeLock = True
                val = expr.value(self.mainMem)
                if isinstance(val, Number): 
                    val = val.fastContinuedFraction(epsilon=st.finalEpsilon)
                self.ui.addText("display", (val.disp(st.get('frac_max_length'), st.get('final_precision')), UI.BRIGHT_GREEN_ON_BLACK))
                self.mainMem.writeLock = False
                self.mainMem.add('ans', val)
                    
            self.ui.redraw("display")
            self.ui.saveHistory()

        self.ui.end()

if __name__ == '__main__':
    calc = Calculator()
    calc.main()
