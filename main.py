from settings import Settings
from vars import *
from memory import GlobalMemory
from errors import *
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
        sys.setrecursionlimit(500000)
    
    def HandleErrors(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)

            except CalculatorError as e:
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
    
    def module_frac(self):
        if m.group(1) is not None: 
            st.set("frac_max_length", int(m.group(1)))
        self.ui.addText("display", ("frac_max_length", UI.LIGHTBLUE_ON_BLACK), (' -> ', ), (f"{st.get('frac_max_length')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_prec(self):
        if m.group(1) is not None: 
            st.set("working_precision", int(m.group(1)))
        self.ui.addText("display", ("working_precision", UI.LIGHTBLUE_ON_BLACK), (' -> ', ), (f"{st.get('working_precision')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_display(self):
        if m.group(1) is not None: 
            st.set("final_precision", int(m.group(1)))
        self.ui.addText("display", ("final_precision", UI.LIGHTBLUE_ON_BLACK), (' -> ', ), (f"{st.get('final_precision')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_debug(self):
        flag = {'on':True, 'off':False}.get(m.group(1) if m.group(1) is None else m.group(1).lower(), None)
        if flag is not None: 
            st.set("debug", flag)
        else: 
            self.ui.addText("display", ("Usage: ", ), ("debug [on/off]", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("debug", UI.LIGHTBLUE_ON_BLACK), (" -> ", ), (f"{st.get('debug')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_keyboard(self):
        flag = {'on':True, 'off':False}.get(m.group(1) if m.group(1) is None else m.group(1).lower(), None)
        if flag is not None:
            st.set("keyboard", flag)
        else:
            self.ui.addText("display", ("Usage: ", ), ("keyboard [on/off] (allows use of keyboard module)", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("keyboard", UI.LIGHTBLUE_ON_BLACK), (" -> ", ), (f"{st.get('keyboard')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_quick_exp(self):
        flag = {'on':True, 'off':False}.get(m.group(1) if m.group(1) is None else m.group(1).lower(), None)
        if flag is not None: 
            st.set("quick_exponents", flag)
        else: 
            self.ui.addText("display", ("Usage: ", ), ("quick_exp[onents] [on/off]", UI.LIGHTBLUE_ON_BLACK))
        self.ui.addText("display", ("quick_exponents", UI.LIGHTBLUE_ON_BLACK), (" -> ", ), (f"{st.get('quick_exponents')}", UI.LIGHTBLUE_ON_BLACK))
    
    def module_sto(self):
        if (ans := self.mainMem.get('ans')) is None:
            self.ui.addText("display", ("Variable '", ), ("ans", UI.LIGHTBLUE_ON_BLACK), ("' does not exist or has been deleted", ))
        else:
            mainMem.add(m.group(1), ans)
            self.ui.trie.insert(m.group(1))
            self.ui.addText("display", (f'{m.group(1)}', UI.LIGHTBLUE_ON_BLACK), (' = ', ), (f'{self.mainMem.get(m.group(1)).value()}', UI.LIGHTBLUE_ON_BLACK))

    @HandleErrors
    def main(self):
        while True:
            inp = self.ui.getInput(trie=self.trie)

            # check for commands
            if m := re.match(r'^\s*help\s*$', inp):
                self.module_help()

            elif m := re.match(r'^\s*vars\s*$', inp):
                self.module_vars()

            elif m := re.match(r'^\s*del\s(.*)$', inp):
                self.module_delete()

            elif m := re.match(r'^\s*(?:quit|exit)\s*$', inp):
                break

            elif m := re.match(r'^\s*frac(?:\s+(\d+))?$', inp):
                self.module_frac()

            elif m := re.match(r'^\s*prec(?:ision)?(?:\s+(\d+))?$', inp):
                self.module_prec()

            elif m := re.match(r'^\s*disp(?:lay)?(?:\s+(\d+))?$', inp):
                self.module_display()

            elif m := re.match(r'^\s*debug(?:\s+(\w+))?$', inp):
                self.module_debug()

            elif m := re.match(r'^\s*(?:kb|keyboard)(?:\s+(\w+))?$', inp):
                self.module_keyboard()

            elif m := re.match(r'^\s*(?:quick_exp(?:onents)?)(?:\s+(\w+))?$', inp):
                self.module_quick_exp()

            elif inp.strip() == '':
                continue

            elif m := re.match(r'^\s*(?:=|sto(?:re)? |->)\s*([A-Za-z]\w*)\s*$', inp):
                self.module_sto()

            else:
                expr = parse(inp)
                if expr is None: continue
                self.mainMem.writeLock = True
                val = expr.value(self.mainMem)
                if isinstance(val, Number): val = val.fastContinuedFraction(epsilon=st.finalEpsilon)
                self.ui.addText("display", (val.disp(st.get('frac_max_length'), st.get('final_precision')), UI.BRIGHT_GREEN_ON_BLACK))
                self.mainMem.writeLock = False
                self.mainMem.add('ans', val)
                    
            self.ui.redraw("display")
            self.ui.saveHistory()

        self.ui.end()

if __name__ == '__main__':
    calc = Calculator()
    calc.main()
