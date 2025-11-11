''' Module Errors '''

class CalculatorError(Exception):
    ''' Base class for calculator errors '''
    def __init__(self, *args, errorMsg='Calculator Error!'):
        super().__init__(errorMsg, *args)
        # self.message = errorMsg

class VariableError(CalculatorError):
    ''' Variable error '''
    def __init__(self, *args, errorMsg='Variable error!'):
        super().__init__(errorMsg, *args)

class EvaluationError(CalculatorError):
    ''' Evaluation error '''
    def __init__(self, *args, errorMsg='Evaluation error!'):
        super().__init__(errorMsg, *args)

class NumberError(CalculatorError):
    ''' Number error '''
    def __init__(self, *args, errorMsg='Number error!'):
        super().__init__(errorMsg, *args)

class ParseError(CalculatorError):
    ''' Parse error '''
    def __init__(self, *args, errorMsg='Parse error!'):
        super().__init__(errorMsg, *args)

class SettingsError(CalculatorError):
    ''' Settings error '''
    def __init__(self, *args, errorMsg='Settings error!'):
        super().__init__(errorMsg, *args)
