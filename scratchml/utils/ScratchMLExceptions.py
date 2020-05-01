class InvalidValueException(ValueError):
    """InvalidValueException is raised when a value passed as parameter is invalid.
    """
    def __init__(self, message):
        if type(message) == dict:
            self.expected = message['expected']
            self.recieved = message['recieved']

            self.message = f'Expected value in {self.expected} but recieved {self.recieved}'

        else:
            self.message = message

        super(InvalidValueException, self).__init__(self.message)

class ModelNotTrainedException(Exception):
    """ModelNotTrainedException is raised when a value passed as parameter is invalid.
    """
    def __init__(self, name):
        self.message = f'Model has to be trained before {name}() can be called'

        super(ModelNotTrainedException, self).__init__(self.message)
