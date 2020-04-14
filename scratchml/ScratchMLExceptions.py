class InvalidValueException(ValueError):
    """InvalidValueException is raised when a value passed as parameter is invalid.
    """
    def __init__(self, message, *args):
        self.expected = message['expected']
        self.recieved = message['recieved']

        self.message = f'Expected value in {self.expected} but recieved {self.recieved}'

        super(InvalidValueException, self).__init__(self.message, *args)
