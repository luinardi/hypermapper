import sys


class Logger:
    """
    This class allows to write on the log file and to the stdout at the same time.
    In Hypermapper the log is always created. The stdout can be switch off in interactive mode for example.
    It works by overloading sys.stdout.
    """

    def __init__(self, log_file: str = "hypermapper_logfile.log"):
        self.filename = log_file
        self.terminal = sys.stdout
        try:
            self.log = open(self.filename, "a")
        except Exception as e:
            print("Unexpected error opening the log file: ", self.filename)
            raise e
        self.log_only_on_file = False

    def write(self, message):
        if not self.log_only_on_file:
            self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def write_protocol(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush_protocol()

    def flush(self):
        if not self.log_only_on_file:
            self.terminal.flush()
        self.log.flush()

    def flush_protocol(self):
        self.terminal.flush()
        self.log.flush()

    def switch_log_only_on_file(self, choice):
        self.log_only_on_file = choice

    def change_log_file(self, filename):
        if self.filename != filename:
            self.close_log_file()
            self.filename = filename
            try:
                self.log = open(self.filename, "a")
            except Exception as e:
                print("Unexpected error opening the log file: ", self.filename)
                raise e

    def write_to_logfile(self, message):
        self.log.write(message)

    def close_log_file(self):
        self.log.close()

    def __del__(self):
        try:
            self.log.close()
        except Exception as e:
            print("Warning: exception raised closing the log file: ", self.filename)
            raise e
