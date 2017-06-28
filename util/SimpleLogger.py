import os
import datetime
import csv

class SimpleLogger:
    def __init__(self, fields, print_format='', logger_name=None):
        if isinstance(print_format, str) and not print_format:
            printstr = ''
            for field in fields:
                printstr = printstr + field + ': %f '
            print_format = printstr
        self.print_format = print_format
        self.fields = fields
        self.log = dict()
        for field in fields:
            self.log[field] = []
        self.log_folder = 'logs'
        self.logger_name = logger_name
        if self.logger_name is None:
            self._gen_logger_name()

    def _gen_logger_name(self):
        self.logger_name = datetime.datetime.today().strftime('%y%m%d_%H%M%S_log')
            
    def add(self, input):
        assert(len(input) == len(self.fields))
        
        for i in range(0, len(self.fields)):
            self.log[self.fields[i]].append(input[i])
            
        if isinstance(self.print_format, str):
            print(self.print_format % input)

    def save_csv(self, fname=None):
        "Save log information to csv"
        if fname is None:
            fname = self.logger_name + '.csv'
        path = os.path.join(self.log_folder, fname)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            if len(self.fields) == 0:
                return
            n_entries = len(self.log[self.fields[0]])
            writer.writerow(self.fields)  # write column labels
            for i in range(n_entries):
                row = []
                for field in self.fields:
                    row.append(self.log[field][i])
                writer.writerow(row)
            print('saved:', path)

