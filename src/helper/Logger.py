import sys

'''add these 2 lines to your file for the logger to work'''
'''
sys.stdout = Logging(folder_fbase + "stdout.log", sys.stdout) # create & save file w/ stdout
sys.stderr = Logging(folder_fbase + "stderr.log", sys.stderr) # create & save file w/ error log for debugging
'''
class Logging(object):
    def __init__(self, fileName, stream):
        self.stream = stream
        self.file = open(fileName, 'w')

        
        # self.encoding = stream.encoding  # Chat GPT told me to add this line so that tensorflow==2.18 would be happy

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.stream.flush()
        self.file.flush()
        # pass

    # def close(self):
    #     self.file.close()