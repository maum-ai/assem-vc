import subprocess


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')


class WavBinaryWrapper(object):
    def __init__(self, data):
        super(WavBinaryWrapper, self).__init__()
        self.data = data
        self.position = 0

    def read(self, n=-1):
        if n == -1:
            data = self.data[self.position:]
            self.position = len(self.data)
            return data
        else:
            data = self.data[self.position:self.position+n]
            self.position += n
            return data

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = len(self.data) + offset
        else:
            raise RuntimeError('test')
        return self.position

    def tell(self):
        return self.position

    def close(self):
        pass
