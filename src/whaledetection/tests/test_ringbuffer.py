import numpy as np
import soundfile as sf
import pywt



#########ringbuffer


class RingBufferMono:
    def __init__(self,capacity:int,dtype=np.float32):
        #capacity is buffer size, dtype is in audio analysis often max float32
        self.capacity = int(capacity)
        #the buffer itself
        self.buf = np.zeros((self.capacity),dtype=dtype)
        #writing index
        self.w=0
        #how many samples from capacity are filled
        self.filled=0

    def write(self,x:np.ndarray):
        #reshape multi channel signal to 1D
        x = np.asarray(x, dtype=self.buf.dtype).reshape(-1)
        n = x.size
        if n==0:
            return
        
        #safe fuction if block bigger than capacity then keep the last part
        if n>=self.cpacity:
            x=x[-self.capacity:]
            n=x.size
            print("Block size: "+n+" bigger than capacity: "+self.capacity)

        end = self.w + n
        if end <= self.capacity:
            self.buf[self.w:end] = x
        else:
            first = self.capacity - self.w
            self.buf[self.w:] = x[:first]
            self.buf[:end % self.capacity] = x[first:]

        self.w = end % self.capacity
        self.filled = min(self.capacity, self.filled + n)