from PIL import Image,ImageOps 
import numpy as np 
np.random.seed(0)

class GammaCorrection(object):
    def __init__(self, gamma_value=0.1):
        self.gamma = gamma_value
        
    def __call__(self, sample):
        sample=sample.convert("LA") 
        row = sample.size[0]
        col = sample.size[1]
        
        result_img = Image.new("L", (row, col))
        
        for x in range(1 , row):
            for y in range(1, col):
                value = pow(sample.getpixel((x,y))[0]/255,(1/self.gamma))*255
                if value >= 255 :
                    value = 255
                result_img.putpixel((x,y), int(value))
                    
        return result_img
    
class HistogramEqualization(object):
    def __call__(self, sample):
        result_img = ImageOps.equalize(sample, mask = None) 
        return result_img