import os
from PIL import Image

min_size = 100
t_size = min_size, min_size

RAW_SICK = './data/raw/sick/'
RAW_HEALTHY = './data/raw/healthy/'

PREPROCESSED = './data/preprocessed/simpleNN'

def preprocess(directory, files, flag, quantity):
    for i in range(quantity):
        outfile = PREPROCESSED +'/' + flag + '/' + str(i) + '.jpg'
        im = Image.open(directory + files[i])
        im.thumbnail(t_size, Image.ANTIALIAS)
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), (0,0,0,0))
        new_im.paste(im, (int((size -x)/2), int((size-y)/2)))
        new_im.save(outfile)

f_sick = []
for (dirpath, dirnames, filenames) in os.walk(RAW_SICK):
    f_sick.extend(filenames)

f_healty = []
for (dirpath, dirnames, filenames) in os.walk(RAW_HEALTHY):
    f_healty.extend(filenames)

preprocess(RAW_SICK, f_sick, 'sick', len(f_sick))
preprocess(RAW_HEALTHY, f_healty, 'healthy', len(f_sick) + 10)