import base64

import requests

url = 'http://localhost/hieroscan'
from skimage import io

# input1 = io.imread('assets/examples/test0.jpg')
image = open("examples/test0.jpg", "rb")
encoded_image = base64.encodebytes(image.read()).decode('utf-8')
myobj = {"image": encoded_image}
headers = {'content-type': 'application/json'}
x = requests.post(url, json=myobj)


print(x)
x = x.json()
image = x['image']
with open('response.jpg', 'wb') as fh:
    fh.write(base64.decodebytes(bytes(image, 'utf-8')))
# x = requests.get(url)
# io.imsave('output0.jpg', x['image'])
print(x['transliteration'])
print(x['translation'])

# print(encoded_image)