# Background removing

#importing the modules
from PIL import Image
from rembg import remove

# loading the image
inp_image = r'C:\Users\panka\Downloads\image-cropped-8x10.jpg'
out_img = 'output.jpg'

# opening the image
input = Image.open(inp_image)

#output after removing the background
output = remove(input)

print(output)