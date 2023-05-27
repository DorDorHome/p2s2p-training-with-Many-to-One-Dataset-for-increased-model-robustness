# this added file was created by Alvin Chan, to downsize photos in order to test 


# Importing Image class from PIL module
from hashlib import new
from PIL import Image
import math



# Opens a image in RGB mode
input_image_file_name = '00059'
input_image_file = input_image_file_name + ".png"
im = Image.open('test_data/'+ input_image_file)
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
print('original size is {} x {}:'.format(width, height) )


# # Setting the points for cropped image
# left = 4
# top = height / 16
# right = 154
# bottom = 3 * height / 5
 
# Cropped image of above dimension
# (It will not change original image)
# im1 = im.crop((left, top, right, bottom))

new_width = math.floor(width/32)
new_height = math.floor(height/32)

new_size = (new_width, new_height)
im1 = im.resize(new_size)
# Shows the image in image viewer
# im1.show()

# im1.save('data_folder/super_res_input_data/{}_size{}x{}.jpg'.format(input_image_file_name,    new_width, new_height))



def downsize_image_tensor(im, resize_factor ):
    """
    input: a tensor of shape (3, h, w)

    output: a downsized tensor of shape ()
    
    
    """

    width, height = im.size
    print('original size is {} x {}:'.format(width, height) )
    
    new_width = math.floor(width/resize_factor)
    new_height = math.floor(height/resize_factor)
    new_size = (new_width, new_height)
    im_resized = im.resize(new_size)
    return im_resized

