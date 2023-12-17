from traceback import print_exc
from os.path import basename, splitext
from PIL import Image
from constants import COMPUTED_DIRECTORY_NAME
from utils import convertToListImage

def labTask(image_path: str) -> None:
    pass
    # computed_directory = f'./{COMPUTED_DIRECTORY_NAME}/'

    # file_name_with_extension = basename(image_path)
    # file_name, file_extension = splitext(file_name_with_extension)

    # with Image.open(image_path) as im:
    #     image = convertToListImage(im)
                
                

if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif') 

         labTask('./assets/lena_gray_256.tif')         
    except Exception as e:
        print('Error occured:')
        print_exc()