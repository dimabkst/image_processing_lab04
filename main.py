from traceback import print_exc
from os.path import basename, splitext
from PIL import Image
from constants import COMPUTED_DIRECTORY_NAME
from services import blurrImage, get2DDiscreteFourierTransform, get2DInverseDiscreteFourierTransform, getGaussianPSF, wienerFiltration
from utils import convertToListImage, convertToPillowImage, saveImage

def labTask(image_path: str) -> None:
    pass
    computed_directory = f'./{COMPUTED_DIRECTORY_NAME}/'

    file_name_with_extension = basename(image_path)
    file_name, file_extension = splitext(file_name_with_extension)

    with Image.open(image_path) as im:
        image = convertToListImage(im)

        fourier_transform = get2DDiscreteFourierTransform(image)

        inversed = get2DInverseDiscreteFourierTransform(fourier_transform)

        saveImage(convertToPillowImage(inversed), f'{computed_directory}{file_name}_inversed_original_{file_extension}')

        gaussian_psf = getGaussianPSF((3, 3))

        # gaussian_psf = [[0.07511360795411151, 0.12384140315297397, 0.07511360795411151], [0.12384140315297397, 0.2041799555716581, 0.12384140315297397], [0.07511360795411151, 0.12384140315297397, 0.07511360795411151]]

        # matlab_gaussian_psf = [[0.0751, 0.1238, 0.0751], [0.1238, 0.2042, 0.1238], [0.0751, 0.1238, 0.0751]]

        blurred_image = blurrImage(image, gaussian_psf)

        # blurred_image = convertToListImage(Image.open('./computed/blurred_cameraman.tif'))

        saveImage(convertToPillowImage(blurred_image), f'{computed_directory}{file_name}_blurred_{file_extension}')

        wiener_alphas = [0.0004, 0.0008, 0.001]

        wiener_results = [wienerFiltration(blurred_image, gaussian_psf, alpha) for alpha in wiener_alphas]

        for i in range(len(wiener_results)):
            saveImage(convertToPillowImage(wiener_results[i]), f'{computed_directory}{file_name}_wiener{i + 1}_{file_extension}')


if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif') 

        #  labTask('./assets/lena_gray_256.tif')         
    except Exception as e:
        print('Error occured:')
        print_exc()