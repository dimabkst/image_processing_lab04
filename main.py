from traceback import print_exc
from os.path import basename, splitext
from PIL import Image
from constants import COMPUTED_DIRECTORY_NAME
from services import addGaussianAdditiveNoise, blurrImage, get2DDiscreteFourierTransform, get2DInverseDiscreteFourierTransform, getGaussianPSF, getStandardDeviation, wienerFiltration
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

        gaussian_psf = getGaussianPSF((len(image), len(image[0])), sigma=5)

        blurred_image = blurrImage(image, gaussian_psf)

        saveImage(convertToPillowImage(blurred_image), f'{computed_directory}{file_name}_blurred_{file_extension}')

        noisy_blurred_image = addGaussianAdditiveNoise(blurred_image, 0.1)

        saveImage(convertToPillowImage(noisy_blurred_image), f'{computed_directory}{file_name}_noisy_blurred_{file_extension}')

        wiener_alphas = [0.00001, 0.0001, 0.001, 0.01]

        wiener_results = [wienerFiltration(noisy_blurred_image, gaussian_psf, alpha) for alpha in wiener_alphas]

        for i in range(len(wiener_results)):
            saveImage(convertToPillowImage(wiener_results[i]), f'{computed_directory}{file_name}_wiener{i + 1}_{file_extension}')


if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif') 

         labTask('./assets/lena_gray_256.tif')         
    except Exception as e:
        print('Error occured:')
        print_exc()