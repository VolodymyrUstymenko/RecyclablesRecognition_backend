import io
from scipy import ndimage
from skimage import io as skio, filters, img_as_ubyte, color

def ObjectDetection(image_stream):
    min_area = 1000
    image = skio.imread(image_stream)
    if image is None: return []
    gray = color.rgb2gray(image)
    gray_blurred = ndimage.gaussian_filter(gray, sigma = 1)
    threshold = filters.threshold_otsu(gray_blurred)
    binary = gray_blurred > threshold
    labeled_array, _ = ndimage.label(binary)
    object_slices = ndimage.find_objects(labeled_array)
    filtered_slices = [
        obj_slice for obj_slice in object_slices
        if obj_slice and (obj_slice[1].stop - obj_slice[1].start) * (obj_slice[0].stop - obj_slice[0].start) > min_area
    ]
    fragments = []
    for i, obj_slice in enumerate(filtered_slices):
        obj = image[obj_slice]
        buffer = io.BytesIO()
        skio.imsave(buffer, img_as_ubyte(obj), format="png")
        buffer.seek(0)
        fragments.append((f"fragment_{i}.png", buffer))
    return fragments[1:]