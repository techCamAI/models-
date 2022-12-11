import tensorflow_datasets as tfds, tensorflow as tf
import pybboxes as pbx, numpy as np

imageSize = 416
labelSize = int(imageSize/8)

def __getGaussian(radius):
    X = np.linspace(-radius, +radius, radius*2)
    Y = np.linspace(-radius, +radius, radius*2)
    X1, Y1 = np.meshgrid(X, Y)

    d = np.sqrt(X1**2+Y1**2)
    sigma, mu = radius/2, 0.0
    gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    gauss = ( gauss - np.min(gauss) ) / ( np.max(gauss) - np.min(gauss) ) # scalling between 0 to 1

    return gauss

def centroids2Images(point_list, im_num_row, im_num_col, g_radius=5):
    circle_mat = __getGaussian(g_radius)
    temp_im = np.zeros((im_num_row+g_radius*2, im_num_col+g_radius*2))

    for one_pnt in point_list:
        pnt_row = int(one_pnt[0])
        pnt_col = int(one_pnt[1])

        current_patch = temp_im[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius]
        temp_im[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius] = np.maximum(current_patch, circle_mat)

    temp_im = temp_im[g_radius:-g_radius, g_radius:-g_radius]
    return temp_im

class tfDataset(tfds.core.GeneratorBasedBuilder):
    """dataset builder for the pre-augmented mask dataset"""
    VERSION = tfds.core.Version('1.1.0')
    RELEASE_NOTES = {
        '1.0.0': 'Inital Release',
        'About': """Augmentation in roboflow on the Medical Mask Dataset BY
                    Humans in the loop, https://humansintheloop.org/resources/datasets/medical-mask-dataset/"""}

    def localizationMap(self, imagePath, pointsDict=None, baseSize=labelSize, cR=5):
        txtPath = 'centroid-mask/labels/' + str(imagePath.name)[:-4] + '.txt'
        _, pointsDict = self.PixelPoints(txtPath)

        zero = pointsDict['0']
        ones = pointsDict['1']
        img1, img2 = centroids2Images(zero, baseSize, baseSize, cR), centroids2Images(ones, baseSize, baseSize, cR)
        label = np.zeros((baseSize, baseSize, 2))
        label[:, :, 0] += img1
        label[:, :, 1] += img2

        return tf.Variable(label)

    def PixelPoints(self, filename:str) -> tuple():
        centers, points = [], {'0': [], '1': []} # base list containing all centers for multiple bounding box images

        with open(filename, 'r') as file:
            for line in file.readlines():
                YOLOF = [float(value) for value in line.split()]
                X, Y, W, H = pbx.convert_bbox(YOLOF[1:], from_type="YOLO".lower(), to_type="voc", image_size=(416, 416))
                centerX = X + int((W - X)/2)
                centerY = Y + int((H - Y)/2)

                centers.append((centerX, centerY, YOLOF[0])) # centroid of bounding box along with class group
                points[str(int(YOLOF[0]))].append([int((centerX/imageSize)*labelSize), int((centerY/imageSize)*labelSize)]) # required for localization map
            return centers, points

    def labelImage(self, imagePath:str, squareSize=imageSize, nSize=labelSize) -> 'heatmap with marked centroid':
        txtPath = 'centroid-mask/labels/' + str(imagePath.name)[:-4] + '.txt'
        centers, _ = self.PixelPoints(txtPath)
        # the image size is 416x416 and 1/8th of that is 52x52
        label = tf.Variable(tf.zeros([nSize, nSize, 2], tf.uint8))

        for CX, CY, classGroup in centers:
            nCX = int((CX/squareSize) * nSize)
            nCY = int((CY/squareSize) * nSize)
            label[nCY, nCX, int(classGroup)].assign(255)

        return label

    def _info(self) -> tfds.core.DatasetInfo:
        "metadata"
        return tfds.core.DatasetInfo(
            builder=self,
            description="""dataset for realtime centroid detection mask/!mask for""",
            supervised_keys=('image', 'label1', 'label2'),

            disable_shuffling=False,
            features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(imageSize, imageSize, 3)),
            'label1': tfds.features.Tensor(shape=(labelSize, labelSize, 2), dtype=tf.float32),
            'label2': tfds.features.Tensor(shape=(labelSize, labelSize, 2), dtype=tf.float32),

            }))

    def _split_generators(self, download_manager:tfds.download.DownloadManager):
        extracted = download_manager.extract('centroid-mask')
        return {
            'train': self._generate_examples(path=extracted / 'train'),
            'test': self._generate_examples(path=extracted / 'test'),
            'valid': self._generate_examples(path=extracted / 'valid')
    }

    def _generate_examples(self, path):
        for img_path in path.glob('*.jpg'):
            # returning the name and values
            yield img_path.name, {
                'image': img_path,
                'label1': self.localizationMap(img_path),
                'label2': self.labelImage(img_path)
            }



#%%
