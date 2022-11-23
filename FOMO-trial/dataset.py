import tensorflow_datasets as tfds, tensorflow as tf
import pybboxes as pbx

imageSize = 224
newImageSize = int(imageSize/8)

class tfDataset(tfds.core.GeneratorBasedBuilder):
    """dataset builder for the pre-augmented mask dataset"""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Inital Release',
        'About': """Augmentation in roboflow on the Medical Mask Dataset BY
                    Humans in the loop, https://humansintheloop.org/resources/datasets/medical-mask-dataset/"""}

    def PixelPoints(self, filename:str) -> tuple():
        centers = [] # base list containing all centers for multiple bounding box images
        with open(filename, 'r') as file:
            for line in file.readlines():
                YOLOF = [float(value) for value in line.split()]
                X, Y, W, H = pbx.convert_bbox(YOLOF[1:], from_type="YOLO".lower(), to_type="voc", image_size=(416, 416))
                centerX = X + int((W - X)/2)
                centerY = Y + int((H - Y)/2)

                centers.append((centerX, centerY, YOLOF[0])) # centroid of bounding box along with class group
            return centers

    def labelImage(self, imagePath:str, squareSize=imageSize, nSize=newImageSize) -> 'heatmap with marked centroid':
        txtPath = 'msk_dataset/labels/' + str(imagePath.name)[:-4] + '.txt'
        centers = self.PixelPoints(txtPath)
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
            supervised_keys=('image', 'label'),

            disable_shuffling=False,
            features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(imageSize, imageSize, 3)),
            'label': tfds.features.Tensor(shape=(newImageSize, newImageSize, 2), dtype=tf.uint8)
            }))

    def _split_generators(self, download_manager:tfds.download.DownloadManager):
        extracted = download_manager.extract('msk_dataset')
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
                'label': self.labelImage(img_path)
            }


#%%
