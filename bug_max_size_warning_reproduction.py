import logging

import numpy as np
from transformers.models.detr.image_processing_detr import DetrImageProcessor

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

images = [np.ones((512, 512, 3))]
annotations = [{'image_id': [], 'annotations': []}]
size = {'max_height': 600, 'max_width': 600}

image_processor = DetrImageProcessor()
images = image_processor.preprocess(images, do_resize=True, do_rescale=False, size=size, annotations=annotations, format='coco_detection')