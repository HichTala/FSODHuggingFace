from dataset_transformations.coco_format import COCO
from datasets import load_dataset
from configs.dataset import dota

def main():
    builder = COCO('dota', dota.ANNOTATIONS_PATH, dota.IMAGES_PATH, dota.FEATURES)
    builder.download_and_prepare()

    dataset = builder.as_dataset()
    dataset.push_to_hub("HichTala/dota")

if __name__ == "__main__":
    main()