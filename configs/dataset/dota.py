import datasets

CATEGORY_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "small-vehicle", "large-vehicle", "roundabout", "swimming-pool", "helicopter",
    "soccer-ball-field", "container-crane"
]

FEATURES = datasets.Features(
    {
        "image_id": datasets.Value("int64"),
        "image": datasets.Image(),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "objects": datasets.Sequence({
            "bbox_id": datasets.Value("int64"),
            "category": datasets.ClassLabel(names=CATEGORY_NAMES),
            "bbox": datasets.Sequence(datasets.Value("int64"), 4),
            "area": datasets.Value("int64"),
        })
    }
)

ANNOTATIONS_PATH = "/home/hicham/Documents/datasets/dota/annotations"
IMAGES_PATH = "/home/hicham/Documents/datasets/dota/"