import json
import os
import time
import shutil
from tqdm import tqdm
from addict import Dict
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCOConverter():
    """
    Converts labels from COCO format to YOLO (seg or detection)

    Args:
         coco_annot_path (str): COCO annotation path to be converted
         converted_label_path (str): Label path for converted labels to be saved
         image_path (str): Image directory path where images are located
         conversion_mode (str): Label output mode to decide whether as segmentation or detection
    """

    def __init__(self, coco_annot_path: str, converted_label_path: str, image_path: str, conversion_mode: str):
        super().__init__()
        self.coco_annot_path = coco_annot_path
        self.converted_label_path = converted_label_path
        self.image_path = image_path
        self.conversion_mode = conversion_mode

    def from_coco_to_yolo(self):
        with open(self.coco_annot_path, 'r') as json_file:
            data_dict = json.load(json_file)

        if os.path.exists(self.converted_label_path):
            print(f"Saving to {self.converted_label_path}")


        else:
            print(f"Couldn't find a file to save in given path! Creating one...")
            os.mkdir(self.converted_label_path)

        data = Dict(data_dict)
        image_info = data["images"]
        annotations = data["annotations"]
        class_ids = []

        start_time = time.time()
        for i_info in tqdm(image_info):
            lines = []

            for anno in annotations:
                if i_info['id'] == anno['image_id']:
                    if (anno["category_id"] - 1) in class_ids:
                        class_id = class_ids.index((anno['category_id'] - 1))
                    else:
                        class_ids.append((anno["category_id"] - 1))
                        class_id = len(class_ids) - 1

                    line = str(class_id)
                    line += " "

                    if self.conversion_mode == "detection":

                        try:
                            x, y, w, h = anno['bbox']
                            x_centre = (x + (x + w)) / 2
                            y_centre = (y + (y + h)) / 2
                            line += str(x_centre / i_info['width']) + " " + str(y_centre / i_info['height']) + " " + str(
                                w / i_info['width']) + " " + str(h / i_info['height'])
                        except:
                            continue

                        lines.append(line)

                    elif self.conversion_mode == "segmentation":

                        try:
                            for idx, seg in enumerate(anno["segmentation"][0]):
                                if idx % 2 == 0:
                                    line += str(seg / i_info["width"])
                                else:
                                    line += str(seg / i_info["height"])

                        except:
                            continue

                        lines.append(line)

                    else:
                        print("You have entered an invalid conversion mode! Please choose 'detection' or 'segmentation' instead.")

            label_name = self.converted_label_path + "/" + i_info['file_name'].split('.')[0] + ".txt"
            with open(label_name, 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

        end_time = time.time()
        print(f"Process Done! \nSpent time: {end_time - start_time} seconds")

    # def train_test_valid_split(self, label_path: str, train_dest_path: str, test_dest_path: str, valid_dest_path: str):
    def train_test_valid_split(self, image_path = None):

        if image_path is not None:
            self.image_path = image_path

        start_time = time.time()
        train_dest_path = self.converted_label_path + "/" + "train"
        test_dest_path = self.converted_label_path + "/" + "test"
        valid_dest_path = self.converted_label_path + "/" + "valid"

        if os.path.exists(train_dest_path) and os.path.exists(test_dest_path) and os.path.exists(valid_dest_path):
            print('Paths already exist! No need to create them!')

        else:
            print("Couldn't find given paths! Creating...")
            os.mkdir(train_dest_path)
            os.mkdir(test_dest_path)
            os.mkdir(valid_dest_path)
            os.mkdir(train_dest_path + "/" + "images")
            os.mkdir(train_dest_path + "/" + "labels")
            os.mkdir(test_dest_path + "/" + "images")
            os.mkdir(test_dest_path + "/" + "labels")
            os.mkdir(valid_dest_path + "/" + "images")
            os.mkdir(valid_dest_path + "/" + "labels")

        images = [i for i in os.listdir(self.image_path) if i.endswith(".jpeg") or i.endswith(".jpg") or i.endswith(".png")]
        train_images = images[:int(0.80 * len(images))]
        validation_images = images[int(0.80 * len(images)):int(0.90 * len(images))]
        test_images = images[int(0.90 * len(images)):]
        print(
            f"Dataset was seperated Successfully! \nTotal Amount of train images is {len(train_images)} \nTotal Amount of test images is {len(test_images)} \nTotal Amount of validation images is {len(validation_images)}")
        print(f"Saving Seperated Images...")

        for image in tqdm(images):
            if image in train_images:
                shutil.copyfile(self.image_path + "/" + image, train_dest_path + "/images/" + image)
                shutil.copyfile(self.converted_label_path + "/" + image[:-3].split('.')[0] + ".txt",
                                train_dest_path + "/labels/" + image[:-3].split('.')[0] + ".txt")
            elif image in test_images:
                shutil.copyfile(self.image_path + "/" + image, test_dest_path + "/images/" + image)
                shutil.copyfile(self.converted_label_path + "/" + image[:-3].split('.')[0] + ".txt",
                                test_dest_path + "/labels/" + image[:-3].split('.')[0] + ".txt")
            elif image in validation_images:
                shutil.copyfile(self.image_path + "/" + image, valid_dest_path + "/images/" + image)
                shutil.copyfile(self.converted_label_path + "/" + image[:-3].split('.')[0] + ".txt",
                                valid_dest_path + "/labels/" + image[:-3].split('.')[0] + ".txt")

        end_time = time.time()
        print(f"Seperated Images are Saved and the process is Finished! \nSpent Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    # converter = COCOConverter(coco_annot_path = '/home/visio-ai/Desktop/task162/annotations/instances_default.json', image_path = 'task162/images/', converted_label_path = '/home/visio-ai/Desktop/project-workspace/converter_tool/labels2', conversion_mode = "detection")
    # converter.from_coco_to_yolo_detect()
    # converter.train_test_valid_split()
    parser = argparse.ArgumentParser(description = "Converter Tool from COCO to Yolo segmentation or detection.")
    parser.add_argument('--coco-annot-path', '--coco-anno-path', dest = "coco_annot_path", required = True, type = str, help = "Path to COCO annotations")
    parser.add_argument('--image-path', '--img-path', dest = "image_path", required = True, type = str, help = "Path to images")
    parser.add_argument('--converted-label-path', '--label-output-path', dest = "converted_label_path", required = True, type = str, help = "Path to save converted labels.")
    parser.add_argument('--conversion-mode', '--conv-mode', dest = "conversion_mode", required = True, type = str, help = "Mode to choose whether to convert yolo segmentation or yolo detection")
    args = parser.parse_args()

    converter = COCOConverter(
        coco_annot_path = args.coco_annot_path,
        image_path = args.image_path,
        converted_label_path = args.converted_label_path,
        conversion_mode = args.conversion_mode
    )

    converter.from_coco_to_yolo()
    # TODO: Add argument to control splitting as well.

