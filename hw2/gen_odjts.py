import cv2
import os
import argparse
from tqdm import tqdm


def odgt(img_path):
    seg_path = img_path.replace("images", "annotations")

    if os.path.exists(seg_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        odgt_dic = {}
        odgt_dic["fpath_img"] = img_path
        odgt_dic["fpath_segm"] = seg_path
        odgt_dic["width"] = h
        odgt_dic["height"] = w
        return odgt_dic
    else:
        print("Not a valid seg path: ", seg_path)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset path")
    args = parser.parse_args()
    modes = ["train", "val", "test"]
    scenes = [
        "apartment_0",
        "apartment_1",
        "apartment_2",
        "frl_apartment_0",
        "frl_apartment_1",
        "hotel_0",
        "office_0",
        "office_1",
        "room_0",
        "room_1",
        "room_2",
    ]

    try:
        assert os.path.exists(args.dataset)
        assert os.path.exists(os.path.join(args.dataset, "images"))
        assert os.path.exists(os.path.join(args.dataset, "annotations"))
    except:
        print("Not a valid dataset path")
        exit(1)
    os.makedirs(os.path.join(args.dataset, "odgts"), exist_ok=True)
    for m in modes:
        os.makedirs(os.path.join(args.dataset, "odgts", m), exist_ok=True)
    for m in modes:
        print("Generating odgts for mode: ", m)
        for scene in tqdm(scenes):
            img_dir = os.path.join(args.dataset, "images", m)
            img_paths = sorted(os.listdir(img_dir))
            odgt_list = []
            for img_path in tqdm(img_paths, leave=False):
                odgt_dic = odgt(os.path.join(img_dir, img_path))
                if odgt_dic is not None:
                    odgt_list.append(odgt_dic)

            with open(
                os.path.join(args.dataset, "odgts", m, scene + ".odgt"), "w"
            ) as f:
                for odgt_dic in odgt_list:
                    f.write(str(odgt_dic).replace("'", '"') + "\n")
