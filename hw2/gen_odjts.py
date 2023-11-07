import cv2
import os
import argparse
from tqdm import tqdm


def odgt(img_path, rp=("images", "annotations")):
    seg_path = img_path.replace(rp[0], rp[1])

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


def gen_odgt_list(img_dir: os.path, rp=("images", "annotations")):
    img_paths = sorted(os.listdir(img_dir))
    odgt_list = []
    for img_path in tqdm(img_paths, leave=False):
        odgt_dic = odgt(os.path.join(img_dir, img_path), rp)
        if odgt_dic is not None:
            odgt_list.append(odgt_dic)
    return odgt_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="dataset path")
    parser.add_argument(
        "-p", "--pred", type=str, help="path to rgb from data collection"
    )
    args = parser.parse_args()
    if args.dataset:
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
                odgt_list = gen_odgt_list(img_dir)
                with open(
                    os.path.join(args.dataset, "odgts", m, scene + ".odgt"), "w"
                ) as f:
                    for odgt_dic in odgt_list:
                        f.write(str(odgt_dic).replace("'", '"') + "\n")
    else:
        print("No dataset path given. Skipping dataset mode")
    if args.pred is not None:
        try:
            assert os.path.exists(args.pred)
            assert os.path.exists(os.path.join(args.pred, "rgb"))
            assert os.path.exists(os.path.join(args.pred, "semantic"))
        except:
            print("Not a valid data collection path")
            exit(1)
        print("Generating odgts for mode: pred")
        os.makedirs(os.path.join(args.pred, "odgts"), exist_ok=True)
        img_dir = os.path.join(args.pred, "rgb")
        odgt_list = gen_odgt_list(img_dir, rp=("rgb", "semantic"))
        with open(os.path.join(args.pred, "odgts", "pred.odgt"), "w") as f:
            for odgt_dic in odgt_list:
                f.write(str(odgt_dic).replace("'", '"') + "\n")
    else:
        print("No pred path given. Skipping pred mode")
