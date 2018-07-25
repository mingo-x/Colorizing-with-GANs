import argparse
import json
import pickle


def _json_to_dict(json_path, dict_path):
    with open(json_path, 'r') as fin:
        data = json.load(fin)

    class_to_id_dict = {}
    for index in data:
        class_number = data[index][0]
        class_to_id_dict[class_number] = index
        print(class_number, index)

    with open(dict_path, 'wb') as fout:
        pickle.dump(class_to_id_dict, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('vgg_util')
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--dict_path', type=str)
    args, _ = parser.parse_known_args()

    _json_to_dict(args.json_path, args.dict_path)
