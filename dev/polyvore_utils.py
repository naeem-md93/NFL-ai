import os
import json
import pandas as pd
from tqdm import tqdm


def read_json_file(_path):
    with open(_path) as json_file:
        return json.load(json_file)


def get_categories(_category_path, _categories):

    _categories_df = pd.read_csv(_category_path, header=None)

    _categories_dict = {}
    for i, row in _categories_df.iterrows():
        if row[1] in _categories:
            _categories_dict[row[0]] = row[1]

    return _categories_dict


def get_data(_data_path, _category_path, _categories):

    _orig_data = read_json_file(_data_path)

    _categories_dict = get_categories(_category_path, _categories)

    _final_data = []
    for _set_data in tqdm(_orig_data):
        _tmp_items = []
        for _item_data in _set_data["items"]:
            if _item_data["categoryid"] in _categories_dict:
                _item_data["categoryname"] = _categories_dict[_item_data["categoryid"]]
                _tmp_items.append(_item_data)

        if len(_tmp_items) > 1:
            _set_data["items"] = _tmp_items
            _final_data.append(_set_data)

    return _final_data