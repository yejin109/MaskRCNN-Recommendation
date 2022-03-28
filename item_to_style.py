import os
import json
from tqdm import tqdm

sources = ['musinsa_codimap', 'codibook']


def name_filling(source, style_id):
    if source == 'codibook':
        return f"codibook_style_{style_id}"
    else:
        return style_id


item_to_style = dict()
for src in sources:
    item_directory = f"data/{src}_data/info/info_item/"
    style_directory = f"data/{src}_data/info/info_style/"

    path_info_items = os.listdir(item_directory)
    for path_info_item in tqdm(path_info_items):
        with open(item_directory + path_info_item, encoding='utf-8') as f:
            info_item = json.load(f)
        if 'style_id' not in info_item.keys():
            continue

        if info_item['title'] not in item_to_style.keys():
            with open(style_directory+name_filling(src, info_item['style_id'])+'.json', encoding='utf-8') as style_f:
                style_info = json.load(style_f)
            item_to_style[info_item['title']] = {'styles': [info_item['style_id']], 'item_link': info_item['link'],
                                                 'style_link': style_info['link']}

        else:
            item_to_style[info_item['title']]['styles'].append(info_item['style_id'])


with open(f"data/item_to_style.json", 'w', encoding='utf-8') as f:
    json.dump(item_to_style, f, indent="\t", ensure_ascii=False)
print()
