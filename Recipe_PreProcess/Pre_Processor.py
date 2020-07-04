import json


class Pre_Processor:
    def __init__(self, json_file):
        with open(json_file) as json_obj:
            self.recipe_dict = json.load(json_obj)

    def raw_to_processed(self):
        i = 0
        for recipe in self.recipe_dict:
            print(self.recipe_dict[recipe])
            i += 1
            if i > 10:
                break
