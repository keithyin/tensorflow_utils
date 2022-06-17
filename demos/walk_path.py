from __future__ import print_function

import os

if __name__ == '__main__':
    for folder_name, sub_folder, filenames in os.walk("./"):
        if "git" in folder_name:
            continue
        print("{}, {}, {}".format(folder_name, sub_folder, filenames))