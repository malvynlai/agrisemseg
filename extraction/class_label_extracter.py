from PIL import Image
import os
import numpy as np
import csv


def contains_white_pixel(image):
    width, height = image.size
    image = image.convert('RGB')
    image_array = np.array(image)
    return np.any(np.all(image_array == [255, 255, 255], axis=-1))


def get_correct_classes(filepath):
    total_classes = []
    class_dict = {}
    output_file = '../supervised/Agriculture-Vision-2021/val/single_class_labels_safe.csv'
    for i in range(len(os.listdir(filepath))):
        print(f'on current folder {i + 1} of 9')
        class_labels = []
        current_class = os.listdir(filepath)[i]
        class_filepath = filepath + f'/{current_class}'
        for j in range(len(os.listdir(class_filepath))):
            img_file = os.listdir(class_filepath)[j]
            image = Image.open(os.path.join(class_filepath, img_file))
            if contains_white_pixel(image):
                class_labels.append(1)
            else:
                class_labels.append(0)
        total_classes.append(class_labels)
        class_dict[os.listdir(class_filepath)[i]] = class_labels
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(class_labels)
    return total_classes, class_dict


def make_class_files(class_labels):
    array = np.array(class_labels)
    single_class_label = '../supervised/Agriculture-Vision-2021/val/single_class_label.csv'
    np.savetxt(single_class_label, array, delimiter=',', fmt='%d')
    array = array.T
    all_class_label = '../supervised/Agriculture-Vision-2021/val/all_class_label.csv'
    np.savetxt(all_class_label, array, delimiter=',', fmt='%d')

    output_folder = filepath + '/ground_classes'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames = []
    for file in class_dict:
        filenames.append(file)
    for i, array in enumerate(array):
        filename = filenames[i]
        classpath = os.path.join(output_folder, f'{filename}.txt')
        np.savetxt(classpath, array, fmt='%d')
        print(f'saved img {filename}, count: {i}')


def main():
    class_list, class_dict = get_correct_classes('../supervised/Agriculture-Vision-2021/val/labels')
    make_class_files(class_list)


if __name__ == '__main__':
    main()