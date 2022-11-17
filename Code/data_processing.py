"""
The INBreast dataset is assumed to be in the /Dataset/INBreast folder
There are 410 images in total
"""

import pandas as pd
import os
import re
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pickle
from skimage.transform import resize

import cv2

path = "./Dataset/INBreast/"
processed_data_folder = "./ProcessedData/"
debug_code = False
scale_height = 32

# Set this to a value not in the range [0, 410] if debugging is not needed
debug_build_DF_count = -1

def test_function():
    #make_jpg_images()
    data = read_image_data("INBreastPNG_RCC/", "png")
    pickle_list(data)


def simple_dicom_to_png(image_folder, filename, destination_folder):
    source = pydicom.dcmread(path + image_folder + filename)
    pixel_array_numpy = source.pixel_array

    destination = filename.replace(".dcm", ".jpg")
    print(destination)
    cv2.imwrite(path + destination_folder + destination, pixel_array_numpy)


def make_jpg_images(image_folder="AllDICOMs/"):
    destination_folder = "TestImages/"
    image_file_list = build_image_file_list(image_folder)
    count = 1
    print("make_jpg_images")
    print(image_file_list[0])
    for image_file_name in image_file_list:
        simple_dicom_to_png(image_folder, image_file_name, destination_folder)
        print("Count =", count)
        count += 1


def read_image_data(image_folder="TestImages/", file_format="png"):
    image_file_list = build_image_file_list(image_folder, file_format)
    pixel_data_list = []
    count = 1

    for file_name in image_file_list:
        image = cv2.imread(path + image_folder + file_name, 1)
        pixel_data = numpy_array_scaler(image, scale_height, scale_height)
        print("Count = ", count)
        count += 1
        print("Shape = ", pixel_data.shape)
        # plt.imshow(pixel_data, cmap="gray")
        # plt.show()
        pixel_data_list.append(pixel_data)

    print("Image data list built")
    return (pixel_data_list, image_file_list)
    

def pickle_list(data, filename="PixelData.pkl", path=processed_data_folder):
    with open(path + filename, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path=processed_data_folder, filename="PixelData.pkl"):
    with open(path + filename, "rb") as f:
        data = pickle.load(f)
    return data


def make_3channel_to_1channel(data):
    # Takes an array of narrays with the shape (a, a, 3) and returns an array of narrays of the shape (a, a, 1)
    new_values = []
    for value in data:
        new_values.append(value[:,:,:1])
    return new_values


def make_3d_array(pixel_data):
    # pixel_data is originally an array of 2D arrays
    # Using np.dstack() function, we are stacking the 2D arrays to create a new 3D array
    new_array = np.dstack(pixel_data)
    # After stacking, the shape is (x, x, 410)  where x is the height/width of the 2D array
    # As we want the 3rd Axis to be at the first, we use the np.moveaxis() function
    new_array = np.moveaxis(new_array, -1, 0)
    # Final shape = (410, x, x)
    print("new array shape=", new_array.shape)
    print(type(new_array))
    return new_array


def fresh_initialization():
    create_processed_data_folder()
    df = build_DataFrame()
    
    # TODO - Add pre-processing steps here


    save_DataFrame(df)
    print("Fresh Initialization & Pickling of DataFrame completed!")


def show_samples(df, plot_count=4):
    plots = []
    titles = []
    for i in range(plot_count):
        random_index = random.randint(0, len(df))
        plots.append(df.iloc[random_index]["PixelData"])
        titles.append(f"Bi-RADS Category {df.iloc[random_index]['BiRads']}")
    figsize = (8, 8)
    suptitle = "INBreast Dataset - Image Set"
    show_data(plots, titles, figsize, suptitle)


def show_data(pixel_data, titles=None, figsize=None, suptitle=None):
    plot_count = len(pixel_data)
    if plot_count > 36:
        plot_count = 36  # Hard limit of maximum 36 subplots in matplotlib
    rsize, csize = calculate_gridsize(plot_count)
    plt.figure(figsize=figsize)
    plt.suptitle(suptitle)
    for i in range(1, plot_count + 1):
        plt.subplot(rsize, csize, i)
        plt.imshow(pixel_data[i-1], cmap="gray")
        plt.title(titles[i-1])
    plt.show()


def calculate_gridsize(plot_count):
    # Used to determine the gridsize for plotting
    # Handling for single plot
    if plot_count == 1:
        return plot_count, plot_count

    # Used to calculate the value of nearest power of 2 which is greater than the number of plots
    two_power = 2
    while plot_count > two_power:
        two_power = two_power * 2
    size = int(math.log2(two_power))

    # Handling for log value is 1 but number of plots is 2
    if size == 1:
        return size, size+1
    else:
        return size, size


def scale_down_image(path, height, width=None):
    # Square Image 
    if width is None:
        width = height

    data = pydicom.read_file(path).pixel_array
    return numpy_array_scaler(data, height, width)


def numpy_array_scaler(data,height, width):
    scaled_down_data =  resize(data, (height, width), anti_aliasing=True)
    scaled_down_data = scaled_down_data.astype('float32')
    return scaled_down_data


def match_birad_to_filename(filename_list):
    birads = []
    print(len(filename_list))
    xls_data = load_xls_data()
    for filename in filename_list:
        splits = filename.split("_")
        x = xls_data[xls_data["File Name"] == splits[0]]["Bi-Rads"].iloc[0]
        birads.append(x)
    return birads


def build_DataFrame(image_folder="AllDICOMs/", show_progress=False):
    image_file_list = build_image_file_list(image_folder)
    xls_data = load_xls_data()
    # Storing the data temporarily for DataFrame construction
    filename = []
    laterality = []
    view = []
    acr = []
    birads = []
    mass = []
    micros = []
    distortion = []
    asymmetry = []
    pixel_data = []

    # Pre-requisites for Progress Tracker
    count = 0
    file_count = len(image_file_list)
    print("DataFrame Building Progress Tracker")

    # Build the individual arrays
    for image_file_name in image_file_list:
        """
            File Name Format
            FileNameNumber_someID_MG_Laterality_View_ANON.dcm
            Ex - 20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm

            splits[0] = FileNameNumber
            splits[1] = someID
            splits[2] = MG
            splits[3] = Laterality
            splits[4] = View
            splits[5] = ANON.dcm
        """

        # Progress Tracker
        progress = count / file_count * 100
        print(f"DataFrame Building Progress = {progress:.2f} %")
        count += 1

        # Debug Code - Used to build a small DataFrame
        if count == debug_build_DF_count:
            break

        splits = image_file_name.split("_")
        filename.append(splits[0])
        laterality.append(splits[3])
        view.append(splits[4])
        acr.append(xls_data[xls_data["File Name"] == splits[0]]["ACR"].iloc[0])
        birads.append(xls_data[xls_data["File Name"] == splits[0]]["Bi-Rads"].iloc[0])
        mass.append(xls_data[xls_data["File Name"] == splits[0]]["Mass "].iloc[0])
        micros.append(xls_data[xls_data["File Name"] == splits[0]]["Micros"].iloc[0])
        distortion.append(xls_data[xls_data["File Name"] == splits[0]]["Distortion"].iloc[0])
        asymmetry.append(xls_data[xls_data["File Name"] == splits[0]]["Asymmetry"].iloc[0])
        image_filepath = path + image_folder + image_file_name
        # Appending the scaled down image
        pixel_data.append(scale_down_image(image_filepath, height=scale_height))

    # Build the DataFrame
    df = pd.DataFrame({
            "Filename" : filename,
            "Laterality" : laterality,
            "View" : view,
            "ACR" : acr,
            "BiRads" : birads,
            "Mass" : mass,
            "Micros" : micros,
            "Distortion" : distortion,
            "Asymmetry" : asymmetry,
            "PixelData" : pixel_data
        })

    print("DataFrame Built successfully!")
    if debug_code:
        print("Start - build_DataFrame() function debug data")
        print(df)
        print(df.info())
        print("End - build_DataFrame() function debug data")
        show_samples(df)
    return df


def create_processed_data_folder(folder_name=processed_data_folder):
    # Creates the 'ProcessedData' folder if it does not exist already
    try:
        os.makedirs(folder_name)
        print("'ProcessedData' folder created")
    except OSError as e:
        print("'ProcessedData' folder already exists")


def load_xls_data(xls_filename="INbreast.xls"):
    data = pd.read_excel(path + xls_filename)
    # Drop last two rows as they are not relevant data. See Row 412 and 413 in the .XLS file
    data = data[0:410]
    data = data.astype({"File Name" : int}) # Direct conversion from float -> str causes the decimal points to remain
    data = data.astype({"File Name" : str})
    return data


def build_image_file_list(image_folder, file_format="dcm"):
    # The images are in the /AllDICOMs folder
    file_list = os.listdir(path + image_folder)
    # Keeping only .dcm files
    dcmRegex = re.compile(r"\." + file_format + r"$")
    image_file_list = [x for x in file_list if dcmRegex.search(x)]
    return image_file_list


def save_DataFrame(df, path=processed_data_folder, filename="INBDataFrame.pkl"):
    # 'Pickles' the data
    df.to_pickle(path + filename)


def load_DataFrame(path=processed_data_folder, filename="INBDataFrame.pkl"):
    # Loads the 'pickled' data
    try:
        df = pd.read_pickle(path + filename)
        print("Pickled data loaded successfully!")
        return df
    except FileNotFoundError:
        # pickled data does not exist. Need to reinitialize the data
        print("Pickled data does not exist. Performing fresh initialization")
        fresh_initialization()
        df = pd.read_pickle(path + filename)
        print("Pickled data loaded successfully!")
        return df
