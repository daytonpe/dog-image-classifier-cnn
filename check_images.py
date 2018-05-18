#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Pat Dayton
# DATE CREATED: 13 May 2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep, clock
from os import listdir

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Main program function defined below
def main():
    # 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()

    # 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels()

    # 4. Define classify_images() function to create the classifier
    # labels with the classifier function uisng in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)


    # 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # TODO: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    #  1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    #  1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    # Prints overall runtime in format hh:mm:ss
    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
      str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" +
      str( int(  ( (tot_time % 3600) % 60 ) ) ) )


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="pet_images/", help="path to folder of images")
    parser.add_argument("--arch", type=str, default="vgg", help= "chosen model")
    parser.add_argument("--dogfile",type=str, default="dognames.txt", help="text file that has dog names")
    return parser.parse_args()

def get_pet_labels():
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    keys = listdir("pet_images/")
    values = []
    for i in keys:
        label = i.strip().lower().split('.')
        label = label[0].split('_')[0:-1]
        label = " ".join(label)
        values.append(label)
    petlabel_dic = dict()


    for idx in range(0, len(keys), 1):
        if keys[idx] not in petlabel_dic:
             petlabel_dic[keys[idx]] = values[idx]
        else:
             print("** Warning: Key=", keys[idx],
                   "already exists in petlabel_dic with value =", petlabel_dic[keys[idx]])

    return petlabel_dic

    #Iterating through a dictionary printing all keys & their associated values
    # print("\nPrinting all key-value pairs in dictionary petlabel_dic:")
    # for key in petlabel_dic:
    #     print("Key=", key, "   Value=", petlabel_dic[key])

def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()

    #use classifier to determine computer's attempt at labeling
    files = listdir(images_dir)

    for file in files:

        pet_image_label = petlabel_dic[file]

        #concatenate the image directory with the specific image name
        #lower() and strip() to make sure matches aren't missed due to capitalization
        classifier_label = classifier(images_dir+file, model).strip().lower()

        match = 0
        if pet_image_label in classifier_label:
            match = 1

        results_dic[file] = [pet_image_label, classifier_label, match]

        # print results_dic one at a time with labels
        # print('file: ',file,'\nlabel: ',pet_image_label,'\nclassification: ',classifier_label,'\nmatch?: ',match, '\n')

    # another way to print results
    # for k, v in results_dic.items():
    #     print(k, v)

    return results_dic

def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    breed_list = []
    with open(dogsfile) as f:
        breed = f.readline()
        while breed != "":
            breed = breed.rstrip().lower()
            if (breed in breed_list):
                print('Warning. breed ', breed, ' is already in breed_list')
            else:
                breed_list.append(breed)
            breed = f.readline()


    for key, value in results_dic.items():
        # print (key, value)
        #append index 3 value -- pet image label is dog
        if (value[0] in breed_list):
            results_dic[key].append(1)
        else:
            results_dic[key].append(0)

        #append index 4 value -- pet image classification is dog
        if (value[1] in breed_list):
            results_dic[key].append(1)
        else:
            results_dic[key].append(0)

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    results_stats = dict()

    num_images = len(results_dic)
    num_dog_matches = 0
    num_dog_images = 0
    num_not_dog_matches = 0
    num_not_dog_images = 0
    num_correct_breed_matches = 0
    num_label_matches = 0

    for key, value in results_dic.items():

        if (value[3] == 1 and value[4] == 1):
            num_dog_matches += 1

        if (value[3] == 1):
            num_dog_images +=1;

        if (value[3] == 0 and value[4] == 0):
            num_not_dog_matches += 1

        if (value[3] == 0):
            num_not_dog_images +=1

        if (value[3] == 1 and value[2] == 1 ):
            num_correct_breed_matches+=1

        if (value[2] == 1):
            num_label_matches +=1

    #prevent divide by zero errors in percentage calculations
    if num_dog_images != 0:
        percent_correct_dog = int((num_dog_matches / num_dog_images)*100)
        percent_dog_breed_match = int((num_correct_breed_matches / num_dog_images)*100)

    if num_not_dog_images != 0:
        percent_correct_not_dog = int((num_not_dog_matches / num_not_dog_images)*100)

    if num_images != 0:
        percent_label_match = int((num_label_matches / num_images)*100)

    #add everything to dictionary
    results_stats['num_images'] = num_images
    results_stats['num_dog_matches'] = num_dog_matches
    results_stats['num_dog_images'] = num_dog_images
    results_stats['num_not_dog_matches'] = num_not_dog_matches
    results_stats['num_not_dog_images'] = num_not_dog_images
    results_stats['num_correct_breed_matches'] = num_correct_breed_matches
    results_stats['num_label_matches'] = num_label_matches
    results_stats['percent_correct_dog'] = percent_correct_dog
    results_stats['percent_correct_not_dog'] = percent_correct_not_dog
    results_stats['percent_dog_breed_match'] = percent_dog_breed_match
    results_stats['percent_label_match'] = percent_label_match

    return results_stats

def print_results(results_dic, results_stats, model, print_incorrect_dogs, print_incorrect_breed):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """

    print()
    print('***RESULTS FOR ', model.upper(), ' ARCHITECTURE CNN***\n')
    print('                 Images: ', results_stats['num_images'])
    print('             Dog Images: ', results_stats['num_dog_images'])
    print('         Not Dog Images: ', results_stats['num_not_dog_images'])
    print()
    print('          Percent Match: ', results_stats['percent_label_match'], '%')
    print('   Percent Correct Dogs: ', results_stats['percent_correct_dog'], '%')
    print('  Percent Correct Breed: ', results_stats['percent_dog_breed_match'], '%')
    print('Percent Correct Not Dog: ', results_stats['percent_correct_not_dog'], '%')
    print()

    if print_incorrect_breed:
        print('*** INCORRECT DOG BREED ASSIGNMENT ***\n')
        for key, value in results_dic.items():
            if value[2] == 0 and (value[3]==1 or value[4]):
                print('Real: {0:20}  Classifier: {1:20}'.format(value[0], value[1]))

    if print_incorrect_dogs:
        print('\n*** INCORRECT DOG ASSIGNMENT***\n')
        for key, value in results_dic.items():
            if value[3] == 0 and value[4] == 1:
                print(key)

# Call to main function to run the program
if __name__ == "__main__":
    main()
