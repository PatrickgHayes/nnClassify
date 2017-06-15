import numpy as np
import Constants
from MyUtils_ import MyUtils


### This class is for evaluating how the performance of our predictions
### It calculates things like the balanced error rate
class Evaluator:

    ### METHODS #################################################################
    # Purpose - A balnced error rate prevents the under represetation of 
    #           ceratin categories from causing false confidence. If we
    #           test on a data set that has 99,000 images of 2 eyed worms
    #           and only 1,000 images of 0 eyed worms, we could get 99% 
    #           accuracy just be always predicting 2 eyes.
    #
    # Takes - labels_test: the true label for each image
    #         labels_predicted: the labels that we predicted
    #         encoding: used to convert the one hot encoding of a category
    #                   into text
    #
    # Returns - Displays the balanced error rate
    @staticmethod
    def calc_balanced_accuracy(labels_test, labels_predicted):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        ns = 0
        abnormal = [Constants.ABNORMAL]
        normal = [Constants.NORMAL]
        skip = [Constants.NOT_SURE, Constants.WEIRD]
        for idx in range(0,len(labels_test)):
            pred = labels_predicted[idx]
            true = labels_test[idx]

            # If the model isn't sure then don't factor
            # it into accuracy metric
            if pred in skip:
                ns += 1
            elif pred in abnormal and true in abnormal:
                tp += 1
            elif pred in abnormal and true not in abnormal:
                fp += 1
            elif pred not in abnormal and true in abnormal:
                fn += 1
            elif pred not in abnormal and true not in abnormal:
                tn += 1

        if tp + fn == 0 or tn + fp == 0:
            bal_err_rate = 1
        else:
            bal_err_rate = 1 - 0.5 * (tp * 1.0 /(tp + fn) + tn * 1.0 /(tn+fp))

        return (bal_err_rate, tp, fp, tn, fn, ns)

    #Finds the category with the highest confidence and the category
    # with the second highest. If the difference between the two
    # category does not meet the thresehold then it returns Not Sure
    @staticmethod
    def max_pred(labels, encoding):
        THRESHOLD = 0.0 
        maxIdx = None
        second_place_val = None
        maxVal = None
        for idx, val in enumerate(labels[1:len(labels)]):
            if maxIdx is None or val > maxVal:
                maxIdx = idx 
                second_place_val = maxVal
                maxVal = val 
            elif (val <= maxVal and 
                    (second_place_val == None or val > second_place_val)):
                second_place_val = val 

        if (second_place_val is not None and 
                (maxVal - second_place_val > THRESHOLD)):
            return encoding[maxIdx]
        else:
            return Constants.NOT_SURE 


    #Finds the category with the highest confidence and the category
    # with the second highest. If the difference between the two
    # category does not meet the thresehold then it returns Not Sure
    @staticmethod
    def is_it_normal(labels):
        image_confidance = 0.85
        confidance_count_threshold = 15
        maybe_threshold = 2 
        confidance_count = 0 
        for i in range(0,len(labels)):
            if labels[i][1] >= image_confidance:
                confidance_count += 1

        if confidance_count >= confidance_count_threshold:
            return Constants.NORMAL
        elif confidance_count >= maybe_threshold:
            return Constants.NOT_SURE
        else:
            return Constants.ABNORMAL