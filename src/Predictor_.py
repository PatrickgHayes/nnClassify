import cv2
import numpy as np
import MyUtils_
import ImageCropper_
import Evaluator_
from MyUtils_ import MyUtils
from ImageCropper_ import ImageCropper
from Evaluator_ import Evaluator
import pickle

# This class contains the logic displaying our predictions to the user
class Predictor:
  #### CLASS VARIABLES ########################################################
    CONTRAST = "Contrast"
  ##### END CLASS VARIABLES ###################################################
 
 ### METHODS #################################################################   
    # Purpose - Takes in folder with all the images for a well (uncropped)
    #           It first crops all the images, then it predicts the number
    #           of eyes for each image. It prints these results to file. At 
    #           the top of this file is the prediction for the entire well
    #
    # Takes - folder_path: location of the well to predict
    #         encoding: used for converting from one-hot to text
    #         model: the neural network to use for prediction
    #         op_params: Contrast is an optional parameter.
    #                    if they pass contrast as a parameter increase
    #                    the contrast of the test images
    #
    # Returns - a string with the results for the well.
    #           Also creates a file and writes the results there
    @staticmethod
    def predict_well(imgs_path, results_path, encoding, model, **op_params):
        images = pickle.load(open(imgs_path,"rb"))
        pred = model.predict(images)

        # Write the results file
        with  open(results_path,'w') as results:
            #Write the header
            results.write("Name\tPredicted Category")
            for cat in encoding:
                results.write("\t" + cat)
            results.write('\n')

            # Write the results for the well
            # Divide each entry of the sum by the number of images to get
            # the average. This makes the confidence metric invariant to
            # the number of images
            # TODO pred_cat = MyUtils.max_pred(
                   # [val * 1.0 / len(pred) for val in sum(pred)],encoding)
            pred_cat = Evaluator.is_it_normal(pred)

            well_results = "Well\t" + pred_cat
            for i in range(0,len(encoding)):
                well_results = well_results + '\t' + str(sum(pred)[i] * 1.0
                                / len(pred))
            results.write(well_results)

            #Write the results for each individual image
            for i in range(0,len(pred)):
                results.write('\n')
                results.write(str(i+1)+"_img" + '\t'
                        + Evaluator.max_pred(pred[i], encoding))
                for confidence in pred[i]:
                    results.write('\t' + str(confidence))
            return well_results, pred_cat


    # Purpose - Predict all the wells in test set
    #
    # Takes - testset_path: the absolute path of the test set
    #
    # Returns - nothing, creates a file and writes the results there
    @staticmethod
    def predict_all_well(testset_path, encoding, model, **op_params):
        all_well_results = list()
        actual_labels = list()
        predicted_labels = list()

        categories = MyUtils.listdir_nohidden(testset_path 
                                                + "/Transfer_Values/")

        for category in categories:
            print ("Predicting " + category)
            wells = MyUtils.listdir_nohidden(testset_path
                                            + "/Transfer_Values/" + category)

            for well in wells:
                print ("   Predicting well " + well )
                #Get the well results from each well
                # Increas the contrast of the image if they asked for it
                trans_value_path = (testset_path + "/Transfer_Values/"
                                        + category + "/" + well)
                results_path = (testset_path+ "/Results/" 
                                        + category + "/" + well + ".txt")
                well_results, pred_cat = Predictor.predict_well(
                                             trans_value_path,
                                             results_path,
                                             encoding,
                                             model)
 
                #Add the well results to a list with all the well results
                #append at the end of the line what category the well actually
                #is
                all_well_results.append(well_results + '\t' + category)
                actual_labels.append(category)
                predicted_labels.append(pred_cat)

        # Calculate the Balanced Error Rate
        BER, TP, FP, TN, FN, NS = Evaluator.calc_balanced_accuracy(
                                                    actual_labels,
                                                    predicted_labels)
        
        # Write the results for the test set in a new file
        with open(testset_path + "/Results/Test_Set.txt",'w') as results:
            results.write("All Well Results\n")
            results.write("Balanced Error Rate: " + str(BER) + '\n')
            results.write("True Positive Count "
                            + "(Worm predicted abnormal and actually was): "
                            + str(TP) + '\n')
            results.write("True Negative Count "
                            + "(Worm predicted normal and actually was): "
                            + str(TN) + '\n')
            results.write("False Positive Count "
                            + "(Worm predicted abnormal but it was normal): "
                            + str(FP) + '\n')
            results.write("False Negative Count "
                            + "(Worm predicted normal but it was abnormal): "
                            + str(FN) + '\n')
            results.write("Unsure Count "
                     + "(Model lacked the confidence to make a prediction): "
                     + str(NS) + '\n')

            # Empty line between overall results and individual well results
            results.write('\n')

            # Individual Well Results Header
            results.write("Well\tActual_Category\tPredicted_Category")
            for cat in encoding:
                results.write("\t" + cat)
            results.write('\n')
            
            for well_results in all_well_results:
                split_results = well_results.split('\t')
                if len(split_results) < 5:
                    print ("Error\n")
                    continue
                well = split_results[0]
                pred_cat = split_results[1]
                act_cat = split_results[len(split_results)-1]
                confidences = split_results[2:len(split_results)-1]
                results.write(well + '\t' + act_cat + '\t' + pred_cat)
                for cat in confidences:
                    results.write('\t' + cat)
                results.write('\n')








