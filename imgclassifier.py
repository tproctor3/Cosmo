#!/usr/bin/env python

##############
#### Your name: Jacqueline Doolittle
##############

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color


class ImageClassifier:
    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir + "*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return (data, labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data
        feature_data = []
        for image in data:
            gray_image = color.rgb2gray(image)
            hist_image = exposure.equalize_hist(gray_image)
            gauss_image = filters.gaussian(hist_image)
            features = feature.hog(gauss_image, orientations=10, pixels_per_cell=(24, 24),
                                   cells_per_block=(8, 8), block_norm='L2-Hys', transform_sqrt=True)
            feature_data.append(features)

        # Please do not modify the return type below
        return (feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier
        self.classifer = svm.SVC(kernel='linear')
        self.classifer.fit(train_data, train_labels)

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels

        # Please do not modify the return type below
        predicted_labels = self.classifer.predict(data)
        return predicted_labels


def train_cozmo():
    img_clf = ImageClassifier()

    (normal_raw, normal_labels) = img_clf.load_data_from_folder('./train/')
    (drones_raw, drones_labels) = img_clf.load_data_from_folder('./train/drones/')
    (hands_raw, hands_labels) = img_clf.load_data_from_folder('./train/hands/')
    (inspections_raw, inspections_labels) = img_clf.load_data_from_folder('./train/inspections/')
    (nones_raw, nones_labels) = img_clf.load_data_from_folder('./train/nones/')
    (orders_raw, orders_labels) = img_clf.load_data_from_folder('./train/orders/')
    (places_raw, places_labels) = img_clf.load_data_from_folder('./train/places/')
    (planes_raw, planes_labels) = img_clf.load_data_from_folder('./train/planes/')
    (trucks_raw, trucks_labels) = img_clf.load_data_from_folder('./train/trucks/')

    train_labels = np.concatenate((drones_labels, hands_labels, inspections_labels, nones_labels, orders_labels,
                                   places_labels, planes_labels, trucks_labels, normal_labels))

    train_drones = img_clf.extract_image_features(drones_raw)
    train_hands = img_clf.extract_image_features(hands_raw)
    train_inspections = img_clf.extract_image_features(inspections_raw)
    train_nones = img_clf.extract_image_features(nones_raw)
    train_orders = img_clf.extract_image_features(orders_raw)
    train_places = img_clf.extract_image_features(places_raw)
    train_planes = img_clf.extract_image_features(planes_raw)
    train_trucks = img_clf.extract_image_features(trucks_raw)
    train_normal = img_clf.extract_image_features(normal_raw)

    train_data = np.concatenate((train_drones, train_hands, train_inspections, train_nones, train_orders, train_places,
                                 train_planes, train_trucks, train_normal))

    img_clf.train_classifier(train_data, train_labels)

    return img_clf


def test_cozmo(img_clf, picture):
    test_data = img_clf.extract_image_features([picture])
    predicted_labels = img_clf.predict_labels(test_data)
    print(predicted_labels[0])
    return predicted_labels[0]


def main():
    img_clf = ImageClassifier()

    # load images
    #(train_raw, train_labels) = img_clf.load_data_from_folder('./train/')

    #(test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    (drones_raw, drones_labels) = img_clf.load_data_from_folder('./train/drones/')
    (hands_raw, hands_labels) = img_clf.load_data_from_folder('./train/hands/')
    (inspections_raw, inspections_labels) = img_clf.load_data_from_folder('./train/inspections/')
    (nones_raw, nones_labels) = img_clf.load_data_from_folder('./train/nones/')
    (orders_raw, orders_labels) = img_clf.load_data_from_folder('./train/orders/')
    (places_raw, places_labels) = img_clf.load_data_from_folder('./train/places/')
    (planes_raw, planes_labels) = img_clf.load_data_from_folder('./train/planes/')
    (trucks_raw, trucks_labels) = img_clf.load_data_from_folder('./train/trucks/')

    train_labels = np.concatenate((drones_labels, hands_labels, inspections_labels, nones_labels, orders_labels,
                                   places_labels, planes_labels, trucks_labels))

    # convert images into features
    #train_data = img_clf.extract_image_features(train_raw)

    train_drones = img_clf.extract_image_features(drones_raw)
    train_hands = img_clf.extract_image_features(hands_raw)
    train_inspections = img_clf.extract_image_features(inspections_raw)
    train_nones = img_clf.extract_image_features(nones_raw)
    train_orders = img_clf.extract_image_features(orders_raw)
    train_places = img_clf.extract_image_features(places_raw)
    train_planes = img_clf.extract_image_features(planes_raw)
    train_trucks = img_clf.extract_image_features(trucks_raw)

    train_data = np.concatenate((train_drones, train_hands, train_inspections, train_nones, train_orders, train_places,
                                 train_planes, train_trucks))

    # train model and test on training data

    img_clf.train_classifier(train_data, train_labels)

    # test model


if __name__ == "__main__":
    main()
