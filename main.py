# Empirical Bayesian Logistic Regression for MNIST & USPS

from load_data import*
from sklearn.metrics import classification_report
from VBLR import*
from utilities import*


def main():
    train_image, train_labels, raw_train_labels, \
    validation_image, valid_labels, raw_valid_labels, \
    test_dataset, test_labels, raw_test_labels = import_MNIST()
    print("MNIST Data fetched, done")
    print('------------------------------')
    validation_usps, validation_usps_label = load_usps('data/usps_test_image.pkl', 'data/usps_test_label.pkl')
    print('usps validation_image: ', validation_usps.shape);
    print('usps validation label: ', validation_usps_label.shape)
    print("USPS Data fetched, done")
    print('------------------------------')


    x_train = train_image[0:50000]
    y_train = raw_train_labels[0:50000]
    x_test = validation_image
    y_test = raw_valid_labels
    x_valid = validation_image[0:10000];
    y_valid = raw_valid_labels[0:10000];

    usps_test = validation_usps
    usps_label = validation_usps_label




    if(True):
        vblr = VBLogisticRegression().fit(x_train, y_train);
        vblr_prediction = vblr.try_predict(x_valid)
        print("\n === VBLogisticRegression on Validation set (MNIST)  ===")
        print(classification_report(y_valid, vblr_prediction))
        # cnf_vblr = metrics.confusion_matrix(y_valid, vblr_prediction)
        # plot_confusion_matrix(cnf_vblr,normalize=False,title='Confusion matrix(VBLR)')
        # plt.show()

        print("\n === VBLR on USPS set ===")
        vblr_prediction_usps = vblr.predict(usps_test)
        print(classification_report(usps_label, vblr_prediction_usps))
        cnf_eblr_usps = metrics.confusion_matrix(usps_label, vblr_prediction_usps)
        plot_confusion_matrix(cnf_eblr_usps, normalize=False, title='Confusion matrix(EBLR) on USPS testSet')
        plt.show()


main()




