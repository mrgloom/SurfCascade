#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::string;
using std::array;

int main(int argc, char *argv[])
{
    string filepath = "";

    /* extract features */
    DenseSURFFeatureExtractor dense_surf_feature_extractor;
    vector<vector<double>> features;

    cout << "Extracting features..." << endl;
    for (int i = 0; i < filepaths.size(); i++)
    {
        vector<vector<double>> features_img;
        dense_surf_feature_extractor.ExtractFeatures(filepaths[i], features_img);
        features_all.push_back(features_img);
    }

    cout << "Detecting..." << endl;
    CascadeClassifier cascade_classifier;
    cascade_classifier.Train(features_all, labels);
    cascade_classifier.Print();

    cout << "Done." << endl;

    return 0;
}