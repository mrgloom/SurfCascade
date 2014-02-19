#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::string;
using std::array;

int get_filepaths(string folder, string wildcard, vector<string>& filepaths)
{
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    int i = 0;

    hFind = FindFirstFile((folder + wildcard).c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE) return i;
    do {
        filepaths.push_back(folder + string(ffd.cFileName));
        i++;
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);

    return i;
}

int main(int argc, char *argv[])
{
    string pos_folder = "D:/facedata/train/face/";
    string neg_folder = "D:/facedata/train/non-face/";
    string wildcard = string("*.pgm");

    /* get file names and labels */
    vector<string> filepaths;
    vector<bool> labels;
    int n;

    n = get_filepaths(pos_folder, wildcard, filepaths);
    labels.assign(n, true);

    n = get_filepaths(neg_folder, wildcard, filepaths);
    vector<bool> neg_labels(n, false);
    labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());

    /* extract features */
    DenseSURFFeatureExtractor dense_surf_feature_extractor;
    vector<vector<vector<double>>> features_all;

    cout << "Extracting features..." << endl;
    for (int i = 0; i < filepaths.size(); i++)
    {
        vector<vector<double>> features_img;
        dense_surf_feature_extractor.ExtractFeatures(filepaths[i], features_img);
        features_all.push_back(features_img);
    }

    cout << "Training cascade classifier..." << endl;
    /* train cascade classifier */
    CascadeClassifier cascade_classifier;
    cascade_classifier.Train(features_all, labels);
    cascade_classifier.Print();

    cout << "Done." << endl;

    return 0;
}
