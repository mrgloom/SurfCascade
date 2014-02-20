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
    string filepath = "D:/facedata/test.jpg";

    /* calculate integral image */
    DenseSURFFeatureExtractor dense_surf_feature_extractor;
    Mat sums[DenseSURFFeatureExtractor::n_bins];

    dense_surf_feature_extractor.IntegralImage(filepath, sums);

    Size img_size(sums[0].cols - 1, sums[0].rows - 1);

    /* scan with variant windows */
    CascadeClassifier cascade_classifier;
    // TODO: read trained classifier

    for (Rect win(0, 0, 10, 10); win.width <= img_size.width && win.height <= img_size.height; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
    {
        for (win.y = 0; win.y + win.height <= img_size.height; win.y+=2)
            for (win.x = 0; win.x + win.width <= img_size.width; win.x+=2)
            {
                vector<vector<double>> features_win;

                dense_surf_feature_extractor.ExtractFeatures(sums, win, features_win);
                //cascade_classifier.Predict(features_win);

                cout << win;
            }
    }

    return 0;
}