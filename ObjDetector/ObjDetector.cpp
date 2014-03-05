#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include "Model.h"
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::ifstream;
using std::string;
using std::array;

int get_filepaths(string folder, string wildcard, vector<string>& filepaths);

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: ObjDetector [--train|--detect]" << endl;
        return 0;
    }

    Model model("model.cfg");

    /************************************************************************/
    /*                             Train Mode                               */
    /************************************************************************/
    if (strcmp(argv[1], "--train") == 0 || strcmp(argv[1], "-t") == 0)
    {
        string prefix_path = "D:/FaceData/Custom/";
        string pos_file(prefix_path + "pos.list");
        string neg_file(prefix_path + "neg.list");

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> patches;

        cout << "Extracting patches..." << endl;
        dense_surf_feature_extractor.LoadFileList(pos_file, prefix_path);
        dense_surf_feature_extractor.ExtractPatches(patches);

        /* extract features in positive samples */
        vector<vector<vector<double>>> features_all;
        vector<vector<double>> features_img;

        cout << "Extracting features in positive samples..." << endl;
        while (dense_surf_feature_extractor.ExtractNextImageFeatures(patches, features_img))
            features_all.push_back(features_img);

        vector<bool> labels(features_all.size(), true);

        /* train cascade classifier */
        cout << "Training cascade classifier..." << endl;
        CascadeClassifier cascade_classifier(dense_surf_feature_extractor);
        cascade_classifier.Train(features_all, labels, neg_file, patches);
        cascade_classifier.Print();

        /* saving model to model.cfg... */
        model.Save(cascade_classifier);

        cout << "Done." << endl;
    }

    /************************************************************************/
    /*                             Detect Mode                              */
    /************************************************************************/
    else if (strcmp(argv[1], "--detect") == 0 || strcmp(argv[1], "-d") == 0)
    {
        string filepath = "D:/facedata/Detect/1974.jpg";

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> dense_patches;

        cout << "Extracting patches..." << endl;
        dense_surf_feature_extractor.win = Rect(0, 0, 60, 90); // TODO
        dense_surf_feature_extractor.ExtractPatches(dense_patches);

        /* get fitted patches */
        CascadeClassifier cascade_classifier(dense_surf_feature_extractor);
        model.Load(cascade_classifier);

        vector<vector<int>> patch_indexes;
        vector<vector<Rect>> fitted_patches;

        cout << "Getting fitted patches..." << endl;
        cascade_classifier.GetFittedPatchIndexes(patch_indexes);
        for (int i = 0; i < patch_indexes.size(); i++)
        {
            vector<Rect> patches_perstage;
            for (int j = 0; j < patch_indexes[i].size(); j++)
                patches_perstage.push_back(dense_patches[patch_indexes[i][j]]);
            fitted_patches.push_back(patches_perstage);
        }

        /* calculate integral image */
        vector<vector<Rect>> patches(fitted_patches);

        cout << "Calculating integral image..." << endl;
        Mat img = imread(filepath, cv::IMREAD_GRAYSCALE);
        dense_surf_feature_extractor.IntegralImage(img);

        /* prepare showing image */
        Mat img_rgb(img.size(), CV_8UC3);
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);
        //cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);

        /* scan with varying windows */
        cout << "Scanning with varying windows..." << endl;
        for (Rect win(0, 0, 60, 90); win.width <= img.size().width && win.height <= img.size().height; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
        {
            for (win.y = 0; win.y + win.height <= img.size().height; win.y += 10)
                for (win.x = 0; win.x + win.width <= img.size().width; win.x += 10)
                {
                    dense_surf_feature_extractor.ProjectPatches(Rect(0, 0, 60, 90), win, fitted_patches, patches); //TODO

                    vector<vector<vector<double>>> features_win;

                    dense_surf_feature_extractor.ExtractFeatures(patches, features_win);

                    if (cascade_classifier.Predict2(features_win))
                    {
                        //cout << "Detected: " << win << endl;
                        rectangle(img_rgb, win, cv::Scalar(255, 0, 0), 1);
                    }

                    //Mat img_tmp = img_rgb.clone();
                    //rectangle(img_tmp, win, cv::Scalar(0, 255, 255), 1);
                    //cv::imshow("Result", img_tmp);
                    //cv::waitKey(100);
                }
        }
        cout << "Over.";

        //cv::destroyWindow("Result");
        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Result", img_rgb);
        cv::waitKey(0);
    }

    return 0;
}

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