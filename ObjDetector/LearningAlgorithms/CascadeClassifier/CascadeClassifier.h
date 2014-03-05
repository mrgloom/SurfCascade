#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/GentleAdaboost.h"
#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/StageClassifier.h"
#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "Model.h"
#include <vector>
#include <array>
#include <memory>

using std::vector;
using std::array;
using std::shared_ptr;
using std::ostream;

class CascadeClassifier
{
    vector<shared_ptr<StageClassifier>> stage_classifiers;
    int max_stages_num = 10;
    double FPR_target = 0.001;
    double TPR_min_perstage = 0.99;
    DenseSURFFeatureExtractor& dense_surf_feature_extractor;

public:
    double FPR;
    double TPR;

    CascadeClassifier(DenseSURFFeatureExtractor& dense_surf_feature_extractor) : dense_surf_feature_extractor(dense_surf_feature_extractor) {};
    void Train(vector<vector<vector<double>>>& X, vector<bool>& y, string neg_file, const vector<Rect>& patches);
    bool Predict(vector<vector<double>>& x);
    bool Predict2(vector<vector<vector<double>>>& x);
    void GetFittedPatchIndexes(vector<vector<int>>& patch_indexes);
    void Print();
    friend class Model;
};

#endif