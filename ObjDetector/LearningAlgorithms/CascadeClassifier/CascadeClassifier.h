#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/GentleAdaboost.h"
#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/StageClassifier.h"
#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include <vector>
#include <array>
#include <memory>

using std::vector;
using std::array;
using std::shared_ptr;
using std::ostream;

class DenseSURFFeatureExtractor;

class CascadeClassifier
{
    int max_stages_num = 10;
    float FPR_target = 1e-6f;
    float TPR_min_perstage = 0.995f;

public:
    float FPR;
    float TPR;
    vector<shared_ptr<StageClassifier>> stage_classifiers;

    void Train(vector<vector<vector<float>>>& X, vector<bool>& y, DenseSURFFeatureExtractor& dense_surf_feature_extractor, string neg_file, const vector<Rect>& patches);
    bool Predict(vector<vector<float>>& x);
    bool Predict2(vector<vector<vector<float>>>& x, double& score);
    void GetFittedPatchIndexes(vector<vector<int>>& patch_indexes);
    void Print();
    friend class Model;
};

#endif