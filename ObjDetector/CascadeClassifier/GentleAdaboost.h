#ifndef GENTLEADABOOST_H
#define GENTLEADABOOST_H

#include "CascadeClassifier/StageClassifier.h"
#include <vector>

using std::vector;

class LogisticRegression;

class GentleAdaboost : public StageClassifier
{
    float total_AUC_score = 0;
    int sample_num = 960; // 30 * 32
    int max_iters = 100;
    vector<shared_ptr<LogisticRegression>> weak_classifiers;

public:
    GentleAdaboost(float TPR_min_perstage): StageClassifier(TPR_min_perstage) {}
    void Train(vector<vector<vector<float>>>& X, vector<bool>& y);
    float Predict(vector<vector<float>>& x);
    float Predict2(vector<vector<float>>& x);
    void GetFittedPatchIndexes(vector<int>& patch_indexes);
    friend class Model;
};

#endif