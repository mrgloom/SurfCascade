#ifndef GENTLEADABOOST_H
#define GENTLEADABOOST_H

#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/StageClassifier.h"
#include <vector>

using std::vector;

class GentleAdaboost : public StageClassifier
{
    double total_AUC_score = 0;
    int sample_num = 960; // 30 * 32
    int max_iters = 100;
    vector<shared_ptr<WeakClassifier>> weak_classifiers;

public:
    GentleAdaboost(double TPR_min_perstage): StageClassifier(TPR_min_perstage) {}
    void Train(vector<vector<vector<double>>>& X, vector<bool>& y);
    double Predict(vector<vector<double>>& x);
    double Predict2(vector<vector<double>>& x);
    void GetFittedPatchIndexes(vector<int>& patch_indexes);
    friend class Model;
};

#endif