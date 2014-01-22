#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

#include "StageClassifiers/GentleAdaboost.h"
#include "StageClassifiers/StageClassifier.h"
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
    int max_stages_num;
    double FPR_target;
    double TPR_min_perstage;

public:
    double FPR;
    double TPR;

    void Train(vector<vector<vector<double>>> X, vector<bool> y);
    bool Predict(vector<vector<double>> x);
    void Print();
};

#endif