#ifndef STAGECLASSIFIER_H
#define STAGECLASSIFIER_H

#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

class StageClassifier
{
    float search_step = 0.01f;
    float auc_step = 0.05f;
    float TPR_min;

protected:
    int n_total;
    int n_pos;
    int n_neg;

public:
    float FPR;
    float TPR;
    float theta;

    StageClassifier(float TPR_min_perstage): TPR_min(TPR_min_perstage) {}
    virtual void Train(vector<vector<vector<float>>>& X, vector<bool>& y) = 0;
    virtual float Predict(vector<vector<float>>& x) = 0;
    virtual float Predict2(vector<vector<float>>& x) = 0;
    virtual void GetFittedPatchIndexes(vector<int>& patch_indexes) = 0;
    float Evaluate(vector<vector<vector<float>>>& X, vector<bool>& y);
    void SearchTheta(vector<vector<vector<float>>>& X, vector<bool>& y);
    friend class Model;
};

#endif