#include "StageClassifier.h"
#include <functional>
#include <algorithm>

using std::count_if;
using std::bind2nd;
using std::greater_equal;
using std::less;

void StageClassifier::SearchTheta(vector<vector<vector<double>>> X, vector<bool> y)
{
    vector<double> probs;

    for (int i = 0; i < X.size(); i++)
        probs.push_back(Predict(X[i]));

    /* search min TPR */
    double threshhold;
    for (threshhold = 0.49980; threshhold <= 0.50015; threshhold += 0.000001)
    {
        TPR = count_if(probs.begin(), probs.begin() + n_pos, bind2nd(greater_equal<double>(), threshhold)) / (double)n_pos;
        if (TPR > TPR_min)
            break;
    }

    /* get corresponding theta and FPR */
    theta = threshhold;
    FPR = count_if(probs.begin() + n_pos, probs.end(), bind2nd(less<double>(), threshhold)) / (double)n_neg;
}

double StageClassifier::Evaluate(vector<vector<vector<double>>> X, vector<bool> y)
{
    vector<double> probs;
    vector<double> TPRs, FPRs;
    double area = 0;

    for (int i = 0; i < X.size(); i++)
        probs.push_back(Predict(X[i]));

    for (double threshhold = 0.50015; threshhold >= 0.49980; threshhold -= 0.000001)
    {
        TPRs.push_back(count_if(probs.begin(), probs.begin() + n_pos, bind2nd(greater_equal<double>(), threshhold)) / (double)n_pos);
        FPRs.push_back(count_if(probs.begin() + n_pos, probs.end(), bind2nd(greater_equal<double>(), threshhold)) / (double)n_neg);

        if (FPRs.size() > 1)
            area += TPRs.back() * (FPRs.back() - FPRs[FPRs.size() - 2]); // right Riemann sum
    }

    return area;
}