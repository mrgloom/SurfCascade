#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/StageClassifier.h"
#include "LOG.h"
#include <functional>
#include <algorithm>

using std::count_if;
using std::bind2nd;
using std::greater_equal;

void StageClassifier::SearchTheta(vector<vector<vector<float>>>& X, vector<bool>& y)
{
    int whole_n_total = (int)y.size();
    int whole_n_pos = (int)count(y.begin(), y.end(), true);
    int whole_n_neg = (int)(whole_n_total - whole_n_pos);

    vector<float> probs;

    for (int i = 0; i < X.size(); i++)
        probs.push_back(Predict(X[i]));

    /* search min TPR */
    float threshhold;
    for (threshhold = 1; threshhold >= 0; threshhold -= search_step)
    {
        TPR = count_if(probs.begin(), probs.begin() + whole_n_pos, bind2nd(greater_equal<float>(), threshhold)) / (float)whole_n_pos;
        if (TPR >= TPR_min)
            break;
    }

    /* get corresponding theta and FPR */
    theta = threshhold;
    FPR = count_if(probs.begin() + whole_n_pos, probs.end(), bind2nd(greater_equal<float>(), threshhold)) / (float)whole_n_neg;
}

float StageClassifier::Evaluate(vector<vector<vector<float>>>& X, vector<bool>& y)
{
    vector<float> probs;
    vector<float> TPRs, FPRs;
    float area = 0;

    for (int i = 0; i < X.size(); i++)
        probs.push_back(Predict(X[i]));

    /* test on training set (0.5 threshhold??) */
    //#if SETLEVEL == DEBUG_LEVEL
    //int TP = 0, TN = 0;
    //for (int i = 0; i < X.size(); i++)
    //{
    //    if ((probs[i] >= 0.5) && y[i] == true)
    //        TP++;
    //    else if ((probs[i] < 0.5) && y[i] == false)
    //        TN++;
    //}
    //LOG_DEBUG_NN("\t\tStrong classifier: ");
    //LOG_DEBUG_NN("TP = " << TP << '/' << y.size() / 2 << ", TN = " << TN << '/' << y.size() / 2);
    //LOG_DEBUG_NN(", Result: " << (float)(TP + TN) / y.size());
    //#endif

    for (float threshhold = 1; threshhold >= 0; threshhold -= auc_step)
    {
        TPRs.push_back(count_if(probs.begin(), probs.begin() + n_pos, bind2nd(greater_equal<float>(), threshhold)) / (float)n_pos);
        FPRs.push_back(count_if(probs.begin() + n_pos, probs.end(), bind2nd(greater_equal<float>(), threshhold)) / (float)n_neg);

        if (FPRs.size() > 1)
            area += (TPRs.back() + TPRs[TPRs.size() - 2]) * (FPRs.back() - FPRs[FPRs.size() - 2]) / 2; // trapezoid
            //area += TPRs[TPRs.size() - 2] * (FPRs.back() - FPRs[FPRs.size() - 2]); // left riemann sum
            //area += TPRs.back() * (FPRs.back() - FPRs[FPRs.size() - 2]); // right Riemann sum
    }

    return area;
}