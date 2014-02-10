#include "CascadeClassifier.h"
#include "StageClassifiers/GentleAdaboost.h"
#include "../../LOG.h"
#include <cassert>

using std::ostream;
using std::endl;

void CascadeClassifier::Train(vector<vector<vector<double>>> X, vector<bool> y)
{
    assert(X.size() == y.size());

    int n_total = (int)y.size();
    int n_pos = (int)count(y.begin(), y.end(), true);
    int n_neg = (int)(n_total - n_pos);

    assert(n_pos < n_neg);

    vector<vector<vector<double>>> samples_X(X.begin(), X.begin() + 2 * n_pos);
    vector<bool> samples_y(y.begin(), y.begin() + samples_X.size());

    FPR = 1.0;
    TPR = 1.0;

    int j = n_pos * 2;

    LOG_INFO("cascade stages begin");
    for (int i = 0; i < max_stages_num && FPR > FPR_target; i++)
    {
        LOG_INFO("cascade stage " << i);

        shared_ptr<StageClassifier> stage_classifier(new GentleAdaboost(TPR_min_perstage));

        stage_classifier->Train(samples_X, samples_y);
        
        /* search ROC curve */
        stage_classifier->SearchTheta(samples_X, samples_y);
        FPR *= stage_classifier->FPR;
        TPR *= stage_classifier->TPR;

        stage_classifiers.push_back(stage_classifier);

        /* renew samples */
        samples_X.erase(samples_X.begin() + n_pos, samples_X.end());
        samples_y.erase(samples_y.begin() + n_pos, samples_y.end());

        for (; j < n_total; j++)
        {
            if (Predict(X[j]) != y[j])
            {
                samples_X.push_back(X[j]);
                samples_y.push_back(y[j]);
            }

            if (samples_y.size() == n_pos * 2)
                break;
        }
        if (samples_y.size() != n_pos * 2)
        {
            LOG_ERROR("Negative samples too few.");
            return;
        }
    }
    LOG_INFO("cascade stages end");
}

bool CascadeClassifier::Predict(vector<vector<double>> x)
{
    for (int i = 0; i < stage_classifiers.size(); i++)
    {
        if (stage_classifiers[i]->Predict(x) < stage_classifiers[i]->theta)
            return false;
    }

    return true;
}

void CascadeClassifier::Print()
{
    cout << "FPR: " << FPR << ", TPR:" << TPR << endl;
}