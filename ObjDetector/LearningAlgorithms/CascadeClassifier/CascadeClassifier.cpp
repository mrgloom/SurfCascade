#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/GentleAdaboost.h"
#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LOG.h"
#include <cassert>

using std::ostream;
using std::endl;

/* parameter X: only positive samples(files) */
void CascadeClassifier::Train(vector<vector<vector<double>>>& X, vector<bool>& y, DenseSURFFeatureExtractor& dense_surf_feature_extractor, string neg_file, const vector<Rect>& patches)
{
    assert(X.size() == y.size());

    int n_pos = (int)y.size();
    int n_neg = n_pos;
    int n_total = n_pos * 2;

    FPR = 1.0;
    TPR = 1.0;

    vector<bool> y_neg(n_neg, false);
    y.insert(y.end(), y_neg.begin(), y_neg.end());

    dense_surf_feature_extractor.LoadFileList(neg_file, dense_surf_feature_extractor.prefix_path);

    LOG_INFO("\tCascade stages begin");
    for (int i = 0; i < max_stages_num && FPR > FPR_target; i++)
    {
        LOG_INFO("\tCascade stage " << i);

        /* renew negative samples */
        LOG_INFO("\tRenew negative samples...");

        X.erase(X.begin() + n_pos, X.end());

        vector<vector<double>> features_img;

        while (X.size() < n_total)
        {
            if (!dense_surf_feature_extractor.ExtractNextImageFeatures(patches, features_img))
                break;
            if (Predict(features_img) == true) // if false positive
                X.push_back(features_img);
        }
        if (X.size() != n_total)
        {
            LOG_WARNING("\tRunning out of negative samples.");
            break;
        }

        shared_ptr<StageClassifier> stage_classifier(new GentleAdaboost(TPR_min_perstage));

        LOG_INFO("\tTraining stage classifier");
        stage_classifier->Train(X, y);
        
        /* search ROC curve */
        stage_classifier->SearchTheta(X, y);
        LOG_INFO("\tStage classifier FPR = " << stage_classifier->FPR << ", TPR = " << stage_classifier->TPR);

        FPR *= stage_classifier->FPR;
        TPR *= stage_classifier->TPR;

        stage_classifiers.push_back(stage_classifier);
    }
    LOG_INFO("\tCascade stages end");
}

bool CascadeClassifier::Predict(vector<vector<double>>& x)
{
    for (int i = 0; i < stage_classifiers.size(); i++)
    {
        if (stage_classifiers[i]->Predict(x) < stage_classifiers[i]->theta)
            return false;
    }

    return true;
}

bool CascadeClassifier::Predict2(vector<vector<vector<double>>>& x)
{
    for (int i = 0; i < stage_classifiers.size(); i++)
    {
        if (stage_classifiers[i]->Predict2(x[i]) < stage_classifiers[i]->theta)
            return false;
    }

    return true;
}

void CascadeClassifier::GetFittedPatchIndexes(vector<vector<int>>& patch_indexes)
{
    for (int i = 0; i < stage_classifiers.size(); i++)
    {
        vector<int> patch_indexes_perstage;
        stage_classifiers[i]->GetFittedPatchIndexes(patch_indexes_perstage);
        patch_indexes.push_back(patch_indexes_perstage);
    }
}

void CascadeClassifier::Print()
{
    cout << "FPR: " << FPR << ", TPR:" << TPR << endl;
}
