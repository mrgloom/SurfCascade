#include "GentleAdaboost.h"
#include "../WeakClassifiers/LogisticRegression.h"
#include "../../../LOG.h"
#include <algorithm>
#include <cassert>

using std::count;
using std::sort;

void GentleAdaboost::Train(vector<vector<vector<double>>> X, vector<bool> y)
{
    assert(X.size() == y.size());

    n_total = (int)y.size();
    n_pos = (int)count(y.begin(), y.end(), true);
    n_neg = (int)(n_total - n_pos);

    assert(n_pos == n_neg);

    /* normalize initial weights */
    vector<double> weights(n_pos, 1.0 / n_pos);
    vector<double> weights_neg(n_neg, 1.0 / n_neg);
    weights.insert(weights.end(), weights_neg.begin(), weights_neg.end());

    /* boosting */
    LOG_INFO("\t\tBoosting begin");
    for (int t = 0; t < max_iters; t++)
    {
        LOG_INFO("\t\tBoosting round " << t);

        /* sort sample by weights */
        vector<int> index(weights.size());
        for (int i = 0; i < index.size(); i++)
            index[i] = i;

        sort(index.begin(), index.begin() + n_pos, [&weights](int a, int b) {return weights[a] > weights[b]; });
        sort(index.begin() + n_pos, index.end(), [&weights](int a, int b) {return weights[a] > weights[b]; });
        
        /* parallel for each patch, train a weak classifier */
        shared_ptr<WeakClassifier> best_weak_classifier;
        double curr_AUC_score;
        double best_AUC_score = 0;

        int patches_num = (int)X[0].size();
        for (int k = 0; k < patches_num; k++)
        {
            /* get weak classifier's training set */
            vector<vector<double>> samples_X;
            vector<bool> samples_y;
            vector<double> feature;

            assert(sample_num <= n_total);
            for (int i = 0; i < sample_num; i++)
            {
                feature = X[index[i]][k];
                samples_X.push_back(feature);
                samples_y.push_back(y[index[i]]);

                feature = X[index[i + n_pos]][k];
                samples_X.push_back(feature);
                samples_y.push_back(y[index[i + n_pos]]);
            }

            /* train weak classifier */
            LOG_INFO_NN("\r\t\tTraining logistic regression " << k << '/' << patches_num << flush);
            shared_ptr<WeakClassifier> weak_classifier(new LogisticRegression(k));

            weak_classifier->Train(samples_X, samples_y);

            /* evaluate on the whole training set to obtain the AUC score */
            weak_classifiers.push_back(weak_classifier);
            curr_AUC_score = Evaluate(X, y);
            weak_classifiers.pop_back();

            /* iterate for best weak classifier */
            if (curr_AUC_score > best_AUC_score)
            {
                best_AUC_score = curr_AUC_score;
                best_weak_classifier = weak_classifier;
            }
        }
        LOG_INFO_NN(endl);

        LOG_INFO("\t\tBest AUC score: " << best_AUC_score);

        weak_classifiers.push_back(best_weak_classifier);

        /* if AUC score is converged, break the loop */
        if (best_AUC_score - total_AUC_score < 0.001)
        {
            total_AUC_score = best_AUC_score;
            break;
        }
        else
            total_AUC_score = best_AUC_score;

        /* update samples weights */
        vector<double> old_weights(weights);
        double sum_weights_pos = 0;
        double sum_weights_neg = 0;

        for (int i = 0; i < n_total; i++)
        {
            weights[i] *= exp(-(int)y[i] * best_weak_classifier->Predict(X[i][best_weak_classifier->patch_index]));
            if (i < n_pos)
                sum_weights_pos += weights[i];
            else
                sum_weights_neg += weights[i];
        }

        /* normalize weights */
        for (int i = 0; i < n_total; i++)
        {
            if (i < n_pos)
                weights[i] /= sum_weights_pos;
            else
                weights[i] /= sum_weights_neg;
        }
    }
    LOG_INFO("\t\tBoosting end");

    /* backward removing */
    while (true)
    {
        break;
    }
}

double GentleAdaboost::Predict(vector<vector<double>> x)
{
    double sum = 0, prob;

    for (int i = 0; i < weak_classifiers.size(); i++)
    {
        sum += weak_classifiers[i]->Predict(x[weak_classifiers[i]->patch_index]);
    }

    prob = sum / weak_classifiers.size();

    return prob;
}