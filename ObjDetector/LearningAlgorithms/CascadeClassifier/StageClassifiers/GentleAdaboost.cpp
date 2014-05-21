#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/GentleAdaboost.h"
#include "LearningAlgorithms/CascadeClassifier/WeakClassifiers/LogisticRegression.h"
#include "linear.h"
#include "LOG.h"
#include <algorithm>
#include <cassert>

using std::count;
using std::sort;

void GentleAdaboost::Train(vector<vector<vector<float>>>& X, vector<bool>& y)
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
        
        /* parallel for each patch, train a weak classifier */
        shared_ptr<LogisticRegression> best_weak_classifier;
        float curr_AUC_score;
        float best_AUC_score = 0;

        int patches_num = (int)X[0].size();
        for (int k = 0; k < patches_num; k++)
        {
            /* get weak classifier's training set */
            vector<float> feature;

            problem* prob = new problem;
            prob->l = sample_num * 2;
            prob->bias = 1;
            prob->n = 32 + (prob->bias != 0);
            prob->y = new double[sample_num * 2];
            prob->x = new feature_node* [sample_num * 2];

            assert(sample_num <= n_total);
            for (int i = 0; i < sample_num; i++)
            {
                // roulette wheel algorithm
                double r;
                double s;
                int n;

                while (1) {
                    r = (double)rand() / RAND_MAX;
                    s = 0;
                    for (n = 0; n < n_pos; n++) {
                        s += weights[n];
                        if (r <= s)
                            break;
                    }
                    if (n != n_pos)
                        break;
                }

                prob->x[i] = new feature_node[32 + 1 + (prob->bias != 0)];
                feature = X[n][k];
                int j;
                for (j = 0; j < 32; j++)
                {
                    prob->x[i][j].index = j + 1;
                    prob->x[i][j].value = feature[j];
                }
                if (prob->bias != 0)
                {
                    prob->x[i][j].index = j + 1;
                    prob->x[i][j].value = prob->bias;
                    ++j;
                }
                prob->x[i][j].index = -1;
                prob->y[i] = 1;

                // roulette wheel algorithm
                while (1) {
                    r = (double)rand() / RAND_MAX;
                    s = 0;
                    for (n = n_pos; n < n_total; n++) {
                        s += weights[n];
                        if (r <= s)
                            break;
                    }
                    if (n != n_total)
                        break;
                }

                prob->x[i + sample_num] = new feature_node[32 + 1 + (prob->bias != 0)];
                feature = X[n][k];
                for (j = 0; j < 32; j++)
                {
                    prob->x[i + sample_num][j].index = j + 1;
                    prob->x[i + sample_num][j].value = feature[j];
                }
                if (prob->bias != 0)
                {
                    prob->x[i + sample_num][j].index = j + 1;
                    prob->x[i + sample_num][j].value = prob->bias;
                    ++j;
                }
                prob->x[i + sample_num][j].index = -1;
                prob->y[i + sample_num] = -1;
            }

            /* train weak classifier */
            shared_ptr<LogisticRegression> weak_classifier(new LogisticRegression(k));

            if (SETLEVEL == DEBUG_LEVEL)
                LOG_INFO("\t\tTraining logistic regression " << k + 1 << '/' << patches_num);
            else
                LOG_INFO_NN("\r\t\tTraining logistic regression " << k + 1 << '/' << patches_num << flush);

            weak_classifier->Train(prob);

            ///* test on training set */
            //#if SETLEVEL == DEBUG_LEVEL
            //int TP = 0, TN = 0;
            //float prob;
            //for (int i = 0; i < samples_X.size(); i++)
            //{
            //    prob = weak_classifier->Predict(samples_X[i]);
            //    if ((prob >= 0.5) && samples_y[i] == true)
            //        TP++;
            //    else if ((prob < 0.5) && samples_y[i] == false)
            //        TN++;
            //}
            //LOG_DEBUG_NN("\t\tWeak classifier: ");
            //LOG_DEBUG_NN("TP = " << TP << '/' << samples_y.size() / 2 << ", TN = " << TN << '/' << samples_y.size() / 2);
            //LOG_DEBUG(", Result: " << (float)(TP + TN) / samples_y.size());
            //#endif

            /* evaluate on the whole training set to obtain the AUC score */
            weak_classifiers.push_back(weak_classifier);
            curr_AUC_score = Evaluate(X, y);
            LOG_DEBUG(", AUC score = " << curr_AUC_score);
            weak_classifiers.pop_back();

            /* iterate for best weak classifier */
            if (curr_AUC_score > best_AUC_score)
            {
                best_AUC_score = curr_AUC_score;
                best_weak_classifier = weak_classifier;
            }

            /* clean up */
            for (int i = 0; i < sample_num * 2; i++)
                delete[] prob->x[i];
            delete[] prob->y;
            delete prob;

            LOG_DEBUG_NN(endl); // empty line between weak classifiers' debug output
        }

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

float GentleAdaboost::Predict(vector<vector<float>>& x)
{
    float sum = 0, prob;

    for (int i = 0; i < weak_classifiers.size(); i++)
    {
        sum += weak_classifiers[i]->Predict(x[weak_classifiers[i]->patch_index]);
    }

    prob = sum / weak_classifiers.size();

    return prob;
}

float GentleAdaboost::Predict2(vector<vector<float>>& x)
{
    assert(weak_classifiers.size() == x.size());

    float sum = 0, prob;

    for (int i = 0; i < weak_classifiers.size(); i++)
    {
        sum += weak_classifiers[i]->Predict(x[i]);
    }

    prob = sum / weak_classifiers.size();

    return prob;
}

void GentleAdaboost::GetFittedPatchIndexes(vector<int>& patch_indexes)
{
    for (int i = 0; i < weak_classifiers.size(); i++)
        patch_indexes.push_back(weak_classifiers[i]->patch_index);
}
