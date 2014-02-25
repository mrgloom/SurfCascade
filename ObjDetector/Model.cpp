#pragma warning( disable : 4290 ) // disable warnings from libconfig

#include "Model.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LearningAlgorithms/CascadeClassifier/StageClassifiers/GentleAdaboost.h"
#include "LearningAlgorithms/CascadeClassifier/WeakClassifiers/LogisticRegression.h"
#include "libconfig.h++"
#include "LOG.h"
#include <memory>

using std::shared_ptr;
using std::static_pointer_cast;
using libconfig::Config;
using libconfig::Setting;

Model::~Model()
{
}

int Model::Save(CascadeClassifier& cascade_classifier)
{
    Config cfg;
    Setting& root = cfg.getRoot();

    Setting& cascade_classifier_grp = root.add("cascade_classifier", Setting::TypeGroup);

    cascade_classifier_grp.add("max_stages_num", Setting::TypeInt) = cascade_classifier.max_stages_num;
    cascade_classifier_grp.add("FPR_target", Setting::TypeFloat) = cascade_classifier.FPR_target;
    cascade_classifier_grp.add("TPR_min_perstage", Setting::TypeFloat) = cascade_classifier.TPR_min_perstage;
    cascade_classifier_grp.add("FPR", Setting::TypeFloat) = cascade_classifier.FPR;
    cascade_classifier_grp.add("TPR", Setting::TypeFloat) = cascade_classifier.TPR;
    Setting& stage_classifiers_lst = cascade_classifier_grp.add("stage_classifiers", Setting::TypeList);

    for (int i = 0; i < cascade_classifier.stage_classifiers.size(); i++)
    {
        shared_ptr<StageClassifier> stage_classifier = cascade_classifier.stage_classifiers[i];
        Setting& stage_classifier_grp = stage_classifiers_lst.add(Setting::TypeGroup);

        stage_classifier_grp.add("search_step", Setting::TypeFloat) = stage_classifier->search_step;
        stage_classifier_grp.add("auc_step", Setting::TypeFloat) = stage_classifier->auc_step;
        stage_classifier_grp.add("TPR_min", Setting::TypeFloat) = stage_classifier->TPR_min;
        stage_classifier_grp.add("n_total", Setting::TypeInt) = stage_classifier->n_total;
        stage_classifier_grp.add("n_pos", Setting::TypeInt) = stage_classifier->n_pos;
        stage_classifier_grp.add("n_neg", Setting::TypeInt) = stage_classifier->n_neg;
        stage_classifier_grp.add("FPR", Setting::TypeFloat) = stage_classifier->FPR;
        stage_classifier_grp.add("TPR", Setting::TypeFloat) = stage_classifier->TPR;
        stage_classifier_grp.add("theta", Setting::TypeFloat) = stage_classifier->theta;

        shared_ptr<GentleAdaboost> gentle_adaboost = static_pointer_cast<GentleAdaboost>(stage_classifier);

        stage_classifier_grp.add("total_AUC_score", Setting::TypeFloat) = gentle_adaboost->total_AUC_score;
        stage_classifier_grp.add("sample_num", Setting::TypeInt) = gentle_adaboost->sample_num;
        stage_classifier_grp.add("max_iters", Setting::TypeInt) = gentle_adaboost->max_iters;

        Setting& weak_classifiers_lst = stage_classifier_grp.add("weak_classifiers", Setting::TypeList);

        for (int j = 0; j < gentle_adaboost->weak_classifiers.size(); j++)
        {
            shared_ptr<WeakClassifier> weak_classifier = gentle_adaboost->weak_classifiers[j];
            Setting& weak_classifier_grp = weak_classifiers_lst.add(Setting::TypeGroup);

            weak_classifier_grp.add("patch_index", Setting::TypeInt) = weak_classifier->patch_index;

            shared_ptr<LogisticRegression> logistic_regression = static_pointer_cast<LogisticRegression>(weak_classifier);

            Setting& theta_arr = weak_classifier_grp.add("theta", Setting::TypeArray);
            for (int k = 0; k < logistic_regression->theta.size(); k++)
                theta_arr.add(Setting::TypeFloat) = logistic_regression->theta[k];

            weak_classifier_grp.add("max_iters", Setting::TypeInt) = logistic_regression->max_iters;
            weak_classifier_grp.add("alpha", Setting::TypeFloat) = logistic_regression->alpha;
            weak_classifier_grp.add("epsilon", Setting::TypeFloat) = logistic_regression->epsilon;
        }
    }

    try
    {
        cfg.writeFile(model_cfg.c_str());
        LOG_INFO("New model successfully written to: " << model_cfg);
    }
    catch (libconfig::FileIOException&)
    {
        LOG_ERROR("I/O error while writing file: " << model_cfg);
        return(EXIT_FAILURE);
    }

    return(EXIT_SUCCESS);
}

int Model::Load(CascadeClassifier& cascade_classifier)
{
    Config cfg;

    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile(model_cfg.c_str());
    }
    catch (const libconfig::FileIOException&)
    {
        LOG_ERROR("I/O error while reading file.");
        return(EXIT_FAILURE);
    }
    catch (const libconfig::ParseException& pex)
    {
        LOG_ERROR("Parse error at " << pex.getFile() << ":" << pex.getLine()
            << " - " << pex.getError());
        return(EXIT_FAILURE);
    }

    Setting& root = cfg.getRoot();

    try
    {
        Setting& cascade_classifier_grp = root["cascade_classifier"];

        cascade_classifier.max_stages_num = cascade_classifier_grp["max_stages_num"];
        cascade_classifier.FPR_target = cascade_classifier_grp["FPR_target"];
        cascade_classifier.TPR_min_perstage = cascade_classifier_grp["TPR_min_perstage"];
        cascade_classifier.FPR = cascade_classifier_grp["FPR"];
        cascade_classifier.TPR = cascade_classifier_grp["TPR"];
        Setting& stage_classifiers_lst = cascade_classifier_grp["stage_classifiers"];

        for (int i = 0; i < stage_classifiers_lst.getLength(); i++)
        {
            shared_ptr<StageClassifier> stage_classifier(new GentleAdaboost(cascade_classifier.TPR_min_perstage));
            Setting& stage_classifier_grp = stage_classifiers_lst[i];

            stage_classifier->search_step = stage_classifier_grp["search_step"];
            stage_classifier->auc_step = stage_classifier_grp["auc_step"];
            stage_classifier->TPR_min = stage_classifier_grp["TPR_min"];
            stage_classifier->n_total = stage_classifier_grp["n_total"];
            stage_classifier->n_pos = stage_classifier_grp["n_pos"];
            stage_classifier->n_neg = stage_classifier_grp["n_neg"];
            stage_classifier->FPR = stage_classifier_grp["FPR"];
            stage_classifier->TPR = stage_classifier_grp["TPR"];
            stage_classifier->theta = stage_classifier_grp["theta"];

            shared_ptr<GentleAdaboost> gentle_adaboost = static_pointer_cast<GentleAdaboost>(stage_classifier);

            gentle_adaboost->total_AUC_score = stage_classifier_grp["total_AUC_score"];
            gentle_adaboost->sample_num = stage_classifier_grp["sample_num"];
            gentle_adaboost->max_iters = stage_classifier_grp["max_iters"];

            Setting& weak_classifiers_lst = stage_classifier_grp["weak_classifiers"];

            for (int j = 0; j < weak_classifiers_lst.getLength(); j++)
            {
                shared_ptr<WeakClassifier> weak_classifier(new LogisticRegression(0));
                Setting& weak_classifier_grp = weak_classifiers_lst[j];

                weak_classifier->patch_index = weak_classifier_grp["patch_index"];

                shared_ptr<LogisticRegression> logistic_regression = static_pointer_cast<LogisticRegression>(weak_classifier);

                Setting& theta_arr = weak_classifier_grp["theta"];
                for (int k = 0; k < theta_arr.getLength(); k++)
                    logistic_regression->theta.push_back(theta_arr[k]);

                logistic_regression->max_iters = weak_classifier_grp["max_iters"];
                logistic_regression->alpha = weak_classifier_grp["alpha"];
                logistic_regression->epsilon = weak_classifier_grp["epsilon"];

                gentle_adaboost->weak_classifiers.push_back(weak_classifier);
            }

            cascade_classifier.stage_classifiers.push_back(stage_classifier);
        }
    }
    catch (const libconfig::SettingNotFoundException&)
    {
        // Ignore
    }

    return(EXIT_SUCCESS);
}