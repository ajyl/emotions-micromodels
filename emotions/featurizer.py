"""
Encoder Module
"""
from typing import List

import os
import numpy as np
import pickle
from nltk import tokenize
from tqdm import tqdm
from interpret.glassbox import ExplainableBoostingClassifier

from emotions.BertFeaturizer import BertFeaturizer
from emotions.config import add_ed_emotions, EMP_CONFIGS, EMP_TASKS, EMP_MMS
from emotions.data_utils import load_ed_data, load_emp_data, reformat_emp_data
from emotions.seeds.custom_emotions import ED_SEEDS
from emotions.seeds.miti_codes import MITI_SEEDS


MM_HOME = os.environ.get("MM_HOME")
MODELS_DIR = os.path.join(MM_HOME, "emotions/models")

EMP_THRESHOLD = 0.6


def _format_clf_results(probs, classes):
    """
    Format results to return sorted predictions and corresponding probabilities
    """
    sorted_probs = np.argsort(probs, axis=1)
    sorted_idxs = [x[::-1] for x in sorted_probs]
    sorted_preds = [classes[x] for x in sorted_idxs]
    pred_probs = [prob[sorted_idxs[idx]] for idx, prob in enumerate(probs)]
    return list(zip(sorted_preds, pred_probs))


def _featurize_emp(featurizer, emp_data):
    """
    Featurize data specifically for EMP
    """
    reformatted_emp_data, query_idxs = reformat_emp_data(emp_data)
    bert_results = featurizer.run_bert(reformatted_emp_data)
    for utt_idx in bert_results.keys():
        _query_idx = query_idxs[utt_idx]
        bert_results[utt_idx]["labels"] = {
            task: {
                "level": emp_data[_query_idx][task]["level"],
                "rationales": emp_data[_query_idx][task]["rationales"],
            }
            for task in EMP_TASKS
        }

    labels = []
    results_by_query = {}
    for utt_idx, data_obj in bert_results.items():
        query_idx = query_idxs[int(utt_idx)]

        if query_idx not in results_by_query:
            _labels = data_obj["labels"]
            results_by_query[query_idx] = {
                "results": [],
                "labels": _labels,
            }
            labels.append(
                [
                    1 if _labels[task]["level"] != "0" else 0
                    for task in EMP_TASKS
                ]
            )

        results_by_query[query_idx]["results"].append(data_obj["results"])

    featurized = None
    for query_idx, result_obj in results_by_query.items():
        _results = result_obj["results"]
        _featurized = []
        for mm in EMP_MMS:
            _scores = [_result[mm]["max_score"] for _result in _results]
            binary = [1 if x > EMP_THRESHOLD else 0 for x in _scores]
            ratio = sum(binary) / len(_results)
            _featurized.append(ratio)
            _featurized.append(max(_scores))

        if featurized is None:
            featurized = np.array(_featurized)
        else:
            featurized = np.vstack([featurized, _featurized])

    if len(featurized.shape) < 2:
        featurized = np.resize(featurized, (1, featurized.size))
    return featurized, labels


def _featurize_emp_raw(featurizer, queries: List[str]):
    data = [
        {
            "query": query,
            "response_tokenized": tokenize.sent_tokenize(query),
            EMP_TASKS[0]: {"level": None, "rationales": None},
            EMP_TASKS[1]: {"level": None, "rationales": None},
            EMP_TASKS[2]: {"level": None, "rationales": None},
        }
        for query in queries
    ]
    featurized, _ = _featurize_emp(featurizer, data)
    return featurized


def setup_emp_config(mm_data):
    """set up orchestrator for featurization"""
    mm_configs = EMP_CONFIGS
    all_rationales = {
        "emotional_reactions": {
            "1": [],
            "2": [],
        },
        "interpretations": {
            "1": [],
            "2": [],
        },
        "explorations": {
            "1": [],
            "2": [],
        },
    }
    for instance in mm_data:
        for task in EMP_TASKS:
            level = instance[task]["level"]
            if level != "0":
                rationales = instance[task]["rationales"].split("|")
                rationales = [x for x in rationales if x != ""]
                all_rationales[task][level].extend(rationales)
    for config in mm_configs:
        config["setup_args"]["infer_config"] = {
            "segment_config": {"window_size": 10, "step_size": 4}
        }
        if config["name"] == "empathy_interpretations_1":
            config["setup_args"]["seed"] = all_rationales["interpretations"][
                "1"
            ]
        if config["name"] == "empathy_interpretations_2":
            config["setup_args"]["seed"] = all_rationales["interpretations"][
                "2"
            ]
        if config["name"] == "empathy_explorations_1":
            config["setup_args"]["seed"] = all_rationales["explorations"]["1"]
        if config["name"] == "empathy_explorations_2":
            config["setup_args"]["seed"] = all_rationales["explorations"]["2"]
        if config["name"] == "empathy_emotional_reactions_1":
            config["setup_args"]["seed"] = all_rationales[
                "emotional_reactions"
            ]["1"]
        if config["name"] == "empathy_emotional_reactions_2":
            config["setup_args"]["seed"] = all_rationales[
                "emotional_reactions"
            ]["2"]
    return mm_configs


class Encoder:
    def __init__(self, **kwargs):
        """Load micromodels for encoding convo"""
        self.mm_configs = []
        self.ed_model = None
        self.emp_model_er = None
        self.emp_model_exp = None
        self.emp_model_int = None
        print("Adding Empathetic Dialogue Micromodels...")
        self._add_ed_mms()
        print("Adding Epitome Micromodels...")
        self._add_epitome_mms()
        print("Adding MITI Micromodels...")
        self._add_miti_mms()
        print("Initializing Featurizer...")
        self._init_featurizer()

        ed_clf_path = kwargs.get("ed_classifier", None)
        emp_er_clf_path = kwargs.get("emp_er_classifier", None)
        emp_exp_clf_path = kwargs.get("emp_exp_classifier", None)
        emp_int_clf_path = kwargs.get("emp_int_classifier", None)

        if ed_clf_path is not None:
            self.ed_model = self.load_clf(ed_clf_path)

        if emp_er_clf_path is not None:
            self.emp_model_er = self.load_clf(emp_er_clf_path)

        if emp_exp_clf_path is not None:
            self.emp_model_exp = self.load_clf(emp_exp_clf_path)

        if emp_int_clf_path is not None:
            self.emp_model_int = self.load_clf(emp_int_clf_path)

    def _init_featurizer(self):
        self.featurizer = BertFeaturizer(MM_HOME, self.mm_configs)
        self.mms = self.featurizer.list_micromodels
        self.ed_mms = [
            mm
            for mm in self.mms
            if mm.startswith("emotion_") or mm.startswith("custom_")
        ]
        self.emp_mms = [mm for mm in self.mms if mm.startswith("empathy_")]

    def _add_ed_mms(self):
        """Add micromodels for EmpatheticDialogue"""
        for emotion, seed in ED_SEEDS.items():
            if len(seed) < 1:
                continue
            config = {
                "name": "custom_%s" % emotion,
                "model_type": "bert_query",
                "model_path": os.path.join(
                    MM_HOME, "models/custom_%s" % emotion
                ),
                "setup_args": {
                    "threshold": 0.85,
                    "seed": seed,
                    "infer_config": {
                        "segment_config": {"window_size": 5, "step_size": 3}
                    },
                },
            }
            self.mm_configs.append(config)
        add_ed_emotions(self.mm_configs)

    def _add_miti_mms(self):
        for miti_code, seed in MITI_SEEDS.items():
            config = {
                "name": "miti_%s" % miti_code.lower(),
                "model_type": "bert_query",
                "model_path": os.path.join(
                    MM_HOME, "models/miti_%s" % miti_code
                ),
                "setup_args": {
                    "threshold": 0.8,
                    "seed": seed,
                    "infer_config": {
                        "segment_config": {"window_size": 10, "step_size": 3}
                    },
                },
            }
            self.mm_configs.append(config)

    def _add_epitome_mms(self):
        """Add epitome micromodels"""
        emp_data = load_emp_data()
        emp_config = setup_emp_config(emp_data)
        self.mm_configs = self.mm_configs + emp_config

    def train_emotion_clf(self, filepath):
        """
        Train emotion classifier
        """
        ed_data = load_ed_data(filepath)
        labels = []
        featurized = None
        for emotion, utts in tqdm(ed_data.items()):
            bert_results = self.featurizer.run_bert(utts)

            for _, result_obj in bert_results.items():
                _results = result_obj["results"]
                scores = [_results[mm]["max_score"] for mm in self.ed_mms]

                if featurized is None:
                    featurized = np.array(scores)
                else:
                    featurized = np.vstack([featurized, scores])

            labels.extend([emotion] * len(utts))

        self.ed_model = ExplainableBoostingClassifier()
        self.ed_model.fit(featurized, labels)

    def run_emotion_clf_queries(self, queries: List[str]):
        """
        Run emotion classifier
        """
        bert_results = self.featurizer.run_bert(queries)
        featurized = None
        for _, result_obj in bert_results.items():
            scores = [
                result_obj["results"][mm]["max_score"] for mm in self.ed_mms
            ]
            if featurized is None:
                featurized = np.array(scores)
            else:
                featurized = np.vstack([featurized, scores])

        output = self.ed_model.predict(featurized)
        return output, featurized

    def run_emotion_clf_features(self, features):
        """
        Run emotion classifier on featurized vector
        """
        emotions = self.ed_model.classes_
        probs = self.ed_model.predict_proba(features)
        emotion_preds = _format_clf_results(probs, emotions)
        return emotion_preds

    def train_empathy_clfs(self, filepath):
        """
        Train empathy classifiers
        """
        emp_data = load_emp_data(filepath)
        featurized, labels = _featurize_emp(self.featurizer, emp_data)

        emotional_reactions_labels = [label[0] for label in labels]
        explorations_labels = [label[1] for label in labels]
        interpretations_labels = [label[2] for label in labels]

        self.emp_model_er = ExplainableBoostingClassifier()
        self.emp_model_er.fit(featurized, emotional_reactions_labels)

        self.emp_model_exp = ExplainableBoostingClassifier()
        self.emp_model_exp.fit(featurized, explorations_labels)

        self.emp_model_int = ExplainableBoostingClassifier()
        self.emp_model_int.fit(featurized, interpretations_labels)

    def run_empathy_clf_queries(self, queries: List[str]):
        """
        Run empathy classifiers
        """
        featurized = _featurize_emp_raw(self.featurizer, queries)

        emotional_reaction = self.emp_model_er.predict(featurized)
        exploration = self.emp_model_exp.predict(featurized)
        interpretation = self.emp_model_int.predict(featurized)
        return emotional_reaction, exploration, interpretation

    def run_empathy_clf_features(self, features):
        """
        Run empathy classifiers on featurized vector
        """
        er_probs = self.emp_model_er.predict_proba(features)
        emotional_reactions = _format_clf_results(
            er_probs, self.emp_model_er.classes_
        )

        exp_probs = self.emp_model_exp.predict_proba(features)
        explorations = _format_clf_results(
            exp_probs, self.emp_model_exp.classes_
        )

        int_probs = self.emp_model_int.predict_proba(features)
        interpretations = _format_clf_results(
            int_probs, self.emp_model_int.classes_
        )

        return emotional_reactions, explorations, interpretations

    def save_clf(self, model, model_path):
        """
        Save a EBM classifier
        """
        with open(model_path, "wb") as file_p:
            pickle.dump(model, file_p)

    def load_clf(self, model_path):
        """
        Load an EBM classifier
        """
        with open(model_path, "rb") as file_p:
            model = pickle.load(file_p)
        return model

    def encode(self, utterance: str):
        """
        Encode a single utterance.
        """
        bert_results = self.featurizer.run_bert([utterance])

        results = bert_results[0]
        emotion_scores = [
            results["results"][mm]["max_score"] for mm in self.ed_mms
        ]
        emotion_scores = np.array(emotion_scores, ndmin=2)

        empathy_features = _featurize_emp_raw(self.featurizer, [utterance])
        emp_er, emp_exp, emp_int = self.run_empathy_clf_features(
            empathy_features
        )

        emotion_preds = self.run_emotion_clf_features(emotion_scores)
        results["classifications"] = {
            "emotions": self.run_emotion_clf_features(emotion_scores),
            "empathy_emotional_reactions": emp_er[0],
            "empathy_explorations": emp_exp[0],
            "empathy_interpretations": emp_int[0],
        }

        results["explanations"] = {
            "global": pickle.dumps(self.ed_model.explain_global()),
            "local": pickle.dumps(self.ed_model.explain_local(emotion_scores)),
        }
        return results

    def encode_convo(self, convo):
        """
        Encode a single conversation
        """
        utts = [utt["utterance"] for utt in convo if utt["utterance"] != ""]
        bert_results = self.featurizer.run_bert(utts)

        emotion_features = None
        for utt_idx, _results in bert_results.items():
            _results["convo"] = convo[utt_idx]
            emotion_scores = [
                _results["results"][mm]["max_score"] for mm in self.ed_mms
            ]
            if emotion_features is None:
                emotion_features = np.array(emotion_scores)
            else:
                emotion_features = np.vstack(
                    [emotion_features, emotion_scores]
                )

        empathy_features = _featurize_emp_raw(self.featurizer, utts)

        emotion_preds = self.run_emotion_clf_features(emotion_features)
        emp_er, emp_exp, emp_int = self.run_empathy_clf_features(
            empathy_features
        )

        for utt_idx, _results in bert_results.items():
            try:
                _results["classifications"] = {
                    "emotions": emotion_preds[utt_idx],
                    "empathy_emotional_reactions": emp_er[utt_idx],
                    "empathy_explorations": emp_exp[utt_idx],
                    "empathy_interpretations": emp_int[utt_idx],
                }
            except:
                breakpoint()

        return bert_results


def load_encoder():
    """
    Initialize and load encoder.
    """
    ed_classifier = os.path.join(MODELS_DIR, "ed_classifier.pkl")
    emp_er_classifier = os.path.join(MODELS_DIR, "emp_er.pkl")
    emp_exp_classifier = os.path.join(MODELS_DIR, "emp_exp.pkl")
    emp_int_classifier = os.path.join(MODELS_DIR, "emp_int.pkl")
    encoder = Encoder(
        ed_classifier=ed_classifier,
        emp_er_classifier=emp_er_classifier,
        emp_exp_classifier=emp_exp_classifier,
        emp_int_classifier=emp_int_classifier,
    )
    return encoder


def main():
    """ Driver """

    encoder = load_encoder()
    result = encoder.encode("I almost got into a car accident.")
    breakpoint()


if __name__ == "__main__":
    main()
