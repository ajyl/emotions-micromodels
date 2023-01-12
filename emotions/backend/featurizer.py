"""
Encoder Module
"""
from typing import List

import os
import numpy as np
import pickle
import torch
from nltk import tokenize
from tqdm import tqdm
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox.ebm.ebm import EBMPreprocessor
from transformers import AutoTokenizer, AutoModel

from emotions.config import (
    add_ed_emotions,
    EMP_CONFIGS,
    EMP_TASKS,
    EMP_MMS,
    COG_DISTS,
)
from emotions.backend.BertFeaturizer import BertFeaturizer
from emotions.backend.data_utils import (
    load_ed_data,
    load_emp_data,
    reformat_emp_data,
)
from emotions.backend.EPITOME.empathy_classifier import EmpathyClassifier
from emotions.backend.PAIR.cross_scorer_model import CrossScorerCrossEncoder
from emotions.seeds.custom_emotions import ED_SEEDS
from emotions.seeds.miti_codes import MITI_SEEDS


MM_HOME = os.environ.get("MM_HOME")
MODELS_DIR = os.path.join(MM_HOME, "emotions/models")
DATA_DIR = os.path.join(MM_HOME, "emotions/data")
EPITOME_0 = os.path.join(DATA_DIR, "epitome_0.json")
EPITOME_1 = os.path.join(DATA_DIR, "epitome_1.json")

PAIR_PRETRAIN_MODEL = "mental/mental-roberta-base"


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
        "emotional_reactions": [],
        "interpretations": [],
        "explorations": [],
    }
    for instance in mm_data:
        for task in EMP_TASKS:
            level = instance[task]["level"]
            if level != "0":
                rationales = instance[task]["rationales"].split("|")
                rationales = [x for x in rationales if x != ""]
                all_rationales[task].extend(rationales)

    for config in mm_configs:
        config["setup_args"]["infer_config"] = {
            "segment_config": {"window_size": 10, "step_size": 4}
        }
        if config["name"] == "empathy_interpretations":
            config["setup_args"]["seed"] = all_rationales["interpretations"]
        if config["name"] == "empathy_explorations":
            config["setup_args"]["seed"] = all_rationales["explorations"]
        if config["name"] == "empathy_emotional_reactions":
            config["setup_args"]["seed"] = all_rationales[
                "emotional_reactions"
            ]
    return mm_configs


class Encoder:
    def __init__(self, **kwargs):
        """Load micromodels for encoding convo"""
        self.mm_configs = []
        self.device = kwargs.get("device", "cpu")
        self.ed_model = None
        self.emp_model_er = None
        self.emp_model_exp = None
        self.emp_model_int = None
        self.pair = None
        self.pair_tokenizer = None
        self.epitome = None

        print("Adding Empathetic Dialogue Micromodels...")
        self._add_ed_mms()
        print("Adding Epitome Micromodels...")
        self._add_empathy_mms(kwargs.get("emp_filepath", EPITOME_0))
        print("Adding MITI Micromodels...")
        self._add_miti_mms()
        print("Adding Cognitive Distortion Micromodels...")
        self._add_cog_dist_mms()
        print("Initializing Featurizer...")
        self._init_featurizer()

        ed_clf_path = kwargs.get("ed_classifier", None)
        emp_er_clf_path = kwargs.get("emp_er_classifier", None)
        emp_exp_clf_path = kwargs.get("emp_exp_classifier", None)
        emp_int_clf_path = kwargs.get("emp_int_classifier", None)

        epitome_er_clf_path = kwargs.get("epitome_er_classifier", None)
        epitome_exp_clf_path = kwargs.get("epitome_exp_classifier", None)
        epitome_int_clf_path = kwargs.get("epitome_int_classifier", None)

        pair_path = kwargs.get("pair_path", None)

        if ed_clf_path is not None:
            self.ed_model = self.load_clf(ed_clf_path)

        if emp_er_clf_path is not None:
            self.emp_model_er = self.load_clf(emp_er_clf_path)

        if emp_exp_clf_path is not None:
            self.emp_model_exp = self.load_clf(emp_exp_clf_path)

        if emp_int_clf_path is not None:
            self.emp_model_int = self.load_clf(emp_int_clf_path)

        if (
            epitome_er_clf_path is not None
            and epitome_exp_clf_path is not None
            and epitome_int_clf_path is not None
        ):
            print("Adding EPITOME...")
            self._init_epitome(
                epitome_er_clf_path, epitome_exp_clf_path, epitome_int_clf_path
            )

        if pair_path is not None:
            print("Adding PAIR...")
            self.pair, self.pair_tokenizer = self._init_pair(pair_path)

    def _init_featurizer(self):
        self.featurizer = BertFeaturizer(MM_HOME, self.mm_configs)
        self.mms = self.featurizer.list_micromodels
        self.ed_mms = [
            mm
            for mm in self.mms
            if mm.startswith("emotion_") or mm.startswith("custom_")
        ]
        self.emp_mms = [mm for mm in self.mms if mm.startswith("empathy_")]

    def _init_epitome(self, er_path, int_path, exp_path):
        """
        Initialize EPITOME.
        """
        self.epitome = EmpathyClassifier(
            self.device,
            ER_model_path=er_path,
            IP_model_path=int_path,
            EX_model_path=exp_path,
        )

    def _init_pair(self, model_path):
        """
        Initialize tokenizer, PAIR models.
        """
        encoder = AutoModel.from_pretrained(
            PAIR_PRETRAIN_MODEL, add_pooling_layer=False
        )
        tokenizer = AutoTokenizer.from_pretrained(PAIR_PRETRAIN_MODEL)
        cross_scorer = CrossScorerCrossEncoder(encoder).to(self.device)

        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        cross_scorer.load_state_dict(ckpt["model_state_dict"])
        cross_scorer.eval()
        return cross_scorer, tokenizer

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

    def _add_cog_dist_mms(self):
        """
        Add cognitive distortions / PHQ-9 responses
        """
        for cog_dist, seed in COG_DISTS.items():
            setup_args = {
                "threshold": 0.75,
                "infer_config": {
                    "segment_config": {"window_size": 10, "step_size": 4}
                },
                "seed": seed,
            }
            config = {
                "name": "cog_dist_%s" % cog_dist,
                "model_type": "bert_query",
                "setup_args": setup_args,
                "model_path": os.path.join(
                    MM_HOME, "models/cog_dist_%s" % cog_dist
                ),
            }
            self.mm_configs.append(config)

    def _add_empathy_mms(self, emp_filepath):
        """Add epitome micromodels"""
        emp_data = load_emp_data(emp_filepath)
        emp_config = setup_emp_config(emp_data)
        self.mm_configs = self.mm_configs + emp_config

    def train_emotion_clf(self, filepath):
        """
        Train emotion classifier
        """
        ed_data = load_ed_data(filepath)
        labels = []
        featurized = None
        emotions = []
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
            emotions.append(emotion)

        self.ed_model = ExplainableBoostingClassifier(
            feature_names=self.ed_mms
        )
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

        self.emp_model_er = ExplainableBoostingClassifier(
            feature_names=EMP_MMS
        )
        self.emp_model_er.fit(featurized, emotional_reactions_labels)

        self.emp_model_exp = ExplainableBoostingClassifier(
            feature_names=EMP_MMS
        )
        self.emp_model_exp.fit(featurized, explorations_labels)

        self.emp_model_int = ExplainableBoostingClassifier(
            feature_names=EMP_MMS
        )
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

    def run_epitome(self, prompt, response):
        """
        Run EPITOME.
        """
        empathy = self.epitome.predict_empathy([prompt], [response])
        return {
            "epitome_er": {
                "probabilities": empathy["er"]["probabilities"],
                "predictions": empathy["er"]["predictions"],
                "rationale": empathy["er"]["rationale"],
            },
            "epitome_int": {
                "probabilities": empathy["int"]["probabilities"],
                "predictions": empathy["int"]["predictions"],
                "rationale": empathy["int"]["rationale"],
            },
            "epitome_exp": {
                "probabilities": empathy["exp"]["probabilities"],
                "predictions": empathy["exp"]["predictions"],
                "rationale": empathy["exp"]["rationale"],
            },
        }

    def run_pair(self, prompt, response):
        """
        Run PAIR.
        """
        batch = self.pair_tokenizer(
            prompt,
            response,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            score = (
                self.pair.score_forward(**batch).sigmoid().flatten().tolist()
            )
        return score

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

    def encode_utterance(self, utterance: str):
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
        return {
            "emotion": {
                "predictions": emotion_preds,
            },
            "empathy": {
                "empathy_emotional_reactions": emp_er,
                "empathy_explorations": emp_exp,
                "empathy_interpretations": emp_int,
            },
            "micromodels": results["results"],
        }

    def encode(self, utterance, prev_utterance=None):
        """
        Encode a single dialogue turn.
        """
        response_encoding = self.encode_utterance(utterance)
        if prev_utterance is not None:
            if self.epitome:
                epitome_results = self.run_epitome(prev_utterance, utterance)
                for epitome_type, _results in epitome_results.items():
                    response_encoding["micromodels"][epitome_type] = {
                        "max_score": _results["probabilities"][1]
                        + _results["probabilities"][2],
                        "segment": _results["rationale"],
                    }

            if self.pair and self.pair_tokenizer:
                response_encoding["micromodels"]["pair"] = {
                    "max_score": self.run_pair(prev_utterance, utterance)[0],
                    "segment": ""
                }

        return response_encoding

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

    def get_explain(self):
        """
        Get explanation object.
        """
        explanation = None
        if self.ed_model:
            explanation = self.ed_model.explain_global()

        return {
            "explanations": {"global": explanation}
        }


def load_encoder(
    ed_path=None,
    emp_er_path=None,
    emp_exp_path=None,
    emp_int_path=None,
    epitome_er_path=None,
    epitome_exp_path=None,
    epitome_int_path=None,
    pair_path=None,
    device="cpu",
):
    """
    Initialize and load encoder.
    """
    encoder = Encoder(
        ed_classifier=ed_path,
        emp_er_classifier=emp_er_path,
        emp_exp_classifier=emp_exp_path,
        emp_int_classifier=emp_int_path,
        epitome_er_classifier=epitome_er_path,
        epitome_exp_classifier=epitome_exp_path,
        epitome_int_classifier=epitome_int_path,
        pair_path=pair_path,
        device=device,
    )
    # encoder.train_empathy_clfs(EPITOME_1)
    # encoder.save_clf(encoder.emp_model_er, emp_er_classifier)
    # encoder.save_clf(encoder.emp_model_exp, emp_exp_classifier)
    # encoder.save_clf(encoder.emp_model_int, emp_int_classifier)
    return encoder


def main():
    """ Driver """
    ed_classifier = os.path.join(MODELS_DIR, "ed_classifier.pkl")
    emp_er_classifier = os.path.join(MODELS_DIR, "emp_er.pkl")
    emp_exp_classifier = os.path.join(MODELS_DIR, "emp_exp.pkl")
    emp_int_classifier = os.path.join(MODELS_DIR, "emp_int.pkl")
    epitome_er_classifier = os.path.join(MODELS_DIR, "EPITOME_ER.pth")
    epitome_exp_classifier = os.path.join(MODELS_DIR, "EPITOME_EXP.pth")
    epitome_int_classifier = os.path.join(MODELS_DIR, "EPITOME_INT.pth")
    pair_path = os.path.join(MODELS_DIR, "pair.pth")

    encoder = load_encoder(
        ed_path=ed_classifier,
        emp_er_path=emp_er_classifier,
        emp_exp_path=emp_exp_classifier,
        emp_int_path=emp_int_classifier,
        epitome_er_path=epitome_er_classifier,
        epitome_exp_path=epitome_exp_classifier,
        epitome_int_path=epitome_int_classifier,
        pair_path=pair_path,
        device="cpu",
    )

    prompt = "I almost got into a car accident."
    response = "You almost got into a car accident."

    testing2 = encoder.encode(prompt, response)

    # result = encoder.encode("well so I 've been working with the weight management clinic and I just when I found out that this was an opportunity I thought why not use this as another resource to help me lose weight")
    # test_utterance = "well so I 've been working with the weight management clinic and I just when I found out that this was an opportunity I thought why not use this as another resource to help me lose weight"
    # test_utterance = "I feel so sad because I almost got in a car accident."
    # encoder.encode(test_utterance)

    # bert_results = encoder.featurizer.run_bert([test_utterance])
    # results = bert_results[0]
    # emotion_scores = [
    #    results["results"][mm]["max_score"] for mm in encoder.ed_mms
    # ]
    # emotion_scores = np.array(emotion_scores, ndmin=2)
    # x = encoder.ed_model.explain_local(emotion_scores)
    # show(x)

    # breakpoint()


if __name__ == "__main__":
    main()
