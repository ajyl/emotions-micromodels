"""
Task-specific classifier.

TODO Items:
* Dump features into file, both train and test.
* Load features
* Configurable aggregators
* List of aggregators
* Format data for test_from_file()
* batch inference
* test_from_file()
* save_model()?
* load_model()?
"""
from typing import List, Mapping, Tuple, Any

import sys
import time
from collections import defaultdict
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from interpret.glassbox import ExplainableBoostingClassifier
from src.Orchestrator import Orchestrator, get_model_name
from src.aggregators.SimpleRatioAggregator import SimpleRatioAggregator
from src.metrics import recall, precision, f1


def _to_binary_vectors(
    mm_output: Mapping[int, List[int]], utterance_lengths: Mapping[int, int]
) -> List[np.ndarray]:
    """
    Format the output of micromodels into one-hot encodings.

    :param mm_output: output vector from micromodels. mm_output is a
        nested dictionary that maps the name of the micromodels to
        an inner dictionary. The inner dictionary maps an index of
        utterance groups to a list of indices that correspond to "hits"
        from each micromodel.
    :param utterance_lengths: mapping of utterance group indices to the
        number of sentences in the group.
    :return: A list of ndarrays, where the length of each ndarray is the
        number of sentences in the corresponding utterance group.
        i.e., i'th ndarray.shape = (utterance_lengths[i], )
    """
    binary_vectors = []
    for utt_idx, length in utterance_lengths.items():
        vec = np.zeros(length, dtype=int)
        vec[mm_output[utt_idx]] = 1
        binary_vectors.append(vec)
    return binary_vectors


class TaskClassifier:
    """
    Classifier for specific tasks.
    """

    positive_value = 1
    negative_value = 0

    def __init__(
        self, mm_basepath: str, configs: List[Mapping[str, Any]] = None
    ) -> None:
        """
        Initialize the classifier (EBM), orchestrator, and aggregators.

        :param mm_basepath: fle path to where micromodels are stored.
        :param configs: list of configurations for each micromodel.
        """
        self.model = None
        self.orchestrator = Orchestrator(mm_basepath)
        if configs:
            self.set_configs(configs)
        # TODO: Load this from config file.
        # TODO: Make this a list of aggregators
        self.aggregator = SimpleRatioAggregator()

    def load_data(
        self, data_path: str = None, **kwargs
    ) -> Tuple[List[Tuple[List[str], str]], List[Tuple[List[str], str]]]:
        """
        Load task-specific data.
        This module assumes the following format for all of its data:

        [([sentence_1, sentence_2, ...], label), ...]

        Each instance of data is a tuple, where the first element
        is a list of sentences, and the second element is the corresponding
        label.

        :param data_path: filepath to where the data is stored.
        :return: tuple of train and test data.
        """
        raise NotImplementedError("load_data() not implemented!")

    def set_configs(self, configs: List[Mapping[str, Any]]) -> None:
        """
        Set configs for the orchestrator.

        :param configs: list of configurations for each micromodel.
        """
        self.orchestrator.set_configs(configs)
        self.features = [get_model_name(config) for config in configs]

    def load_micromodels(self) -> None:
        """
        Load all micromodels, while training those that need training.
        """
        self.orchestrator.train_all()
        self.orchestrator.load_models(force_reload=True)

    def featurize_data(
        self, data: List[Tuple[List[str], Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Featurize data, where the input is a list of tuples.
        The first element of the tuple is a list of utterances and the
        second element is their label.

        :param data: list of data instances. See load_data() for details
            on the data format.

        :return:
            - featurized - ndarray of shape (len(data), number of micromodels)
            - labels - list of labels
        """
        utterances = [instance[0] for instance in data]
        labels = [instance[1] for instance in data]

        micromodel_output = self.run_micromodels(utterances)
        utterance_lengths = {
            idx: len(sentences) for idx, sentences in enumerate(utterances)
        }
        featurized = None

        for mm_output in micromodel_output.values():
            # mm_output: {utt_idx: list[int]}:
            # map utterance ids to list of matched indices

            # mm_outputs is a list of binary vectors, represented as ndarrays
            mm_outputs = _to_binary_vectors(mm_output, utterance_lengths)

            feature_values = self.aggregator.aggregate(mm_outputs)
            if featurized is None:
                featurized = feature_values
            else:
                featurized = np.vstack([featurized, feature_values])
        featurized = np.transpose(featurized)
        return featurized, labels

    def run_micromodel(
        self,
        config: Mapping[str, Any],
        utterances: List[List[str]],
    ) -> Mapping[int, List[int]]:
        """
        Run a single micromodel that's specified in config.
        Processes the input utterances in parallel.

        :param config: micromodel specs.
        :param utterances: list of utterance groups.

        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        model_type = config.get("model_type")
        if not model_type:
            raise RuntimeError("model_type not found in config.")
        model_name = get_model_name(config)
        if not model_name:
            raise RuntimeError("name not found in config.")

        binary_vectors = self.orchestrator.batch_infer_config(
            utterances, config
        )
        return binary_vectors[model_name]

    def run_micromodels(
        self, utterances: List[List[str]]
    ) -> Mapping[str, Mapping[int, List[int]]]:
        """
        Run micromodels on group of utterances.

        :param utterances: list of utterance groups. This is
            a list of lists, where each inner list corresponds
            to a single utterance group, a.k.a. a list of sentences.
        :return: micromodel results. These are represented in the
            following way:
            {

                micromodel_name: {

                    utterance_group_idx: [

                        hit_1_idx, hit_2_idx, ..., hit_n_idx

                    ],

                ...

            }
            where utterance_group_idx is the index of each utterance group,
            and hit_n_idx are the indices within each utterance_group that
            correspond to hits.
        """
        micromodel_outputs = {}
        for idx, config in enumerate(self.orchestrator.configs):
            micromodel_name = get_model_name(config)
            print(
                "Running micromodel %s (%d/%d)."
                % (micromodel_name, idx + 1, len(self.orchestrator.configs))
            )
            micromodel_outputs[micromodel_name] = self.run_micromodel(
                config, utterances
            )
        return micromodel_outputs

    def train(self, training_data: List[Tuple[List[str], str]]) -> None:
        """
        Train the task-specific classifier. This method includes featurizing the
        input data (i.e., running the micromodels and aggregating the results)
        and feeding the featurized data to the task-specific classifier.

        :param training_data: Training data in the following format:

            [

                ([sentence_1, sentence_2, ...], label),

                ...

            ]

            See load_data() for more information.
        """
        if self.orchestrator.num_micromodels < 2:
            raise RuntimeError(
                "EBM requires more than 1 micromodel in order to be trained."
            )
        self.model = ExplainableBoostingClassifier()
        x_train, y_train = self.featurize_data(training_data)
        self.model.fit(x_train, y_train)

    def train_featurized(
        self, featurized_data: np.ndarray, labels: List[str]
    ) -> None:
        """
        Train task-specific classifier using already featurized data.

        :param featurized_data: ndarray of shape
            (len(input_data), number of micromodels).
        :param labels: list of labels.
        """
        input_shape = featurized_data.shape
        if len(input_shape) != 2:
            raise RuntimeError(
                "Invalid shape for featurized_data: %s" % (input_shape,)
            )
        if input_shape[1] != self.orchestrator.num_micromodels:
            raise RuntimeError(
                "Mismatch between input shape (%s) and number of micromodels (%d)"
                % (input_shape, self.orchestrator.num_micromodels)
            )
        if featurized_data.shape[1] < 2:
            raise RuntimeError(
                "EBM requires more than 1 micromodel to be trained."
            )

        self.model = ExplainableBoostingClassifier()
        self.model.fit(featurized_data, labels)

    def infer(self, utterances: List[str]) -> Tuple[Any, float]:
        """
        Infer on a single instance (list of utterances).

        :param utterances: A single utterance group.
        :return: tuple of prediction and probability.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        formatted = [(utterances, None)]
        featurized, _ = self.featurize_data(formatted)
        return self.infer_featurized(featurized)

    def infer_featurized(
        self, feature_vector: np.ndarray
    ) -> Tuple[Any, float]:
        """
        Infer on a single featurized vector.

        :param feature_vector: ndarray of size
            (number of utterances, number of micromodels).
        :return: tuple of preidction and probability.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        prediction = self.model.predict(feature_vector)[0]
        probability = self.model.predict_proba(feature_vector)[0][1]
        return prediction, probability

    def test(
        self, test_data: List[Tuple[List[str], Any]]
    ) -> Mapping[str, float]:
        """
        Run tests where test_data is in the form of
        [([sentence_1, sentence_2, ...], label), ...]
        See load_data() for more details.

        This method evaluates the following metrics:
        * Precision
        * Recall
        * F1 score
        * Accuracy

        :param test_data: data to test.
        :return: test restuls.
        """
        predictions = []
        x_test, groundtruth = self.featurize_data(test_data)
        # TODO: do batch inference
        for row in x_test:
            prediction, _ = self.infer_featurized([row])
            predictions.append(prediction)

        num_correct = 0
        for idx, pred in enumerate(predictions):
            if pred == groundtruth[idx]:
                num_correct += 1

        accuracy = num_correct / len(groundtruth)
        return {
            "f1": f1(
                predictions,
                groundtruth,
            ),
        }
