import os
import time
import pickle
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

# Assuming that the TrajectoryBatchDataset class is defined as provided
# and that the dataset instance is ready and create_batches() has been called.

class HigherOrderMarkovChain:
    def __init__(self, config, order=1):
        self.order = order
        self.transition_counts = None
        self.transition_probs = None
        self.state_index_mapping = {}
        self.index_state_mapping = {}
        self.states = []
        self.num_states = 0
        self.config = config
        self.logger = None


    def train(self, dataset, logger, model_checkpoint_directory: str):
        """
        Train the Higher-Order Markov Chain using the provided sequences.

        :param sequences: List of sequences (list of lists), where each sequence is a list of states.
        """
        logger.info("Building transition matrix...")
        self.logger = logger
        sequences = dataset.data
        self._build_state_mappings(sequences)
        self._build_transition_matrix(sequences)
        self.save_checkpoint(model_checkpoint_directory)

    def _build_state_mappings(self, sequences):
        # Build state to index mappings
        state_set = set()
        for sequence in sequences:
            state_set.update(sequence)
        self.states = list(state_set)
        self.num_states = len(self.states)
        self.state_index_mapping = {state: idx for idx, state in enumerate(self.states)}
        self.index_state_mapping = {idx: state for state, idx in self.state_index_mapping.items()}


    def _build_transition_matrix(self, sequences):
        # Initialize transition counts with ones for smoothing
        self.transition_counts = defaultdict(lambda: defaultdict(lambda: 1))

        for sequence in tqdm(sequences, desc="Processing training sequences"):
            for i in range(len(sequence) - self.order):
                current_state = tuple(sequence[i:i + self.order])
                next_state = sequence[i + self.order]
                self.transition_counts[current_state][next_state] += 1

        # Convert counts to probabilities
        self.transition_probs = {}
        for current_state, next_states in self.transition_counts.items():
            total = sum(next_states.values())
            self.transition_probs[current_state] = {state: count / total for state, count in next_states.items()}

    def predict_next_state(self, current_sequence):
        """
        Predict the next state(s) given the current sequence.

        :param current_sequence: List of the current sequence of states.
        :return: List of predicted next states, sorted by probability in descending order.
        """
        current_state = tuple(current_sequence[-self.order:])
        next_state_probs = self.transition_probs.get(current_state, {})
        if not next_state_probs:
            return []
        sorted_next_states = sorted(next_state_probs.items(), key=lambda item: item[1], reverse=True)
        return [state for state, prob in sorted_next_states[:5]]  # Return top 5

    def predict_next_n_steps(self, sequence, n=5):
        """
        Predict the next n steps given the initial sequence.

        :param sequence: Initial sequence of states (list).
        :param n: Number of steps to predict.
        :return: List of lists containing predicted states at each step.
        """
        predictions = []
        current_sequence = sequence.copy()
        for _ in range(n):
            next_states = self.predict_next_state(current_sequence)
            if not next_states:
                break  # If no next state is found
            predictions.append(next_states)
            current_sequence.append(next_states[0])  # Append the most probable next state
        return predictions

    def save_checkpoint(self, model_checkpoint_directory):
        """
        Save the trained model to a file.

        :param filepath: Path to the file where the model will be saved.
        """
        model_data = {
            'order': self.order,
            'transition_probs': self.transition_probs,
            'state_index_mapping': self.state_index_mapping,
            'index_state_mapping': self.index_state_mapping,
            'states': self.states,
            'num_states': self.num_states
        }
        checkpoint = {
            'model': model_data,
            'config': self.config,
        }
        checkpoint_path = os.path.join(model_checkpoint_directory, 'checkpoint.pt')
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_state_dict(self, model_data):
        """
        Load a trained model from a file.

        :param filepath: Path to the file where the model is saved.
        :return: An instance of HigherOrderMarkovChain with loaded parameters.
        """
        self.transition_probs = model_data['transition_probs']
        self.state_index_mapping = model_data['state_index_mapping']
        self.index_state_mapping = model_data['index_state_mapping']
        self.states = model_data['states']
        self.num_states = model_data['num_states']
        self.order = model_data['order']

    def evaluate(self, test_dataset):
        """
        Evaluate the model using the test sequences.

        :param test_sequences: List of test sequences.
        :return: Dictionary containing evaluation metrics.
        """
        print("Evaluating model...")
        total_predictions = 0
        hit_step5_at1 = 0
        hit_step5_at3 = 0
        hit_step5_at5 = 0
        bleu_scores = 0

        start_time = time.time()
        for test_sequence in tqdm(test_dataset.data, desc="Processing test sequences"):
            if len(test_sequence) < self.order + 5:
                continue
            for i in range(len(test_sequence) - (self.order + 5) + 1):
                original_sequence = test_sequence[i:i + self.order + 5]  # Include the initial sequence and actual next steps
                observe_sequence = original_sequence[:self.order]
                actual_next_steps = original_sequence[self.order:self.order + 5]

                predicted_next_steps = self.predict_next_n_steps(observe_sequence, n=5)

                if len(predicted_next_steps) < 5:
                    continue  # Skip if predictions are incomplete

                actual_step5 = actual_next_steps[4]
                predicted_step5 = predicted_next_steps[4]

                if actual_step5 == predicted_step5[0]:
                    hit_step5_at1 += 1
                if actual_step5 in predicted_step5[:3]:
                    hit_step5_at3 += 1
                if actual_step5 in predicted_step5[:5]:
                    hit_step5_at5 += 1

                # For BLEU score calculation
                predicted_sentence = [predicted_next_steps[j][0] for j in range(len(predicted_next_steps))]
                bleu_scores += sentence_bleu([actual_next_steps], predicted_sentence)

                total_predictions += 1

        test_duration = time.time() - start_time

        return [
            f"Accuracy@1: {hit_step5_at1 / total_predictions if total_predictions else 0}",
            f"Accuracy@3: {hit_step5_at3 / total_predictions if total_predictions else 0}",
            f"Accuracy@5: {hit_step5_at5 / total_predictions if total_predictions else 0}",
            f"BLEU score: {bleu_scores / total_predictions if total_predictions else 0}",
            f"Test duration: {test_duration:.3f}(s)",
            f"Samples: {total_predictions}"
        ]
