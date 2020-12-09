"""Script for training and evaluating QA models.

Example command to train the (medium-sized) baseline model on SQuAD
with a GPU, and write its predictions to an output file:

Usage:
    python3 main.py \
        --use_gpu \
        --model "baseline" \
        --model_path "squad_model.pt" \
        --train_path "datasets/squad_train.jsonl.gz" \
        --dev_path "datasets/squad_dev.jsonl.gz" \
        --output_path "squad_predictions.txt" \
        --hidden_dim 256 \
        --bidirectional \
        --do_train \
        --do_test

    Testing write_predictions:
    python3 main.py \
        --use_gpu \
        --model "baseline" \
        --model_path "baseline_small_squad.pt" \
        --train_path "datasets/squad_train.jsonl.gz" \
        --dev_path "datasets/squad_dev.jsonl.gz" \
        --output_path "baseline_small_squad_predictions.txt" \
        --hidden_dim 128 \
        --bidirectional \
        --do_test

Author:
    Shrey Desai and Yasumasa Onoe
"""

import argparse
import pprint
import json

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data import QADataset, Tokenizer, Vocabulary

from model import BaselineReader
from utils import cuda, search_span_endpoints, topk_span_endpoints, unpack, calculate_multiplier_increments

import spacy
sp = spacy.load("en_core_web_sm")

interrogative_dict = {
  "who": ["PERSON","NORP","ORG"],
  "where": ["FAC","GRE","LOC"],
  "what": [],
  "which": []
}

import heapq

punc_string = '.!?'

_TQDM_BAR_SIZE = 75
_TQDM_LEAVE = False
_TQDM_UNIT = ' batches'
_TQDM_OPTIONS = {
    'ncols': _TQDM_BAR_SIZE, 'leave': _TQDM_LEAVE, 'unit': _TQDM_UNIT
}


parser = argparse.ArgumentParser()

# Training arguments.
parser.add_argument('--device', type=int)
parser.add_argument(
    '--use_gpu',
    action='store_true',
    help='whether to use GPU',
)
parser.add_argument(
    '--model',
    type=str,
    required=True,
    choices=['baseline'],
    help='which model to use',
)
parser.add_argument(
    '--model_path',
    type=str,
    required=True,
    help='path to load/save model checkpoints',
)
parser.add_argument(
    '--embedding_path',
    type=str,
    default='glove/glove.6B.300d.txt',
    help='GloVe embedding path',
)
parser.add_argument(
    '--train_path',
    type=str,
    required=True,
    help='training dataset path',
)
parser.add_argument(
    '--dev_path',
    type=str,
    required=True,
    help='dev dataset path',
)
parser.add_argument(
    '--max_context_length',
    type=int,
    default=384,
    help='maximum context length (do not change!)',
)
parser.add_argument(
    '--max_question_length',
    type=int,
    default=64,
    help='maximum question length (do not change!)',
)
parser.add_argument(
    '--output_path',
    type=str,
    required=False,
    help='predictions output path',
)
parser.add_argument(
    '--shuffle_examples',
    action='store_true',
    help='shuffle training example at the beginning of each epoch',
)

# Optimization arguments.
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='number of training epochs',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='training and evaluation batch size',
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help='training learning rate',
)
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.,
    help='training weight decay',
)
parser.add_argument(
    '--grad_clip',
    type=float,
    default=0.5,
    help='gradient norm clipping value',
)
parser.add_argument(
    '--early_stop',
    type=int,
    default=3,
    help='number of epochs to wait until early stopping',
)
parser.add_argument(
    '--do_train',
    action='store_true',
    help='flag to enable training',
)
parser.add_argument(
    '--do_test',
    action='store_true',
    help='flag to enable testing',
)

# Model arguments.
parser.add_argument(
    '--vocab_size',
    type=int,
    default=50000,
    help='vocabulary size (dynamically set, do not change!)',
)
parser.add_argument(
    '--embedding_dim',
    type=int,
    default=300,
    help='embedding dimension',
)
parser.add_argument(
    '--hidden_dim',
    type=int,
    default=256,
    help='hidden state dimension',
)
parser.add_argument(
    '--rnn_cell_type',
    choices=['lstm', 'gru'],
    default='lstm',
    help='Type of RNN cell',
)
parser.add_argument(
    '--bidirectional',
    action='store_true',
    help='use bidirectional RNN',
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.,
    help='dropout on passage and question vectors',
)


def _print_arguments(args):
    """Pretty prints command line args to stdout.

    Args:
        args: `argparse` object.
    """

    args_dict = vars(args)
    pprint.pprint(args_dict)


def _select_model(args):
    """
    Selects and initializes model. To integrate custom models, (1)
    add the model name to the parser choices above, and (2) modify
    the conditional statements to include an instance of the model.

    Args:
        args: `argparse` object.

    Returns:
        Instance of a PyTorch model supplied with args.
    """
    if args.model == 'baseline':
        return BaselineReader(args)
    else:
        raise RuntimeError(f'model \'{args.model}\' not recognized!')


def _early_stop(args, eval_history):
    """
    Determines early stopping conditions. If the evaluation loss has
    not improved after `args.early_stop` epoch(s), then training
    is ended prematurely. 

    Args:
        args: `argparse` object.
        eval_history: List of booleans that indicate whether an epoch resulted
            in a model checkpoint, or in other words, if the evaluation loss
            was lower than previous losses.

    Returns:
        Boolean indicating whether training should stop.
    """
    return (
        len(eval_history) > args.early_stop
        and not any(eval_history[-args.early_stop:])
    )


def _calculate_loss(
    start_logits, end_logits, start_positions, end_positions
):
    """
    Calculates cross-entropy loss for QA samples, which is defined as
    the mean of the loss values incurred by the starting and ending position
    distributions when compared to the gold endpoints.

    Args:
        start_logits: Predicted distribution over start positions.
        end_logits: Predicted distribution over end positions.
        start_positions: Gold start positions.
        end_positions: Gold end positions.

    Returns:
        Loss value for a batch of sasmples.
    """
    # If the gold span is outside the scope of the maximum
    # context length, then ignore these indices when computing the loss.
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    # Compute the cross-entropy loss for the start and end logits.
    criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = criterion(start_logits, start_positions)
    end_loss = criterion(end_logits, end_positions)

    return (start_loss + end_loss) / 2.


def train(args, epoch, model, dataset):
    """
    Trains the model for a single epoch using the training dataset.

    Args:
        args: `argparse` object.
        epoch: Epoch number (used in the `tqdm` bar).
        model: Instance of the PyTorch model.
        dataset: Training dataset.

    Returns:
        Training cross-entropy loss normalized across all samples.
    """
    # Set the model in "train" mode.
    model.train()

    # Cumulative loss and steps.
    train_loss = 0.
    train_steps = 0

    # Set up optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Set up training dataloader. Creates `args.batch_size`-sized
    # batches from available samples.
    train_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=args.shuffle_examples),
        **_TQDM_OPTIONS,
    )

    for batch in train_dataloader:
        # Zero gradients.
        optimizer.zero_grad()

        # Forward inputs, calculate loss, optimize model.
        start_logits, end_logits = model(batch)
        loss = _calculate_loss(
            start_logits,
            end_logits,
            batch['start_positions'],
            batch['end_positions'],
        )
        loss.backward()
        if args.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Update tqdm bar.
        train_loss += loss.item()
        train_steps += 1
        train_dataloader.set_description(
            f'[train] epoch = {epoch}, loss = {train_loss / train_steps:.6f}'
        )

    return train_loss / train_steps


def evaluate(args, epoch, model, dataset):
    """
    Evaluates the model for a single epoch using the development dataset.

    Args:
        args: `argparse` object.
        epoch: Epoch number (used in the `tqdm` bar).
        model: Instance of the PyTorch model.
        dataset: Development dataset.

    Returns:
        Evaluation cross-entropy loss normalized across all samples.
    """
    # Set the model in "evaluation" mode.
    model.eval()

    # Cumulative loss and steps.
    eval_loss = 0.
    eval_steps = 0

    # Set up evaluation dataloader. Creates `args.batch_size`-sized
    # batches from available samples. Does not shuffle.
    eval_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    with torch.no_grad():
        for batch in eval_dataloader:
            # Forward inputs, calculate loss.
            start_logits, end_logits = model(batch)
            loss = _calculate_loss(
                start_logits,
                end_logits,
                batch['start_positions'],
                batch['end_positions'],
            )

            # Update tqdm bar.
            eval_loss += loss.item()
            eval_steps += 1
            eval_dataloader.set_description(
                f'[eval] epoch = {epoch}, loss = {eval_loss / eval_steps:.6f}'
            )

    return eval_loss / eval_steps


def get_full_sentence(passage, start_index, end_index):
    #Maybe devalue sentences ending in question mark?

    first_index = start_index
    last_index = end_index

    while (first_index >= 0 and passage[first_index] not in punc_string):
        first_index -= 1

    while (last_index < len(passage) and passage[last_index] not in punc_string):
        last_index += 1

    return first_index, last_index


#Matches first question interrogative to appropriate entity labels in the answer
#i.e 'Where' to Location or Geographic Place
#Weights higher if answer contains entity of the appropriate type
def compare_interrogatives(increments, topk, passage_truecase, first_interrogative):
    heap = []

    if not first_interrogative == 'who' and not first_interrogative == 'where':
        return sorted(topk)


    for max_prob, start_index, end_index in topk:
        pred_span = str(passage_truecase[start_index:(end_index + 1)])
        ans_ents = sp(pred_span).ents
        print(first_interrogative)
        #list of entity labels that match the interrogative
        mapped_ents = interrogative_dict[first_interrogative]

        matching_ents = 0
        for ent in ans_ents:
            if ent.label_ in mapped_ents:
                matching_ents += 1
            print(ent.label_)
            heapq.heappush(heap, (matching_ents, max_prob, start_index, end_index))

    
    multipliers = calculate_multiplier_increments(increments)
    second_heap = []
    topk_index = 0
    while heap:
        temp = heapq.heappop(heap)
        temp = (-(temp[1] * multipliers[topk_index]), temp[2], temp[3])
        second_heap.append(temp)
        topk_index += 1
    second_heap.sort()
    return second_heap

                


def count_common_entities(increments, topk, passage_truecase, question_has_ents, question_ents_text):
    heap = []
    for max_prob, start_index, end_index in topk:
        pred_span = str(passage_truecase[start_index:(end_index + 1)])

        if question_has_ents:
            sent_start, sent_end = get_full_sentence(passage_truecase, start_index, end_index)
            full_sent = str(passage_truecase[sent_start+1:(sent_end + 1)])
            ans_ents = sp(full_sent).ents

            common_ents = 0
            for ent in ans_ents:
                if ent.text in question_ents_text:
                    common_ents += 1
            heapq.heappush(heap, (common_ents, max_prob, start_index, end_index))

    
    multipliers = calculate_multiplier_increments(increments)
    second_heap = []
    topk_index = 0
    while heap:
        temp = heapq.heappop(heap)
        temp = (-(temp[1] * multipliers[topk_index]), temp[2], temp[3])
        second_heap.append(temp)
        topk_index += 1
    second_heap.sort()
    return second_heap
        

def write_predictions(args, model, dataset, dataset_truecase):
    """
    Writes model predictions to an output file. The official QA metrics (EM/F1)
    can be computed using `evaluation.py`. 

    Args:
        args: `argparse` object.
        model: Instance of the PyTorch model.
        dataset: Test dataset (technically, the development dataset since the
            official test datasets are blind and hosted by official servers).
    """
    # Load model checkpoint.
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    # Set up test dataloader.
    test_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    # Output predictions.
    outputs = []

    with torch.no_grad():
        for (i, batch) in enumerate(test_dataloader):
            # Forward inputs.
            start_logits, end_logits = model(batch)

            # Form distributions over start and end positions.
            batch_start_probs = F.softmax(start_logits, 1)
            batch_end_probs = F.softmax(end_logits, 1)

            for j in range(start_logits.size(0)):
                # Find question index and passage.
                sample_index = args.batch_size * i + j
                qid, passage, _, _, _ = dataset.samples[sample_index]
                _, passage_truecase, question_truecase, _, _ = dataset_truecase.samples[sample_index]

                # Unpack start and end probabilities. Find the constrained
                # (start, end) pair that has the highest joint probability.
                start_probs = unpack(batch_start_probs[j])
                end_probs = unpack(batch_end_probs[j])

                # original way
                # start_index, end_index = search_span_endpoints(start_probs, end_probs)

                # new way using our custom topk_span_endpoints function
                topk = topk_span_endpoints(start_probs, end_probs)
                
                # question = 'Who won the 2004 Super Bowl?'
                question = question_truecase
                question_words = [x.lower() for x in question]
                question_ents = set(sp(" ".join(question)).ents)
                question_ents_text = set([ent.text for ent in question_ents])
                question_has_ents = len(question_ents) > 0
                
                #Finds the first interrogative (who, where, what) in the sentence
                #Use this to match with NER in answer span
                first_interrogative = ''
                for word in question_words:
                    if word in interrogative_dict:
                        first_interrogative = word
                        break

                old_topk = topk

                print(question_words)

                topk = count_common_entities(5, topk, passage_truecase, question_has_ents, question_ents_text)

                print(topk)

                topk = compare_interrogatives(5, topk, passage_truecase, first_interrogative)

                print(topk)


                # probably want additional logic to not completely sort based on number of common entities
                # some sort of voting system that takes into account both probabilities and num common entities?
                temp2 = topk[0] if topk else old_topk[0]
                final_start, final_end = temp2[1], temp2[2]
                
                # Grab predicted span.
                # pred_span = ' '.join(passage[start_index:(end_index + 1)])
                pred_span = ' '.join(passage[final_start:(final_end + 1)])

                # Add prediction to outputs.
                outputs.append({'qid': qid, 'answer': pred_span})

    # Write predictions to output file.
    with open(args.output_path, 'w+') as f:
        for elem in outputs:
            f.write(f'{json.dumps(elem)}\n')


def main(args):
    """
    Main function for training, evaluating, and checkpointing.

    Args:
        args: `argparse` object.
    """
    # Print arguments.
    print('\nusing arguments:')
    _print_arguments(args)
    print()

    # Check if GPU is available.
    if not args.use_gpu and torch.cuda.is_available():
        print('warning: GPU is available but args.use_gpu = False')
        print()

    # Set up datasets.
    train_dataset = QADataset(args, args.train_path)
    dev_dataset = QADataset(args, args.dev_path)
    train_dataset_truecase = QADataset(args, args.train_path, False)
    dev_dataset_truecase = QADataset(args, args.dev_path, False)

    # Create vocabulary and tokenizer.
    vocabulary = Vocabulary(train_dataset.samples, args.vocab_size)
    tokenizer = Tokenizer(vocabulary)
    for dataset in (train_dataset, dev_dataset):
        dataset.register_tokenizer(tokenizer)
    args.vocab_size = len(vocabulary)
    args.pad_token_id = tokenizer.pad_token_id
    print(f'vocab words = {len(vocabulary)}')

    # Print number of samples.
    print(f'train samples = {len(train_dataset)}')
    print(f'dev samples = {len(dev_dataset)}')
    print()

    # Select model.
    model = _select_model(args)
    num_pretrained = model.load_pretrained_embeddings(
        vocabulary, args.embedding_path
    )
    pct_pretrained = round(num_pretrained / len(vocabulary) * 100., 2)
    print(f'using pre-trained embeddings from \'{args.embedding_path}\'')
    print(
        f'initialized {num_pretrained}/{len(vocabulary)} '
        f'embeddings ({pct_pretrained}%)'
    )
    print()

    if args.use_gpu:
        model = cuda(args, model)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'using model \'{args.model}\' ({params} params)')
    print(model)
    print()

    if args.do_train:
        # Track training statistics for checkpointing.
        eval_history = []
        best_eval_loss = float('inf')

        # Begin training.
        for epoch in range(1, args.epochs + 1):
            # Perform training and evaluation steps.
            train_loss = train(args, epoch, model, train_dataset)
            eval_loss = evaluate(args, epoch, model, dev_dataset)

            # If the model's evaluation loss yields a global improvement,
            # checkpoint the model.
            eval_history.append(eval_loss < best_eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), args.model_path)
            
            print(
                f'epoch = {epoch} | '
                f'train loss = {train_loss:.6f} | '
                f'eval loss = {eval_loss:.6f} | '
                f"{'saving model!' if eval_history[-1] else ''}"
            )

            # If early stopping conditions are met, stop training.
            if _early_stop(args, eval_history):
                suffix = 's' if args.early_stop > 1 else ''
                print(
                    f'no improvement after {args.early_stop} epoch{suffix}. '
                    'early stopping...'
                )
                print()
                break

    if args.do_test:
        # Write predictions to the output file. Use the printed command
        # below to obtain official EM/F1 metrics.
        write_predictions(args, model, dev_dataset, dev_dataset_truecase)
        eval_cmd = (
            'python3 evaluate.py '
            f'--dataset_path {args.dev_path} '
            f'--output_path {args.output_path}'
        )
        print()
        print(f'predictions written to \'{args.output_path}\'')
        print(f'compute EM/F1 with: \'{eval_cmd}\'')
        print()


if __name__ == '__main__':
    main(parser.parse_args())
