import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.autograd import Function
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler

# from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from src.NeuralModel import TransformerDataset

import transformers
from transformers import BertModel


transformers.logging.set_verbosity_error()


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class twoHeadsModel(torch.nn.Module):
    def __init__(
        self,
        num_labels,
        num_domain_labels,
        pretrained="bert-base-uncased",
        hidden_dropout_prob=0.1,
        grad_reverse=False,
    ):
        super(twoHeadsModel, self).__init__()

        self.pretrained = pretrained
        self.hidden_dropout_prob = hidden_dropout_prob

        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)
        self.bert = BertModel.from_pretrained(self.pretrained, use_auth_token=True)
        self.hidden_size = self.bert.pooler.dense.out_features
        self.main_classifier_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=num_labels, bias=True
        )
        self.domain_classifier_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=num_domain_labels, bias=True
        )
        self.grad_reverse = grad_reverse

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs_bert = self.bert(input_ids, token_type_ids, attention_mask)

        outputs_pooler = outputs_bert["pooler_output"]

        outputs_pooler = self.dropout(outputs_pooler)

        # into TWO classifiers
        outputs_main_classifier = self.main_classifier_layer(outputs_pooler)

        if self.grad_reverse:
            # outputs_pooler = grad_reverse(outputs_pooler)
            outputs_pooler_apply_reverse = grad_reverse(outputs_pooler)
            outputs_domain_classifier = self.domain_classifier_layer(
                outputs_pooler_apply_reverse
            )
        else:
            outputs_domain_classifier = self.domain_classifier_layer(outputs_pooler)

        return {
            "outputs_main_classifier": outputs_main_classifier,
            "outputs_domain_classifier": outputs_domain_classifier,
        }


class GradientReverseModel:
    def __init__(
        self,
        pretrained="bert-base-uncased",
        max_length=50,
        num_labels=2,
        num_domain_labels=None,
        hidden_dropout_prob=0.1,
        num_epochs=10,
        num_warmup_steps=0,
        batch_size=50,
        lr=5e-5,
        balance_weights=False,
        grad_norm=1.0,
        grad_reverse=False,
    ):

        """init

        Args:
            pretrained (str, optional): pretrained language model from huggingface. Defaults to "bert-base-uncased".
            max_length (int, optional): max length for tokenizer. Defaults to 50.
            num_labels (int, optional): number of classes for Main Label. Defaults to 2.
            num_domain_labels (int, optional): number of classes for Domain (Secondary) Label. Defaults to None.
            num_epochs (int, optional): training epochs. Defaults to 10.
            num_warmup_steps (int, optional): optimizer warmup. Defaults to 0.
            batch_size (int, optional): batch size. Defaults to 50.
            lr (float, optional): learning rate. Defaults to 5e-5.
            balance_weights (bool, optional): whether to balance weights. Defaults to False.
            grad_norm (float, optional): gradient norm clip. Defaults to 1.0.
            grad_reverse (bool, optional): whether to use Gradient Reversal method (by injecting a Gradient Reversal Layer). Defaults to False.
        """

        self.pretrained = pretrained
        self.max_length = max_length
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_epochs = num_epochs
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.lr = lr
        self.balance_weights = balance_weights
        self.grad_norm = grad_norm
        self.model = None
        self.grad_reverse = grad_reverse

        # assert domain_labels is not None
        self.num_domain_labels = num_domain_labels
        self.trainMainEpochLossAvg = []
        self.trainDomainEpochLossAvg = []

    def load_pretrained(self):

        # load pretrained model
        # instantiate model
        self.model = twoHeadsModel(
            num_labels=self.num_labels,
            num_domain_labels=self.num_domain_labels,
            pretrained=self.pretrained,
            hidden_dropout_prob=self.hidden_dropout_prob,
            grad_reverse=self.grad_reverse,
        )

    def trainModel(self, X, y, y_domain, device=None):
        """_summary_

        Args:
            X (pandas.Series, or numpy.array, 1D): 1D-array (pandas or numpy or list) containing X
            y (pandas.Series): main outcome(s). Could be multiple dimensions. For each column, y should be indices after conversion from original categories, e.g., [0,2,1,1,...], INSTEAD of ["High","Medium","Low","Low",...]
            y_domain (pandas.Series): secondary outcome(s). In the same format as of `y`
            device (_type_, optional): cuda or cpu. Defaults to None.
        """

        dataset = TransformerDataset(
            X=X,
            y=y,
            y_domain=y_domain,
            pretrained=self.pretrained,
            max_length=self.max_length,
        )

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        self.model.to(device)
        # logging.info(f"Model:\n{self.model}")

        # define optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        # create scheduler
        num_training_steps = self.num_epochs * len(dataloader)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if self.balance_weights:
            class_weights = len(y) / y.sum(axis=0)
            class_weights = class_weights.to_list()
            class_weights = [min(w, 10000) for w in class_weights]
        else:
            class_weights = [1.0] * self.num_labels

        # criterion = BCEWithLogitsLoss(
        #     pos_weight=torch.tensor(class_weights, device=device)
        # )

        criterion = CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

        # progress_bar = tqdm(range(num_training_steps))

        self.model.train()

        # iterate over epochs
        for epoch in range(self.num_epochs):

            loss_epoch_main = 0
            loss_epoch_domain = 0
            n_train = 0
            # iterate over batches
            for batch in dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}

                outputs_all = self.model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                )

                # calculate custom loss
                _loss_main = criterion(
                    outputs_all["outputs_main_classifier"],
                    batch["labels"].squeeze(1).type(torch.long),
                )
                _loss_domain = criterion(
                    outputs_all["outputs_domain_classifier"],
                    batch["domain_labels"].squeeze(1).type(torch.long),
                )
                loss = _loss_main + _loss_domain

                # back propagate loss and clip gradients
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_norm)

                # update loss plot
                loss_epoch_main += _loss_main.item() * len(batch)
                loss_epoch_domain += _loss_domain.item() * len(batch)
                n_train += len(batch)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)

            self.trainMainEpochLossAvg.append(loss_epoch_main/n_train)
            self.trainDomainEpochLossAvg.append(loss_epoch_domain/n_train)

    def trainModelWithTest(
        self, X, y, y_domain_train, X_test, y_test, y_domain_test, device=None
    ):

        dataset = TransformerDataset(
            X=X,
            y=y,
            y_domain=y_domain_train,
            pretrained=self.pretrained,
            max_length=self.max_length,
        )

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        dataset_test = TransformerDataset(
            X=X_test,
            y=y_test,
            y_domain=y_domain_test,
            pretrained=self.pretrained,
            max_length=self.max_length,
        )

        dataloader_test = DataLoader(
            dataset_test, shuffle=False, batch_size=self.batch_size
        )

        self.model.to(device)
        # logging.info(f"Model:\n{self.model}")

        # define optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        # create scheduler
        num_training_steps = self.num_epochs * len(dataloader)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if self.balance_weights:
            class_weights = len(y) / y.sum(axis=0)
            class_weights = class_weights.to_list()
            class_weights = [min(w, 10000) for w in class_weights]
        else:
            class_weights = [1.0] * self.num_labels

        criterion = CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
        criterion_test = CrossEntropyLoss()

        # progress_bar = tqdm(range(num_training_steps))

        self.loss_epochs = []
        self.loss_steps = []
        self.loss_test_main_epochs = []
        self.loss_test_domain_epochs = []
        self.loss_test_total_epochs = []

        # iterate over epochs
        for epoch in range(self.num_epochs):
            self.model.train()

            loss_epoch = 0

            # iterate over batches
            for batch in dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}

                outputs_all = self.model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                )

                # calculate custom loss
                loss = criterion(
                    outputs_all["outputs_main_classifier"],
                    batch["labels"].squeeze(1).type(torch.long),
                ) + criterion(
                    outputs_all["outputs_domain_classifier"],
                    batch["domain_labels"].squeeze(1).type(torch.long),
                )

                # back propagate loss and clip gradients
                self.loss_steps.append(loss.item())
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_norm)

                # update loss plot
                loss_epoch += loss.item()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)

            self.loss_epochs.append(loss_epoch)

            # test loss
            self.model.eval()

            loss_test_main_epoch = 0
            loss_test_domain_epoch = 0

            with torch.no_grad():
                for batch in dataloader_test:

                    batch = {k: v.to(device) for k, v in batch.items()}

                    outputs_all = self.model(
                        input_ids=batch["input_ids"],
                        token_type_ids=batch["token_type_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    # calculate custom loss
                    loss_test_main = criterion(
                        outputs_all["outputs_main_classifier"],
                        batch["labels"].squeeze(1).type(torch.long),
                    )

                    loss_test_domain = criterion(
                        outputs_all["outputs_domain_classifier"],
                        batch["domain_labels"].squeeze(1).type(torch.long),
                    )

                    loss_test_main_epoch += loss_test_main.item()
                    loss_test_domain_epoch += loss_test_domain.item()

            self.loss_test_main_epochs.append(loss_test_main_epoch)
            self.loss_test_domain_epochs.append(loss_test_domain_epoch)
            self.loss_test_total_epochs.append(
                loss_test_main_epoch + loss_test_domain_epoch
            )

    def predict(self, X, device=None):

        dataset = TransformerDataset(
            X=X,
            pretrained=self.pretrained,
            max_length=self.max_length,
        )

        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)

        self.model.to(device)
        # logging.info(f"Model:\n{self.model}")

        # progress_bar = tqdm(range(len(dataloader)))

        self.model.eval()

        y_main_pred = []
        y_main_prob = []

        y_domain_pred = []
        y_domain_prob = []

        # iterate over batches
        for batch in dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs_all = self.model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                )

            # get logits
            logits_main = outputs_all["outputs_main_classifier"]
            logits_domain = outputs_all["outputs_domain_classifier"]

            # get probabilities from logits
            softmax = torch.nn.Softmax(dim=1)
            probs_main = softmax(logits_main)
            probs_domain = softmax(logits_domain)

            # get predictions from probabilities
            predictions_main = probs_main.max(axis=1).indices.cpu().detach().numpy()
            y_main_pred.append(predictions_main)
            y_main_prob.append(probs_main.cpu().detach().numpy())

            predictions_domain = probs_domain.max(axis=1).indices.cpu().detach().numpy()
            y_domain_pred.append(predictions_domain)
            y_domain_prob.append(probs_domain.cpu().detach().numpy())

            # progress_bar.update(1)

        # concatenate predictions
        y_main_pred = np.concatenate(y_main_pred, axis=0)
        y_main_prob = np.concatenate(y_main_prob, axis=0)

        y_domain_pred = np.concatenate(y_domain_pred, axis=0)
        y_domain_prob = np.concatenate(y_domain_prob, axis=0)

        # package predictions in data frame
        # y_main_pred = pd.DataFrame(y_main_pred, columns=self.labels)
        # y_main_prob = pd.DataFrame(y_main_prob, columns=self.labels)
        y_main_pred = pd.DataFrame(y_main_pred)
        y_main_prob = pd.DataFrame(y_main_prob)

        # y_domain_pred = pd.DataFrame(y_domain_pred, columns=self.domain_labels)
        # y_domain_prob = pd.DataFrame(y_domain_prob, columns=self.domain_labels)
        y_domain_pred = pd.DataFrame(y_domain_pred)
        y_domain_prob = pd.DataFrame(y_domain_prob)

        return y_main_pred, y_main_prob, y_domain_pred, y_domain_prob
