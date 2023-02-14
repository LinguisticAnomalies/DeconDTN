import pandas as pd
import numpy as np

# from tqdm import tqdm

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from src.MultiLabel import TransformerDataset
from tqdm.notebook import trange
transformers.logging.set_verbosity_error()

# TODO: num_labels (1 or 2? how will it affect AutoModelForSequenceClassification in config),
# calculate loss, criterion


class MultiClassificationNN:
    def __init__(
        self,
        pretrained="bert-base-uncased",
        max_length=50,
        labels=None,
        num_labels=2,
        hidden_dropout_prob=0.1,
        num_epochs=10,
        num_warmup_steps=0,
        batch_size=50,
        lr=5e-5,
        balance_weights=False,
        grad_norm=1.0,
        problem_type="single_label_classification",
    ):

        self.pretrained = pretrained
        self.max_length = max_length
        self.labels = labels
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_epochs = num_epochs
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.lr = lr
        self.balance_weights = balance_weights
        self.grad_norm = grad_norm
        self.problem_type = problem_type
        self.model = None

    def load_pretrained(self):

        # load pretrained model
        # load default configuration
        config = AutoConfig.from_pretrained(self.pretrained)

        # update default configuration
        config.problem_type = self.problem_type
        config.num_labels = self.num_labels
        config.hidden_dropout_prob = self.hidden_dropout_prob

        # instantiate model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained, config=config
        )

    def trainModel(self, X, y, device=None):

        dataset = TransformerDataset(
            X=X,
            y=y,
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

        criterion = CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

        # progress_bar = tqdm(range(num_training_steps))

        self.model.train()

        # iterate over epochs
        pbar = trange(self.num_epochs, desc = "Training Epoch")
        for epoch in pbar:
            loss_epoch = 0

            # iterate over batches
            for batch in dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}
                batch["labels"] = batch["labels"].type(torch.long)

                outputs = self.model(**batch)

                # calculate custom loss
                loss = criterion(outputs.logits, batch["labels"].squeeze(1))

                # back propagate loss and clip gradients
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_norm)

                # update loss plot
                loss_epoch += loss.item()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)

    def trainModelWithTest(self, X, y, X_test, y_test, device=None):

        dataset = TransformerDataset(
            X=X,
            y=y,
            pretrained=self.pretrained,
            max_length=self.max_length,
        )

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        dataset_test = TransformerDataset(
            X=X_test,
            y=y_test,
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
        self.loss_test_epochs = []

        # iterate over epochs
        for epoch in trange(self.num_epochs, desc = "Training Epoch"):
            self.model.train()

            loss_epoch = 0

            # iterate over batches
            for batch in dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}
                batch["labels"] = batch["labels"].type(torch.long)

                outputs = self.model(**batch)

                # calculate custom loss
                loss = criterion(outputs.logits, batch["labels"].squeeze(1))

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

            loss_test_epoch = 0

            with torch.no_grad():
                for batch in dataloader_test:

                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch["labels"] = batch["labels"].type(torch.long)

                    outputs = self.model(**batch)

                    # calculate custom loss
                    loss_test = criterion_test(
                        outputs.logits, batch["labels"].squeeze(1)
                    )
                    loss_test_epoch += loss_test.item()

            self.loss_test_epochs.append(loss_test_epoch)

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

        y_pred = []
        y_prob = []

        # iterate over batches
        for batch in dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            # get logits
            logits = outputs.logits

            # get probabilities from logits
            softmax = torch.nn.LogSoftmax(dim=1)
            probs = softmax(logits)

            # get predictions from probabilities
            predictions = probs.max(axis=1).indices.cpu().detach().numpy()

            y_pred.append(predictions)
            y_prob.append(probs.cpu().detach().numpy())

            # progress_bar.update(1)

        # concatenate predictions
        y_pred = np.concatenate(y_pred, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)

        # package predictions in data frame
        y_pred = pd.DataFrame(y_pred, columns=self.labels)
        y_prob = pd.DataFrame(y_prob, columns=self.labels)

        return y_pred, y_prob
