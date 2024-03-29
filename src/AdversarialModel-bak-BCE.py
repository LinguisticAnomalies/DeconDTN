import pandas as pd
import numpy as np

# from tqdm import tqdm

import torch
import transformers
from transformers import BertModel
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler

# from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from src.NeuralModel import TransformerDataset

transformers.logging.set_verbosity_error()


class twoHeadsModel(torch.nn.Module):
    def __init__(
        self,
        num_labels,
        num_domain_labels,
        pretrained="bert-base-uncased",
        hidden_dropout_prob=0.1,
    ):
        super(twoHeadsModel, self).__init__()

        self.pretrained = pretrained
        self.hidden_dropout_prob = hidden_dropout_prob

        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)
        self.bert = BertModel.from_pretrained(self.pretrained)
        self.hidden_size = self.bert.pooler.dense.out_features
        self.main_classifier_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=num_labels, bias=True
        )
        self.domain_classifier_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=num_domain_labels, bias=True
        )


    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs_bert = self.bert(input_ids, token_type_ids, attention_mask)

        outputs_pooler = outputs_bert["pooler_output"]

        outputs_pooler = self.dropout(outputs_pooler)

        # into TWO classifiers
        outputs_main_classifier = self.main_classifier_layer(outputs_pooler)
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
        labels=None,
        num_labels=1,
        hidden_dropout_prob=0.1,
        num_epochs=10,
        num_warmup_steps=0,
        batch_size=50,
        lr=5e-5,
        balance_weights=False,
        grad_norm=1.0,
        domain_labels=None,
    ):
        """
        domain_labels: list, use column header: ["WLS"] or ["WLS", "ADRESS"] or ["ADRESS"]. 0/1 coded in each column
        """
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

        # self.problem_type = "multi_label_classification"
        #         self.overwritable_params = ["num_epochs", "hidden_dropout_prob"]
        self.model = None

        assert domain_labels is not None
        self.num_domain_labels = len(domain_labels)

    def load_pretrained(self):

        # load pretrained model
        # instantiate model
        self.model = twoHeadsModel(
            num_labels=self.num_labels,
            num_domain_labels=self.num_domain_labels,
            pretrained=self.pretrained,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def trainModel(self, X, y, y_domain, device=None):
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
                    outputs_all["outputs_main_classifier"], batch["labels"]
                ) + criterion(
                    outputs_all["outputs_domain_classifier"],
                    batch["domain_labels"],
                )
                # back propagate loss and clip gradients
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_norm)

                # update loss plot
                loss_epoch += loss.item()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)

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
                    outputs_all["outputs_main_classifier"], batch["labels"]
                ) + criterion(
                    outputs_all["outputs_domain_classifier"],
                    batch["domain_labels"],
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
                        outputs_all["outputs_main_classifier"], batch["labels"]
                    )

                    loss_test_domain = criterion(
                        outputs_all["outputs_domain_classifier"],
                        batch["domain_labels"],
                    )

                    loss_test_main_epoch += loss_test_main.item()
                    loss_test_domain_epoch += loss_test_domain.item()

            self.loss_test_main_epochs.append(loss_test_main_epoch)
            self.loss_test_domain_epochs.append(loss_test_domain_epoch)
            self.loss_test_total_epochs.append(
                loss_test_main_epoch + loss_test_domain_epoch
            )

    # TODO: check
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
            probs_main = torch.sigmoid(logits_main)
            probs_domain = torch.sigmoid(logits_domain)

            # get predictions from probabilities
            predictions_main = (
                (probs_main >= 0.5).type(torch.LongTensor).detach().numpy()
            )
            y_main_pred.append(predictions_main)
            y_main_prob.append(probs_main.cpu().detach().numpy())

            predictions_domain = (
                (probs_domain >= 0.5).type(torch.LongTensor).detach().numpy()
            )
            y_domain_pred.append(predictions_domain)
            y_domain_prob.append(probs_domain.cpu().detach().numpy())

            # progress_bar.update(1)

        # concatenate predictions
        y_main_pred = np.concatenate(y_main_pred, axis=0)
        y_main_prob = np.concatenate(y_main_prob, axis=0)

        y_domain_pred = np.concatenate(y_domain_pred, axis=0)
        y_domain_prob = np.concatenate(y_domain_prob, axis=0)

        # package predictions in data frame
        y_main_pred = pd.DataFrame(y_main_pred, columns=self.labels)
        y_main_prob = pd.DataFrame(y_main_prob, columns=self.labels)

        y_domain_pred = pd.DataFrame(y_domain_pred, columns=self.domain_labels)
        y_domain_prob = pd.DataFrame(y_domain_prob, columns=self.domain_labels)

        return y_main_pred, y_main_prob, y_domain_pred, y_domain_prob
