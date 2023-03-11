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
from NeuralModel import TransformerDataset

import transformers
from transformers import BertModel


transformers.logging.set_verbosity_error()


class baseModel(torch.nn.Module):
    def __init__(
        self,
        num_labels,
        num_zCats, 
        pretrained="bert-base-uncased",
        hidden_dropout_prob=0.1,

    ):
        super(baseModel, self).__init__()

        self.pretrained = pretrained
        self.hidden_dropout_prob = hidden_dropout_prob

        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)
        self.bert = BertModel.from_pretrained(self.pretrained, use_auth_token=True)
        self.hidden_size = self.bert.pooler.dense.out_features
        self.main_classifier_layer = torch.nn.Linear(
            in_features=(self.hidden_size + num_zCats), out_features=num_labels, bias=True
        )
       
       
    def forward(self, input_ids, token_type_ids, attention_mask, confound_dummies):
        outputs_bert = self.bert(input_ids, token_type_ids, attention_mask)

        outputs_pooler = outputs_bert["pooler_output"]

        outputs_pooler = self.dropout(outputs_pooler)

        outputs_extend_confounds = torch.cat([outputs_pooler, confound_dummies], 1)

        outputs_main_classifier = self.main_classifier_layer(outputs_pooler)  #shape: (N, 2)



        return {
            "outputs_main_classifier": outputs_main_classifier,

        }


class backdoorAdjustBERTModel:
    def __init__(
        self,
        pretrained="bert-base-uncased",
        max_length=50,
        num_labels=2,
        hidden_dropout_prob=0.1,
        num_epochs=10,
        num_warmup_steps=0,
        batch_size=50,
        lr=5e-5,
        balance_weights=False,
        grad_norm=1.0,
        
    ):

        """init

        Args:
            pretrained (str, optional): pretrained language model from huggingface. Defaults to "bert-base-uncased".
            max_length (int, optional): max length for tokenizer. Defaults to 50.
            num_labels (int, optional): number of classes for Main Label. Defaults to 2.
            num_epochs (int, optional): training epochs. Defaults to 10.
            num_warmup_steps (int, optional): optimizer warmup. Defaults to 0.
            batch_size (int, optional): batch size. Defaults to 50.
            lr (float, optional): learning rate. Defaults to 5e-5.
            balance_weights (bool, optional): whether to balance weights. Defaults to False.
            grad_norm (float, optional): gradient norm clip. Defaults to 1.0.
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

        self.trainMainEpochLossAvg = []


    def load_pretrained(self):

        # load pretrained model
        # instantiate model
        self.model = baseModel(
            num_labels=self.num_labels,
            pretrained=self.pretrained,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def trainModel(self, X, y, device=None):
        """_summary_

        Args:
            X (pandas.Series, or numpy.array, 1D): 1D-array (pandas or numpy or list) containing X
            y (pandas.Series): main outcome(s). Could be multiple dimensions. For each column, y should be indices after conversion from original categories, e.g., [0,2,1,1,...], INSTEAD of ["High","Medium","Low","Low",...]
            device (_type_, optional): cuda or cpu. Defaults to None.
        """

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

        # criterion = BCEWithLogitsLoss(
        #     pos_weight=torch.tensor(class_weights, device=device)
        # )

        criterion = CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

        # progress_bar = tqdm(range(num_training_steps))

        self.model.train()

        # iterate over epochs
        for epoch in range(self.num_epochs):

            loss_epoch_main = 0
            
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
               
                loss = _loss_main

                # back propagate loss and clip gradients
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_norm)

                # update loss plot
                loss_epoch_main += _loss_main.item() * len(batch)
               
                n_train += len(batch)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)

            self.trainMainEpochLossAvg.append(loss_epoch_main/n_train)
           
    def trainModelWithTest(
        self, X, y, X_test, y_test, device=None
    ):

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
        self.loss_test_main_epochs = []
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
                    batch["labels"].squeeze(1).type(torch.long),)
               

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


                    loss_test_main_epoch += loss_test_main.item()
                   

            self.loss_test_main_epochs.append(loss_test_main_epoch)
        

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


            # get probabilities from logits
            softmax = torch.nn.Softmax(dim=1)
            probs_main = softmax(logits_main)


            # get predictions from probabilities
            predictions_main = probs_main.max(axis=1).indices.cpu().detach().numpy()
            y_main_pred.append(predictions_main)
            y_main_prob.append(probs_main.cpu().detach().numpy())


            # progress_bar.update(1)

        # concatenate predictions
        y_main_pred = np.concatenate(y_main_pred, axis=0)
        y_main_prob = np.concatenate(y_main_prob, axis=0)

        # package predictions in data frame
        # y_main_pred = pd.DataFrame(y_main_pred, columns=self.labels)
        # y_main_prob = pd.DataFrame(y_main_prob, columns=self.labels)
        y_main_pred = pd.DataFrame(y_main_pred)
        y_main_prob = pd.DataFrame(y_main_prob)

        return y_main_pred, y_main_prob
