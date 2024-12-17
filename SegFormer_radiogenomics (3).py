#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers


# In[2]:


pip install pytorch-lightning==1.9.4


# In[3]:


pip install datasets


# In[4]:


pip install PIL


# In[5]:


pip install torchvision


# In[6]:


from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
class SegmentationAndClassificationDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, mutation_status_mapping):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.patient_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        self.image_paths = []
        self.mask_paths = []
        self.classification_labels = {}
        self.id2label = {0: "background", 1: "lung"}

        # # Preprocessing transformations
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize to fixed size if needed
        #     transforms.ToTensor(),  # Convert to tensor
        #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        # ])

        self._load_mutation_status(mutation_status_mapping)
        for patient_dir in self.patient_dirs:
            scans_dir = os.path.join(self.root_dir, patient_dir, 'scans')
            masks_dir = os.path.join(self.root_dir, patient_dir, 'masks')
            if not os.path.exists(scans_dir) or not os.path.exists(masks_dir):
                continue

            scan_files = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith('.dcm')]
            mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.dcm')]
            if len(scan_files) == len(mask_files):
                self.image_paths.extend(scan_files)
                self.mask_paths.extend(mask_files)

    def _load_mutation_status(self, mutation_status_mapping):
        for patient_id, labels in mutation_status_mapping.items():
            self.classification_labels[patient_id] = labels
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

        image = pydicom.dcmread(image_path).pixel_array
        mask = pydicom.dcmread(mask_path).pixel_array
        image = (image / np.max(image) * 255).astype(np.uint8)
        image_rgb = np.stack((image,) * 3, axis=-1)
        image_rgb = Image.fromarray(image_rgb)

        # image_rgb = self.transform(image_rgb)  # Apply transform
        mask = (mask > 0).astype(np.uint8)
        mask = Image.fromarray(mask, mode='L')
        # mask = self.transform(mask)

        classification_label = self.classification_labels[patient_id]
        encoded_inputs = self.feature_extractor(image_rgb, mask, return_tensors="pt")
        
        return {
            'pixel_values': encoded_inputs['pixel_values'].squeeze(),
            'segmentation_labels': encoded_inputs['labels'].squeeze(),
            'classification_labels': {
                'EGFR': torch.tensor(classification_label['EGFR'], dtype=torch.long),
                'KRAS': torch.tensor(classification_label['KRAS'], dtype=torch.long)
            }
        }


# In[7]:


class SegmentationAndClassificationDataset(Dataset):
    """Dataset for both image segmentation and classification."""

    def __init__(self, root_dir, feature_extractor, mutation_status_mapping):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.patient_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        self.image_paths = []
        self.mask_paths = []
        self.classification_labels = {}  # Store mutation statuses
        self.id2label = {0: "background", 1: "lung"}

        # Load mutation statuses for each patient
        self._load_mutation_status(mutation_status_mapping)

        # Collect paths for all images and masks
        for patient_dir in self.patient_dirs:
            scans_dir = os.path.join(self.root_dir, patient_dir, 'scans')
            masks_dir = os.path.join(self.root_dir, patient_dir, 'masks')

            if not os.path.exists(scans_dir) or not os.path.exists(masks_dir):
                continue

            scan_files = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith('.dcm')]
            mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.dcm')]

            # Ensure there is a matching number of scans and masks
            if len(scan_files) == len(mask_files):
                scan_files_reversed = list(reversed(scan_files))
                self.image_paths.extend(scan_files_reversed)
                self.mask_paths.extend(mask_files)

    def _load_mutation_status(self, mutation_status_mapping):
        """Loads mutation statuses directly into classification labels."""
        for patient_id, labels in mutation_status_mapping.items():
            # Expecting labels to be a dictionary with 'EGFR' and 'KRAS'
            self.classification_labels[patient_id] = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(image_path)))  # Get the patient ID from the file path

        # Load the DICOM image and mask using pydicom
        image = pydicom.dcmread(image_path).pixel_array
        mask = pydicom.dcmread(mask_path).pixel_array

        # Normalize and convert to RGB for the image
        image = (image / np.max(image) * 255).astype(np.uint8)  # Normalize to range [0, 255]
        image_rgb = np.stack((image,) * 3, axis=-1)  # Stack to create RGB channels
        image_rgb = Image.fromarray(image_rgb)  # Convert to PIL Image

        # Convert mask to binary format (0 for background, 1 for lung)
        mask = (mask > 0).astype(np.uint8)
        mask = Image.fromarray(mask, mode='L')

        # Get classification label for this patient
        classification_label = self.classification_labels[patient_id]

        # Apply feature extractor
        encoded_inputs = self.feature_extractor(image_rgb, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = torch.squeeze(v, 0)



        # Debug: Print keys in encoded_inputs
        

        # Prepare the output dictionary
        return {
            'pixel_values': encoded_inputs['pixel_values'].squeeze(),
            'segmentation_labels': encoded_inputs['labels'].squeeze(),  # Using 'labels'
            'classification_labels': {
                'EGFR': torch.tensor(classification_label['EGFR'], dtype=torch.long),
                'KRAS': torch.tensor(classification_label['KRAS'], dtype=torch.long)
            }
        }


# In[8]:


pip install pydicom


# In[9]:


import pandas as pd
import numpy as np
import pydicom
import os


# In[10]:


train_directory ="E:\\NSCLC\\train"


# In[11]:


test_directory="E:\\NSCLC\\test"


# In[12]:


df= pd.read_csv("clinical.csv")


# In[13]:


patient_folders = [folder for folder in os.listdir(train_directory) if os.path.isdir(os.path.join(train_directory, folder))]

patient_id_mapping = {folder.split('-')[0] + '-' + folder.split('-')[1]: folder for folder in patient_folders}



# In[14]:


df = df[df['Case ID'].isin(patient_id_mapping.keys())]
train_mutation_status_mapping = {}

# Map mutation statuses to numerical values
status_mapping = {'Unknown': 0, 'Wildtype': 1, 'Mutant': 2}

# Populate the mapping for each patient
for index, row in df.iterrows():
    short_patient_id = row['Case ID']
    full_patient_id = patient_id_mapping.get(short_patient_id)
    
    egfr_status = row['EGFR mutation status']
    kras_status = row['KRAS mutation status']
    
    # Map the statuses to their numerical values
    egfr_mapped = status_mapping.get(egfr_status, -1)  # Use -1 for unmapped statuses
    kras_mapped = status_mapping.get(kras_status, -1)  # Use -1 for unmapped statuses
    
    # Store the status in the mapping using the full patient ID
    train_mutation_status_mapping[full_patient_id] = {
        'EGFR': egfr_mapped,
        'KRAS': kras_mapped
    }

# Output the mapping to check
print("Mutation Status Mapping:", train_mutation_status_mapping)


# In[15]:


df= pd.read_csv("clinical.csv")


# In[16]:


test_patient_folders = [folder for folder in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, folder))]

test_patient_id_mapping = {folder.split('-')[0] + '-' + folder.split('-')[1]: folder for folder in test_patient_folders}


# In[17]:


test_patient_id_mapping


# In[18]:


df = df[df['Case ID'].isin(test_patient_id_mapping.keys())]
test_mutation_status_mapping = {}

# Map mutation statuses to numerical values
status_mapping = {'Unknown': 0, 'Wildtype': 1, 'Mutant': 2}

# Populate the mapping for each patient
for index, row in df.iterrows():
    short_patient_id = row['Case ID']
    full_patient_id = test_patient_id_mapping.get(short_patient_id)
    
    egfr_status = row['EGFR mutation status']
    kras_status = row['KRAS mutation status']
    
    # Map the statuses to their numerical values
    egfr_mapped = status_mapping.get(egfr_status, -1)  # Use -1 for unmapped statuses
    kras_mapped = status_mapping.get(kras_status, -1)  # Use -1 for unmapped statuses
    
    # Store the status in the mapping using the full patient ID
    test_mutation_status_mapping[full_patient_id] = {
        'EGFR': egfr_mapped,
        'KRAS': kras_mapped
    }

# Output the mapping to check
print("Mutation Status Mapping:", test_mutation_status_mapping)


# In[19]:


from transformers import SegformerFeatureExtractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.do_reduce_labels = False
feature_extractor.size = 128
train_directory ="E:\\NSCLC\\train"
train_dataset = SegmentationAndClassificationDataset("E:\\NSCLC\\train", feature_extractor,train_mutation_status_mapping)
# val_dataset = SemanticAndClassificationDataset("D:\\NSCLC\\validate", feature_extractor)
test_dataset = SegmentationAndClassificationDataset("E:\\NSCLC\\test", feature_extractor, test_mutation_status_mapping)


# In[20]:


train_dataset


# In[21]:


import pandas as pd
import numpy as np


# In[22]:


df.columns


# In[23]:


df["KRAS mutation status"].values


# In[24]:


pip install evaluate


# In[25]:


get_ipython().system('python.exe -m pip install --upgrade pip')


# In[23]:


import evaluate


# In[318]:


model1= SegformerForSemanticSegmentation


# In[26]:


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import SegformerPreTrainedModel, SegformerModel, SegformerDecodeHead
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional, Union, Tuple

class SegformerForSegmentationAndClassification(SegformerPreTrainedModel):
    def __init__(self, config, alpha=0.5, dropout_rate=0.3, l1_lambda=1e-5, l2_lambda=1e-4):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        self.egfr_classifier = nn.Linear(config.hidden_sizes[-1], 3)
        self.kras_classifier = nn.Linear(config.hidden_sizes[-1], 3)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        segmentation_labels: Optional[torch.FloatTensor] = None,
        classification_labels: Optional[dict] = None,  # Classification labels as a dict
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # Segmentation logits
        segmentation_logits = self.decode_head(encoder_hidden_states)
        # print("segmentation logits", segmentation_logits.shape)
        # print("--------------------------------------------------")
        # print("segmentation labels", segmentation_labels.shape)
        # print("--------------------------------------------------")
        
        # Classification logits (global pooling over the last hidden state)
        sequence_output = encoder_hidden_states[-1]
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            sequence_output = sequence_output.permute(0, 2, 3, 1).reshape(batch_size, -1, self.config.hidden_sizes[-1])
        sequence_output = sequence_output.mean(dim=1)  # Global average pooling
        sequence_output = self.dropout(sequence_output)  # Apply dropout
        # Separate classification for EGFR and KRAS
        egfr_logits = self.egfr_classifier(sequence_output)
        kras_logits = self.kras_classifier(sequence_output)
        # print("class EGFR logits")
        # print(egfr_logits)
        # print("-------------------------------------------------")
          # Apply softmax to get probabilities for each node
        egfr_probs = F.softmax(egfr_logits, dim=1)
        kras_probs = F.softmax(kras_logits, dim=1)
        
        # # Display the probability distribution for each batch element
        # print("EGFR Probabilities (after softmax):", egfr_probs)
        # print("KRAS Probabilities (after softmax):", kras_probs)
        
        # Show which node has the highest probability for each
        egfr_pred = torch.argmax(egfr_probs, dim=1)
        
        
        print("EGFR Predicted Class:", egfr_pred)
        print("EGFR Actual Class:", classification_labels["EGFR"])

        # Compute losses if labels are provided
        loss = None
        segmentation_loss = None
        classification_loss = None

        # Segmentation Loss Calculation
        if segmentation_labels is not None:
            if segmentation_labels.dim() == 3:  # [batch_size, height, width]
                upsampled_logits = nn.functional.interpolate(
                    segmentation_logits[:, 0:1, :, :], size=segmentation_labels.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)  # Now shape is [32, 128, 128]
                
                # Compute the loss
                loss_fct = torch.nn.BCEWithLogitsLoss()
                segmentation_loss = loss_fct(upsampled_logits, segmentation_labels.float()) 

        # Classification Loss Calculation
        if classification_labels is not None:
            egfr_labels = classification_labels["EGFR"]
            kras_labels = classification_labels["KRAS"]

            # Loss function for multi-class classification (CrossEntropyLoss)
            loss_fct = CrossEntropyLoss()

            # Ensure labels are LongTensors
            if egfr_labels.dtype != torch.long:
                egfr_labels = egfr_labels.long()
            if kras_labels.dtype != torch.long:
                kras_labels = kras_labels.long()

            # Calculate loss for both EGFR and KRAS
            egfr_loss = loss_fct(egfr_logits, egfr_labels)
            kras_loss = loss_fct(kras_logits, kras_labels)

            classification_loss = egfr_loss + kras_loss
        # print("segmentation loss in class")
        # print(segmentation_loss)
        # print("class loss in class")
        # print(classification_loss)
        # print("----------------------------------------------------")
        

        # Combined Loss Calculation
        if segmentation_loss is not None and classification_loss is not None:
            loss = segmentation_loss + self.alpha * classification_loss
        elif segmentation_loss is not None:
            loss = segmentation_loss
        elif classification_loss is not None:
            loss = classification_loss

        # print("total loss in class")
        # print(loss)
        # print("-------------------------------------------------------")
        
        # l1_reg = sum(torch.norm(param, 1) for param in self.parameters())
        # loss += self.l1_lambda * l1_reg

        # # L2 regularization
        # l2_reg = sum(torch.norm(param, 2) ** 2 for param in self.parameters())
        # loss += self.l2_lambda * l2_reg
        if not return_dict:
            return (loss, segmentation_logits, egfr_logits, kras_logits) if loss is not None else (segmentation_logits, egfr_logits, kras_logits)

        
      
        # Return SemanticSegmenterOutput with classification logits
        return CustomSemanticSegmenterOutput(
            loss=loss,
            logits=segmentation_logits,
            egfr_logits=egfr_logits,  # Return EGFR logits
            kras_logits=kras_logits,  # Return KRAS logits
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )




# In[27]:


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import SegformerPreTrainedModel, SegformerModel, SegformerDecodeHead
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional, Union, Tuple

class SegformerForSegmentationAndClassification(SegformerPreTrainedModel):
    def __init__(self, config, alpha=0.5, dropout_rate=0.3, l1_lambda=1e-5, l2_lambda=1e-4, B=1.0):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        self.egfr_classifier = nn.Linear(config.hidden_sizes[-1], 3)
        self.kras_classifier = nn.Linear(config.hidden_sizes[-1], 3)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.B = B  # Regularization scaling factor

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        segmentation_labels: Optional[torch.FloatTensor] = None,
        classification_labels: Optional[dict] = None,  # Classification labels as a dict
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # Segmentation logits
        segmentation_logits = self.decode_head(encoder_hidden_states)

        # Classification logits (global pooling over the last hidden state)
        sequence_output = encoder_hidden_states[-1]
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            sequence_output = sequence_output.permute(0, 2, 3, 1).reshape(batch_size, -1, self.config.hidden_sizes[-1])
        sequence_output = sequence_output.mean(dim=1)  # Global average pooling
        if self.training:  # Apply dropout only during training
            sequence_output = self.dropout(sequence_output)
        
        # Separate classification for EGFR and KRAS
        egfr_logits = self.egfr_classifier(sequence_output)
        kras_logits = self.kras_classifier(sequence_output)
        egfr_probs = F.softmax(egfr_logits, dim=1)
        kras_probs = F.softmax(kras_logits, dim=1)
        
        # # Display the probability distribution for each batch element
        # print("EGFR Probabilities (after softmax):", egfr_probs)
        # print("KRAS Probabilities (after softmax):", kras_probs)
        
        # Show which node has the highest probability for each
        egfr_pred = torch.argmax(egfr_probs, dim=1)
        
        
        # print("EGFR Predicted Class:", egfr_pred)
        # print("EGFR Actual Class:", classification_labels["EGFR"])

        # Compute losses if labels are provided
        loss = None
        segmentation_loss = None
        classification_loss = None

        # Segmentation Loss Calculation
        if segmentation_labels is not None:
            if segmentation_labels.dim() == 3:  # [batch_size, height, width]
                upsampled_logits = nn.functional.interpolate(
                    segmentation_logits[:, 0:1, :, :], size=segmentation_labels.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)  # Now shape is [batch_size, height, width]

                loss_fct = torch.nn.BCEWithLogitsLoss()
                segmentation_loss = loss_fct(upsampled_logits, segmentation_labels.float())
        # print(segmentation_logits.shape)


        # Classification Loss Calculation
        if classification_labels is not None:
            egfr_labels = classification_labels["EGFR"]
            kras_labels = classification_labels["KRAS"]

            # Loss function for multi-class classification (CrossEntropyLoss)
            loss_fct = CrossEntropyLoss()

            # Ensure labels are LongTensors
            if egfr_labels.dtype != torch.long:
                egfr_labels = egfr_labels.long()
            if kras_labels.dtype != torch.long:
                kras_labels = kras_labels.long()

            # Calculate loss for both EGFR and KRAS
            egfr_loss = loss_fct(egfr_logits, egfr_labels)
            kras_loss = loss_fct(kras_logits, kras_labels)

            classification_loss = egfr_loss + kras_loss

        # Combined Loss Calculation
        if segmentation_loss is not None and classification_loss is not None:
            loss = segmentation_loss + self.alpha * classification_loss
        elif segmentation_loss is not None:
            loss = segmentation_loss
        elif classification_loss is not None:
            loss = classification_loss

        # Regularization (apply only during training)
        if self.training:
            # # L1 regularization
            # l1_reg = sum(torch.norm(param, 1) for param in self.parameters())
            # loss += self.B * self.l1_lambda * l1_reg

            # L2 regularization
            l2_reg = sum(torch.norm(param, 2) ** 2 for param in self.parameters())
            loss += self.B * self.l2_lambda * l2_reg

        if not return_dict:
            return (loss, segmentation_logits, egfr_logits, kras_logits) if loss is not None else (segmentation_logits, egfr_logits, kras_logits)

        return CustomSemanticSegmenterOutput(
            loss=loss,
            logits=segmentation_logits,
            egfr_logits=egfr_logits,  # Return EGFR logits
            kras_logits=kras_logits,  # Return KRAS logits
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


# In[28]:


from transformers import SegformerConfig 


# In[29]:


config=SegformerConfig()


# In[30]:


import torch.nn.functional as F


# In[31]:


train_dataset[400]


# In[32]:


from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional, Tuple

class CustomSemanticSegmenterOutput(SemanticSegmenterOutput):
    def __init__(self, loss: Optional[torch.FloatTensor] = None, 
                 logits: torch.FloatTensor = None,
                 egfr_logits: torch.FloatTensor = None, 
                 kras_logits: torch.FloatTensor = None,
                 hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
                 attentions: Optional[Tuple[torch.FloatTensor]] = None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.egfr_logits = egfr_logits
        self.kras_logits = kras_logits


# In[33]:


import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SegformerConfig
import matplotlib.pyplot as plt
from evaluate import load


class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, alpha=0.5, train_dataloader=None, metrics_interval=10):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.alpha = alpha  # Hyperparameter for combined loss
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        
        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Using the new class `SegformerForSegmentationAndClassification`
        self.model = SegformerForSegmentationAndClassification(config=SegformerConfig(
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        )
        self.batch_segmentation_losses = []
        self.batch_classification_losses = []
        self.batch_total_losses = []
        
        self.epoch_segmentation_losses = []
        self.epoch_classification_losses = []
        self.epoch_total_losses = []


    def forward(self, images, segmentation_labels=None, classification_labels=None):
        # Forward pass through your new model
        # print(self.model) # what is this?
        outputs = self.model(pixel_values=images, segmentation_labels=segmentation_labels, classification_labels=classification_labels)
        return outputs
    

    def training_step(self, batch, batch_nb):

        images = batch['pixel_values']
        segmentation_labels = batch['segmentation_labels']
        classification_labels = batch['classification_labels']
    
        outputs = self(images, segmentation_labels, classification_labels)
        print("checking egfr logits")
        print("--------------------------------------------------")
        
        # Extract losses
        total_loss= outputs.loss
        segmentation_logits = outputs.logits  # From your model's output
        # Use the correct logits for classification
        egfr_logits = outputs.egfr_logits  # Extract EGFR logits
        kras_logits = outputs.kras_logits  # Extract KRAS logits
        # print("----------------------------")
        # print("seg and class loss here")
        classification_loss = self.compute_classification_loss(egfr_logits, kras_logits, classification_labels)
        segmentation_loss= self.compute_segmentation_loss(segmentation_logits, segmentation_labels)
        print("Total loss with regularization and dropout")
        print(total_loss)
        print("-----------------------------------")
        
        # Log individual losses
        self.log("train_segmentation_loss", segmentation_loss)
        self.log("train_classification_loss", classification_loss)
        self.log("train_total_loss", total_loss)
        
        self.batch_segmentation_losses.append(segmentation_loss.item())
        self.batch_classification_losses.append(classification_loss.item())
        self.batch_total_losses.append(total_loss.item())
    
        return total_loss


    def training_epoch_end(self, outputs):
        # Calculate average losses for the epoch
        avg_segmentation_loss = sum(self.batch_segmentation_losses) / len(self.batch_segmentation_losses)
        avg_classification_loss = sum(self.batch_classification_losses) / len(self.batch_classification_losses)
        avg_total_loss = sum(self.batch_total_losses) / len(self.batch_total_losses)

        print(f"Epoch {self.current_epoch}: "
              f"Avg Segmentation Loss: {avg_segmentation_loss:.4f}, "
              f"Avg Classification Loss: {avg_classification_loss:.4f}, "
              f"Avg Total Loss: {avg_total_loss:.4f}")

        # Store epoch-level losses
        self.epoch_segmentation_losses.append(avg_segmentation_loss)
        self.epoch_classification_losses.append(avg_classification_loss)
        self.epoch_total_losses.append(avg_total_loss)

        # Plot batch losses for the current epoch (if applicable)
        self.plot_batch_losses(epoch=self.current_epoch)

        # Clear batch losses for the next epoch
        self.batch_segmentation_losses.clear()
        self.batch_classification_losses.clear()
        self.batch_total_losses.clear()


    def plot_batch_losses(self, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(self.batch_segmentation_losses, label='Batch Segmentation Loss', color='blue')
        plt.plot(self.batch_classification_losses, label='Batch Classification Loss', color='green')
        plt.plot(self.batch_total_losses, label='Batch Total Loss', color='red')
        
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Batch Losses for Epoch {epoch}')
        plt.legend()
        
        plt.savefig(f'Seg_batch_loss_plot_epoch_with_L2{epoch}.png')
        plt.close()

    def on_train_end(self):
        self.plot_epoch_losses()

    def plot_epoch_losses(self):
        plt.figure(figsize=(10, 5))
        
        plt.plot(self.epoch_segmentation_losses, label='Epoch Segmentation Loss', color='blue')
        plt.plot(self.epoch_classification_losses, label='Epoch Classification Loss', color='green')
        plt.plot(self.epoch_total_losses, label='Epoch Total Loss', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses Across Epochs')
        plt.legend()
        
        plt.savefig(f'Seg_epoch_loss_plot_with_L2.png')
        plt.close()


    def compute_classification_loss(self, egfr_logits, kras_logits, classification_labels):
        egfr_labels = classification_labels['EGFR']  # Shape: [N]
        kras_labels = classification_labels['KRAS']  # Shape: [N]
        
        # Ensure labels are LongTensors
        if egfr_labels.dtype != torch.long:
            egfr_labels = egfr_labels.long()
        if kras_labels.dtype != torch.long:
            kras_labels = kras_labels.long()
    
        # Calculate loss for both EGFR and KRAS
        egfr_loss = nn.CrossEntropyLoss()(egfr_logits, egfr_labels)  # For EGFR
        kras_loss = nn.CrossEntropyLoss()(kras_logits, kras_labels)  # For KRAS
    
        classification_loss = egfr_loss + kras_loss
    
        return classification_loss
    def compute_segmentation_loss(self, segmentation_logits, segmentation_labels):
        if segmentation_labels is not None:
            if segmentation_labels.dim() == 3:  # [batch_size, height, width]
                    upsampled_logits = nn.functional.interpolate(
                        segmentation_logits[:, 0:1, :, :], size=segmentation_labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)  # Now shape is [32, 128, 128]
                    
                    #loss
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    segmentation_loss = loss_fct(upsampled_logits, segmentation_labels.float()) 
        return segmentation_loss

     


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=4e-5)
    
    def train_dataloader(self):
        return self.train_dl




# ### Training

# In[34]:


# Trainer setup
batch_size = 64
num_workers = 0

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

segformer_finetuner = SegformerFinetuner(
    train_dataset.id2label, 
    train_dataloader=train_dataloader, 
    metrics_interval=5,
)

# Trainer initialization
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else 1
)



# In[35]:


trainer


# In[54]:


non_empty_mask_count


# In[48]:


train_dataloader.dataset.mask_paths


# In[94]:


trainer.fit(segformer_finetuner)


# In[33]:


model_save_path = "segformer_finetuned_model_with_L2.ckpt"

# Save the model checkpoint
trainer.save_checkpoint(model_save_path)


# ### How many images have tumor segmentations??

# In[64]:


len(test_dataset)


# In[66]:


def count_non_empty_masks(dataset):
    """Counts how many masks in the dataset have at least one '1' (non-zero pixel)."""
    count = 0
    t=0
    for idx in range(len(dataset)):
        t+=1
        sample = dataset[idx]
        mask = sample['segmentation_labels']  # Mask tensor (binary 0s and 1s)
     

        
        # Check if mask contains at least one '1'
        if torch.any(mask > 0):  # Non-zero value check
            count += 1
            print("masks:", count)
            print("actual",t)

    return count

non_empty_mask_count = count_non_empty_masks(test_dataset)


# In[67]:


non_empty_mask_count


# In[40]:


segformer_finetuner


# In[55]:


test_dataset


# In[57]:


model_save_path = "segformer_finetuned_model_with_L2.ckpt"


# In[58]:


loaded_model = SegformerFinetuner.load_from_checkpoint(
    model_save_path,
    id2label=train_dataset.id2label  # Pass the required argument
)


# In[59]:


loaded_model


# ### Dice Score of only images with Tumor segmentations??

# In[81]:


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchmetrics.classification import Dice

def test_model(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    
    # Set up metrics with appropriate configurations
    dice_metric = Dice(num_classes=2, average='macro').to(device)  # Binary segmentation mask


    dice_scores = []

    with torch.no_grad():
        # Wrap test_dataloader with tqdm for a progress bar
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            pixel_values = batch['pixel_values'].to(device)
            segmentation_labels = batch['segmentation_labels'].to(device)
            classification_labels = batch['classification_labels']

            # Forward pass
            outputs = model(images=pixel_values, segmentation_labels=segmentation_labels, classification_labels=classification_labels)
            segmentation_logits = outputs.logits
            egfr_logits = outputs.egfr_logits
            kras_logits = outputs.kras_logits
         
            
            
            if segmentation_labels.dim() == 3:  # [batch_size, height, width]
                upsampled_logits = nn.functional.interpolate(
                    segmentation_logits[:, 0:1, :, :], size=segmentation_labels.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            
            for i in range(segmentation_labels.size(0)):  # Iterate through the batch
                seg_label = segmentation_labels[i]  # Shape: [H, W]
                pred_logit = upsampled_logits[i]  # Shape: [H, W]
                
                # Check if the segmentation label contains at least one pixel with a value of 1
                if torch.any(seg_label == 1):
                    # Compute Dice score
                  
                    print("yes")
                    pred_binary = (torch.sigmoid(pred_logit) > 0.5).int()
                    dice_score = dice_metric(pred_binary.unsqueeze(0), seg_label.unsqueeze(0).int())
                    dice_scores.append(dice_score.item())

    # Calculate mean and std for Dice scores
    dice_mean, dice_std = np.mean(dice_scores), np.std(dice_scores)

    print("Testing Results:")
    print(f"Dice Score - Mean: {dice_mean:.4f}, Std: {dice_std:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = loaded_model.to(device)
batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
c=test_model(loaded_model, test_dataloader, device)


# In[80]:


c


# In[176]:


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchmetrics.classification import Dice

def test_model(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    
    # Set up metrics with appropriate configurations
    dice_metric = Dice(num_classes=2, average='macro').to(device)  # Binary segmentation mask
    accuracy_metric = MulticlassAccuracy(num_classes=3, average='macro').to(device)  # Multi-class classification
    auroc_metric = MulticlassAUROC(num_classes=3).to(device)  # Multi-class classification for AUROC

    dice_scores = []
    egfr_accuracies = []
    kras_accuracies = []
    egfr_aurocs = []
    kras_aurocs = []

    with torch.no_grad():
        # Wrap test_dataloader with tqdm for a progress bar
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            pixel_values = batch['pixel_values'].to(device)
            segmentation_labels = batch['segmentation_labels'].to(device)
            classification_labels = batch['classification_labels']

            # Forward pass
            outputs = model(images=pixel_values, segmentation_labels=segmentation_labels, classification_labels=classification_labels)
            segmentation_logits = outputs.logits
            egfr_logits = outputs.egfr_logits
            kras_logits = outputs.kras_logits

            
            
            if segmentation_labels.dim() == 3:  # [batch_size, height, width]
                upsampled_logits = nn.functional.interpolate(
                    segmentation_logits[:, 0:1, :, :], size=segmentation_labels.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            

            # Compute Dice score for segmentation
            segmentation_preds = torch.sigmoid(upsampled_logits) > 0.5  # Binarize predictions
            dice_score = dice_metric(segmentation_preds.int(), segmentation_labels.int())
            dice_scores.append(dice_score.item())

            # Compute accuracy and AUROC for classification
            egfr_preds = torch.argmax(egfr_logits, dim=1)
            kras_preds = torch.argmax(kras_logits, dim=1)
            egfr_labels = classification_labels['EGFR'].to(device)
            kras_labels = classification_labels['KRAS'].to(device)
            
            egfr_accuracy = accuracy_metric(egfr_preds, egfr_labels)
            kras_accuracy = accuracy_metric(kras_preds, kras_labels)
            egfr_accuracies.append(egfr_accuracy.item())
            kras_accuracies.append(kras_accuracy.item())

            # AUROC requires probabilities, so apply softmax
            egfr_probs = torch.softmax(egfr_logits, dim=1)
            kras_probs = torch.softmax(kras_logits, dim=1)

            egfr_auroc = auroc_metric(egfr_probs, egfr_labels)
            kras_auroc = auroc_metric(kras_probs, kras_labels)
            egfr_aurocs.append(egfr_auroc.item())
            kras_aurocs.append(kras_auroc.item())
            print("Dice Score:\n", dice_score)
            print("EGFR Accuracy:\n", egfr_accuracy)
            print("KRAS Accuracy:\n", kras_accuracy)
            print(egfr_labels.dim())
            print(egfr_preds.dim())
            print(egfr_logits.dim())

    # Calculate mean and std for each metric
    dice_mean, dice_std = np.mean(dice_scores), np.std(dice_scores)
    egfr_acc_mean, egfr_acc_std = np.mean(egfr_accuracies), np.std(egfr_accuracies)
    kras_acc_mean, kras_acc_std = np.mean(kras_accuracies), np.std(kras_accuracies)
    egfr_auroc_mean, egfr_auroc_std = np.mean(egfr_aurocs), np.std(egfr_aurocs)
    kras_auroc_mean, kras_auroc_std = np.mean(kras_aurocs), np.std(kras_aurocs)

    print("Testing Results:")
    print(f"Dice Score - Mean: {dice_mean:.4f}, Std: {dice_std:.4f}")
    print(f"EGFR Accuracy - Mean: {egfr_acc_mean:.4f}, Std: {egfr_acc_std:.4f}")
    print(f"KRAS Accuracy - Mean: {kras_acc_mean:.4f}, Std: {kras_acc_std:.4f}")
    print(f"EGFR AUROC - Mean: {egfr_auroc_mean:.4f}, Std: {egfr_auroc_std:.4f}")
    print(f"KRAS AUROC - Mean: {kras_auroc_mean:.4f}, Std: {kras_auroc_std:.4f}")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = loaded_model.to(device)
batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_model(loaded_model, test_dataloader, device)


# In[39]:


test_dataset.patient_dirs


# In[170]:


import matplotlib.pyplot as plt

def test_single_image_from_loader(model, test_dataloader, index, device):
    model.eval()  # Set the model to evaluation mode
    
    # Get a batch of size 1
    for i, batch in enumerate(test_dataloader):
        if i == index:  # Pick the batch corresponding to the index
            pixel_values = batch['pixel_values'].to(device)  # Shape [1, C, H, W]
            segmentation_labels = batch['segmentation_labels'].to(device)  # Shape [1, H, W]
            classification_labels = batch['classification_labels']

            # Run inference
            with torch.no_grad():
                outputs = model(images=pixel_values, segmentation_labels=segmentation_labels, classification_labels=classification_labels)
                segmentation_logits = outputs.logits
                egfr_logits = outputs.egfr_logits
                kras_logits = outputs.kras_logits
                
                plt.hist(segmentation_logits.cpu().numpy().flatten(), bins=50)
                plt.title("Histogram of Logits")
                plt.show()
                # Upsample logits for segmentation to match ground truth size
                if segmentation_labels.dim() == 3:  # [batch_size, height, width]
                    upsampled_logits = F.interpolate(
                        segmentation_logits[:, 0:1, :, :], size=segmentation_labels.shape[-2:], mode="bilinear", align_corners=False
                    ).squeeze(1)
                upsampled_logits= torch.abs(upsampled_logits)
                print("-------------------------------------------------------------")
                print(segmentation_labels)
                # Get the predicted segmentation mask
                segmentation_preds = torch.sigmoid(upsampled_logits) >0.5# Binary segmentation prediction
                print(segmentation_preds)
                any_true = segmentation_preds.any()

                if any_true:
                    print("There are True values in the segmentation predictions!")
                else:
                    print("All values in the segmentation predictions are False.")
                # Get predicted class labels for EGFR and KRAS
                egfr_preds = torch.argmax(egfr_logits, dim=1)
                kras_preds = torch.argmax(kras_logits, dim=1)

                # Get actual class labels for EGFR and KRAS
                egfr_labels = classification_labels['EGFR'].to(device)
                kras_labels = classification_labels['KRAS'].to(device)

                # Print predicted and actual values
                print(f"Predicted EGFR Class: {egfr_preds.item()}")
                print(f"Predicted KRAS Class: {kras_preds.item()}")
                print(f"Actual EGFR Class: {egfr_labels.item()}")
                print(f"Actual KRAS Class: {kras_labels.item()}")

                # Plot the results: predicted vs actual
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                
                # Plot the actual mask
                ax[0].imshow(segmentation_labels[0].cpu().numpy(), cmap='gray')
                ax[0].set_title("Actual Mask")
                ax[0].axis('off')
                
                # Plot the predicted mask
                ax[1].imshow(segmentation_preds[0].cpu().numpy(), cmap='gray')
                ax[1].set_title("Predicted Mask")
                ax[1].axis('off')

                # Plot the image itself
                ax[2].imshow(pixel_values[0][0].cpu().numpy(), cmap='gray')  
                ax[2].set_title("Input Image")
                ax[2].axis('off')

                plt.show()
            break 



# In[171]:


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = loaded_model.to(device)

# Define your test dataloader with batch size of 1
batch_size = 1
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Choose the index of the image you want to predict (e.g., index 10)
index = 224
test_single_image_from_loader(model, test_dataloader, index, device)


# In[129]:


test_dataset.image_paths


# In[128]:


test_dataset.mask_paths


# In[137]:


images=test_dataset.image_paths
masks=test_dataset.mask_paths
for image_path, mask_path in zip(images, masks):
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")

    image_dcm = pydicom.dcmread(image_path)
    mask_dcm = pydicom.dcmread(mask_path)
    
    image_array = image_dcm.pixel_array
    mask_array = mask_dcm.pixel_array
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Scan")
    plt.imshow(image_array, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_array, cmap='gray')
    
    plt.show()


# In[160]:


torch.set_printoptions(threshold=torch.inf)
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/ --port 6008')


# In[57]:


import torch
from torchviz import make_dot


# In[58]:


pip install torchviz


# In[59]:


get_ipython().run_line_magic('dot', '')


# In[44]:


len(train_dataset)



# In[46]:


len(test_dataset)


# In[41]:


get_ipython().system('pip install Dice')


# In[42]:


pip install AUROC


# In[43]:


pip install Accuracy


# In[54]:


import pydicom
import matplotlib.pyplot as plt


# In[56]:


train_dataset = SemanticSegmentationDataset("D:\\NSCLC\\train", feature_extractor)

first_item = train_dataset[0] 
patient_id = "R01-004-R01-004" 


images, masks = train_dataset.get_patient_files(patient_id)

print(f"Scans and masks for patient {patient_id}:")

for image_path, mask_path in zip(images, masks):
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")
    

    image_dcm = pydicom.dcmread(image_path)
    mask_dcm = pydicom.dcmread(mask_path)
    
    image_array = image_dcm.pixel_array
    mask_array = mask_dcm.pixel_array
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Scan")
    plt.imshow(image_array, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_array, cmap='gray')
    
    plt.show()


# In[15]:


batch_size = 9
num_workers = 0
chunk_size = 50



# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

# Early stopping and checkpoint callbacks
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=10, 
    verbose=False, 
    mode="min",
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath="checkpoints",
    filename="best-checkpoint",
)

# Trainer setup
from pytorch_lightning.profiler import SimpleProfiler

profiler = SimpleProfiler()
trainer = pl.Trainer(
    accelerator="cpu",
    num_sanity_val_steps=0,
    devices=1,
    max_epochs=2,
    callbacks=[early_stop_callback, checkpoint_callback],
    fast_dev_run=False,  # Set to False to perform the actual training
    profiler=profiler
)

# Load your model


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print(len(train_dataloader))
print(len(train_dataset))
segformer_finetuner = SegformerFinetuner(
    train_dataset.id2label, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    test_dataloader=test_dataloader, 
    metrics_interval=10,
)


# In[16]:


trainer.fit(segformer_finetuner)


# In[34]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[33]:


import tensorboard


# In[29]:


res = trainer.test(ckpt_path="best")


# In[ ]:


trainer


# In[31]:


segformer_finetuner


# In[247]:


pip install tensorboard


# In[32]:


get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')


# In[ ]:


import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation

class CustomSegformer(pl.LightningModule):
    
    def __init__(self, num_classes=2, num_gene_mutations=4, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(CustomSegformer, self).__init__()
        self.num_classes = num_classes
        self.num_gene_mutations = num_gene_mutations
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        
        # Load the pretrained Segformer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            return_dict=False, 
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        
        # Freeze the backbone layers if needed
        for param in self.model.segformer.parameters():
            param.requires_grad = False
        
        # Additional heads for gene mutation predictions
        self.egfr_head = nn.Linear(256, self.num_gene_mutations)
        self.kras_head = nn.Linear(256, self.num_gene_mutations)
        self.alk_head = nn.Linear(256, self.num_gene_mutations)
        
        # Metrics for evaluation
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")
    
    def forward(self, images):
        features = self.model.segformer(images)  # Get features from the second-to-last layer
        logits = self.model.decode_head(features)  # Get segmentation logits
        
        # Gene mutation predictions
        egfr_logits = self.egfr_head(features)
        kras_logits = self.kras_head(features)
        alk_logits = self.alk_head(features)
        
        return logits, egfr_logits, kras_logits, alk_logits
    
    def compute_loss(self, logits, masks, egfr_logits, kras_logits, alk_logits, batch):
        # Compute the loss for segmentation and gene mutation predictions
        loss_fct = nn.CrossEntropyLoss()
        
        seg_loss = loss_fct(logits.view(-1, self.num_classes), masks.view(-1))
        egfr_loss = loss_fct(egfr_logits, batch['egfr_labels'])
        kras_loss = loss_fct(kras_logits, batch['kras_labels'])
        alk_loss = loss_fct(alk_logits, batch['alk_labels'])
        
        total_loss = seg_loss + egfr_loss + kras_loss + alk_loss
        return total_loss
    
    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        
        logits, egfr_logits, kras_logits, alk_logits = self(images)
        
        loss = self.compute_loss(logits, masks, egfr_logits, kras_logits, alk_logits, batch)
        
        self.train_mean_iou.add_batch(predictions=logits.argmax(dim=1).detach().cpu().numpy(), references=masks.detach().cpu().numpy())
        
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False)
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            for k, v in metrics.items():
                self.log(k, v)
            return metrics
        else:
            return {'loss': loss}
    
    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        
        logits, egfr_logits, kras_logits, alk_logits = self(images)
        
        loss = self.compute_loss(logits, masks, egfr_logits, kras_logits, alk_logits, batch)
        
        self.val_mean_iou.add_batch(predictions=logits.argmax(dim=1).detach().cpu().numpy(), references=masks.detach().cpu().numpy())
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        metrics = self.val_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False)
        
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou, "val_mean_accuracy": val_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl


# In[ ]:




