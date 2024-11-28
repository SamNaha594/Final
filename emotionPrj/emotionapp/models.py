from django.db import models
from datetime import datetime
# Create your models here.

# import torch
# import torch.nn as nn
# import torch.nn.parallel

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParallelModel(nn.Module):
    def __init__(self,num_emotions,num_vocal_channels,num_intensity_levels,num_genders,):
        super().__init__()
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        # 1st parallel 2d conv block
        self.conv_block_one = nn.Sequential(

            # 1st 2D conv layer
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            # 2nd 2D conv layer
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D conv layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        # 2nd parallel 2d conv block
        self.conv_block_two = nn.Sequential(

            # 1st 2D conv layer
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            # 2nd 2D conv layer
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D conv layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        # self.fc1_linear = nn.Linear(512*2+40,num_emotions)
        self.fc1_linear = nn.Linear(512*2+40,num_emotions)  # For emotion
        self.fc2_linear = nn.Linear(512*2+40, num_vocal_channels)  # For vocal channel (e.g., 2: speech/song)
        self.fc3_linear = nn.Linear(512*2+40, num_intensity_levels)  # For intensity (e.g., 2: normal/strong)
        self.fc4_linear = nn.Linear(512*2+40, num_genders)  # For gender (e.g., 2: male/female)


        self.softmax_out = nn.Softmax(dim=1)
    def forward(self,x):
        x = x.to(device)
        conv_embedding_one = nn.parallel.data_parallel(self.conv_block_one, x)
        conv_embedding_two = nn.parallel.data_parallel(self.conv_block_two, x)
        conv_embedding_one = torch.flatten(conv_embedding_one, start_dim=1)
        conv_embedding_two = torch.flatten(conv_embedding_two, start_dim=1)

        x_maxpool = self.transformer_maxpool(x)
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        x = x_maxpool_reduced.permute(2,0,1)

        transformer_output = self.transformer_encoder(x)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        complete_embedding = torch.cat([conv_embedding_one, conv_embedding_two, transformer_embedding], dim=1)

        # output_logits = self.fc1_linear(complete_embedding)
        # output_softmax = self.softmax_out(output_logits)

        emotion_logits = self.fc1_linear(complete_embedding)
        vocal_channel_logits = self.fc2_linear(complete_embedding)
        intensity_logits = self.fc3_linear(complete_embedding)
        gender_logits = self.fc4_linear(complete_embedding)

        emotion_softmax = self.softmax_out(emotion_logits)
        vocal_channel_softmax = self.softmax_out(vocal_channel_logits)
        intensity_softmax = self.softmax_out(intensity_logits)
        gender_softmax = self.softmax_out(gender_logits)

        return emotion_logits, emotion_softmax, vocal_channel_logits, vocal_channel_softmax, intensity_logits, intensity_softmax, gender_logits, gender_softmax



