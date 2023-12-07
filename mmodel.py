import torch
import torch.nn as nn
from transformers import ViTModel, Wav2Vec2Model
import torch.nn.functional as F

# class M5(nn.Module):
#     def __init__(self, n_input=1, n_output=768, stride=4, n_channel=32):
#         super().__init__()
#         self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
#         self.bn1 = nn.BatchNorm1d(n_channel)
#         self.pool1 = nn.MaxPool1d(4)
#         # ...更多层可以按需添加
#         self.fc1 = nn.Linear(n_channel, n_output)
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         # ...通过更多层
#         x = self.fc1(x.mean(-1))
#         return x



class MultimodalClassifier(nn.Module):
    def __init__(self, audio_feature_size, image_feature_size, num_classes):
        super().__init__()


        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # self.audio_model = M5()

        # Vision Transformer (ViT) 模型
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")  #swim-base-128

        self.env_feature_increaser = nn.Linear(1, 128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.env_feature_decreaser = nn.Linear(7, 1)
        # 特征降维到128维
        self.audio_feature_reducer = nn.Linear(audio_feature_size, 128)
        self.image_feature_reducer = nn.Linear(image_feature_size, 128)

        self.crossatt1 = nn.Linear(128*3,128)
        self.crossatt2 = nn.Linear(128 * 3, 128)
        self.crossatt3 = nn.Linear(128 * 3, 128)
        self.Sigmod=nn.Sigmoid()


        # 分类网络
        self.classifier = nn.Sequential(
            nn.Linear(128*3 , 32),
            nn.ReLU6(),
            nn.Linear(32 , num_classes),
            nn.Softmax(dim=1)  # 分类任务使用Softmax
        )

    def forward(self, input_values, pixel_values,env):
        # 从音频中提取特征
        input_values=input_values.to(torch.float32)

        # audio_features = self.audio_model(input_values).last_hidden_state.mean(dim=1)


        # input_values=input_values.unsqueeze(1)
        audio_features = self.audio_model(input_values)
        audio_features = self.audio_feature_reducer(audio_features.last_hidden_state.mean(dim=1))

        # 从图像中提取特征
        image_features_list = [self.vit(image).last_hidden_state.mean(dim=1) for image in pixel_values]  # images是图像列表
        image_features = torch.mean(torch.stack(image_features_list), dim=0)

        # image_features = self.vit(pixel_values).last_hidden_state.mean(dim=1)
        image_features = self.image_feature_reducer(image_features)


        env=env.unsqueeze(-1)
        env_feature=self.env_feature_increaser(env)
        env_features=self.encoder_layer(env_feature)
        env_features=self.env_feature_decreaser(env_features.permute(0,2,1))
        env_features=env_features.squeeze(-1)
        # 特征组合
        combined_features = torch.cat((audio_features, image_features,env_features), dim=1)

        auf=self.Sigmod(self.crossatt1(combined_features))*audio_features+audio_features
        imf=self.Sigmod(self.crossatt2(combined_features))*image_features+image_features
        enf=self.Sigmod(self.crossatt3(combined_features))*env_features+env_features
        combined_features=torch.cat((imf,auf,enf),dim=1)

        # 分类
        logits = self.classifier(combined_features)

        return logits
