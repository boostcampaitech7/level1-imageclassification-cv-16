import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 생성자 모델 구조 정의
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()  # 출력을 -1에서 1 사이로 정규화
        )

    def forward(self, z):
        # 잠재 공간 벡터로부터 이미지 생성
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        # 판별자 모델 구조 정의
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 출력을 0에서 1 사이로 정규화 (진짜 이미지일 확률)
        )

    def forward(self, img):
        # 이미지를 평탄화하고 진짜/가짜 판별
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(GAN, self).__init__()
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.latent_dim = latent_dim

    def forward(self, z):
        # 잠재 공간 벡터로부터 이미지 생성
        return self.generator(z)

    def discriminate(self, img):
        # 이미지 판별
        return self.discriminator(img)

    def generate(self, batch_size):
        # 무작위 잠재 공간 벡터로부터 이미지 생성
        z = torch.randn(batch_size, self.latent_dim)
        return self.generator(z)