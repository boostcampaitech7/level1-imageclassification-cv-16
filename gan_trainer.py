import torch
import torch.nn as nn
import torch.optim as optim

class GANTrainer:
    def __init__(self, gan_model, lr=0.0002, b1=0.5, b2=0.999):
        self.gan_model = gan_model
        self.latent_dim = gan_model.latent_dim

        # 최적화 함수 설정
        self.optimizer_G = optim.Adam(self.gan_model.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = optim.Adam(self.gan_model.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # 손실 함수
        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

    def train_step(self, real_imgs, labels):
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # 생성자 학습
        self.optimizer_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim)
        gen_imgs = self.gan_model.generate(z, labels)
        validity, pred_label = self.gan_model.discriminate(gen_imgs)
        g_loss = self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, labels)
        g_loss.backward()
        self.optimizer_G.step()

        # 판별자 학습
        self.optimizer_D.zero_grad()
        real_validity, real_pred_label = self.gan_model.discriminate(real_imgs)
        d_real_loss = (self.adversarial_loss(real_validity, valid) + self.auxiliary_loss(real_pred_label, labels)) / 2

        fake_validity, fake_pred_label = self.gan_model.discriminate(gen_imgs.detach())
        d_fake_loss = (self.adversarial_loss(fake_validity, fake) + self.auxiliary_loss(fake_pred_label, labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                losses = self.train_step(imgs, labels)
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {losses['d_loss']:.4f}] [G loss: {losses['g_loss']:.4f}]")

        print("Training finished.")

    def evaluate(self, dataloader):
        self.gan_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in dataloader:
                _, pred_labels = self.gan_model.discriminate(imgs)
                _, predicted = torch.max(pred_labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on test images: {accuracy:.2f}%')
        self.gan_model.train()