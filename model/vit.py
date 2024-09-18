import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 이미지를 패치로 나누고 임베딩하는 합성곱 레이어
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 입력: (batch_size, in_channels, img_size, img_size)
        # 출력: (batch_size, n_patches, embed_dim)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # 어텐션 스케일 팩터

        # Query, Key, Value를 한 번에 계산하는 선형 레이어
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        # 입력: (batch_size, n_patches + 1, dim)
        # n_patches + 1은 클래스 토큰을 포함한 패치의 수
        batch_size, n_tokens, dim = x.shape

        qkv = self.qkv(x)  # (batch_size, n_tokens, 3 * dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)  # (batch_size, n_tokens, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각 (batch_size, n_heads, n_tokens, head_dim)

        # 어텐션 스코어 계산
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, n_heads, n_tokens, n_tokens)
        attn = attn.softmax(dim=-1)  # softmax를 사용하여 가중치 정규화
        attn = self.attn_drop(attn)

        # 가중치를 value에 적용하고 헤드 연결
        out = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        # 첫 번째 정규화 후 어텐션 적용 (잔차 연결 포함)
        x = x + self.attn(self.norm1(x))
        # 두 번째 정규화 후 MLP 적용 (잔차 연결 포함)
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # 클래스 토큰을 추가하고 위치 인코딩 적용
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # 위치 인코딩 추가
        x = self.pos_drop(x)

        # Transformer 블록을 통과
        for block in self.blocks:
            x = block(x)

        # 마지막 정규화
        x = self.norm(x)

        # 분류를 위해 클래스 토큰만 사용
        x = x[:, 0]

        # 분류 헤드
        x = self.head(x)

        return x