import einops
import flax.linen as nn
import jax.numpy as jnp

class MlpBlock(nn.Module):
    mlp_dim: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        return nn.Dense(x.shape[-1])(y)

class MixerBlock(nn.Module):
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, self.dropout_rate, name='token_mixing')(y, train=train)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MlpBlock(self.channels_mlp_dim, self.dropout_rate, name='channel_mixing')(y, train=train)

class MlpMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, train: bool):
        s = self.patch_size
        x = nn.Conv(self.hidden_dim, (s, s), strides=(s, s), name='stem')(x)
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, self.dropout_rate)(x, train=train)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_classes, name='head', kernel_init=nn.initializers.zeros)(x)