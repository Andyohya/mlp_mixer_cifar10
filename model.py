import flax.linen as nn                     #  匯入 Flax 的高階神經網路模組 API（linen）
import jax.numpy as jnp                     #  使用 JAX 的 NumPy 實作來處理張量運算
import einops                               #  引入 einops 可以簡潔地做 tensor 重新排列操作

class MlpBlock(nn.Module):
    mlp_dim: int                            #  定義一個模組類別，接受超參數 mlp_dim，這代表中間層的隱藏維度
    @nn.compact                             #  @nn.compact 代表你會在這個函式裡面建構所有神經網路層
    def __call__(self, x):            
        original_dim = x.shape[-1]          #  記住輸入資料的 channel 維度，以便最後還原
        x = nn.Dense(self.mlp_dim)(x)       #  第一層全連接層，將輸入轉換成 mlp_dim 維度的表示
        x = nn.gelu(x)                      #  使用 GELU 激勵函數
        x = nn.Dense(original_dim)(x)       #  第二層將資料還原回原始 channel 數，確保 residual connection 可以順利相加
        return x                            #  回傳輸出結果

class MixerBlock(nn.Module):                #  設定兩個超參數，分別代表 token 軸和 channel 軸上的 MLP hidden dim
    tokens_mlp_dim: int
    channels_mlp_dim: int
    @nn.compact
    def __call__(self, x):                    #  開始定義 forward pass
        y = nn.LayerNorm()(x)                 #  對整個輸入做 LayerNorm（先標準化 token）
        y = jnp.swapaxes(y, 1, 2)             #  交換 token 和 channel 軸，準備做 token-wise 的運算
        y = MlpBlock(self.tokens_mlp_dim)(y)  #  使用 MLP block 在 token 軸上做混合
        y = jnp.swapaxes(y, 1, 2)             #  換回原來的軸順序，保證 residual shape 相符
        x = x + y                             #  加回 residual，保持訊息流暢
        y = nn.LayerNorm()(x)                  #  再做一次 LayerNorm，這次準備做 channel mixing
        y = MlpBlock(self.channels_mlp_dim)(y) #  用另一個 MLP block 做 channel 軸的混合
        return x + y                           #  加回 residual，完成一個 Mixer block

class MlpMixer(nn.Module):                     #  定義整體模型的參數
    num_classes: int
    num_blocks: int
    patch_size: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    @nn.compact
    def __call__(self, x):                                                     #  開始模型的 forward pass
        s = self.patch_size                                                    #  取得 patch 大小
        x = nn.Conv(self.hidden_dim, (s, s), strides=(s, s))(x)                #  做 patch embedding（相當於將影像切成非重疊區塊）
        x = einops.rearrange(x, 'n h w c -> n (h w) c')                        #  將 2D patch 展平成 sequence，讓輸入變成 (batch, tokens, hidden_dim)
        for _ in range(self.num_blocks):                                       #  堆疊多個 Mixer block，讓模型進行 token / channel 的交互學習
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm()(x)                                                  #  最後做 LayerNorm，增加穩定性
        x = jnp.mean(x, axis=1)                                                #  對所有 token 做平均，相當於抽取全域表示
        x = nn.Dense(self.num_classes)(x)                                      #  接上分類器，輸出 logits
        return x                                                               #  回傳預測結果

