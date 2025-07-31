import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from model import MlpMixer                            # 主模型架構
from dataset import load_dataset                      # 載入與格式化 CIFAR-10
from train import create_train_state, train_step      # 建立初始訓練狀態（帶學習率策略）
from eval import evaluate                             # 評估模型在測試集上的準確率
from utils import plot_metrics                        # 繪製訓練過程中的準確率、損失與學習率變化

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',     # CIFAR-10 的十個類別標籤，用來顯示預測時的名稱
           'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess(image):                                          # 將圖像數據轉換為 JAX 支持的格式
    return jnp.array(image, dtype=jnp.float32) / 255.0          # 將圖像數據轉換為浮點數並歸一化到 [0, 1] 範圍

def visualize_prediction(image, pred_class, true_class):        # 顯示預測結果與真實標籤
    plt.imshow(image.astype(np.float32))                        # 將圖像數據轉換為浮點數以便正確顯示
    plt.title(f"Prediction: {classes[pred_class]}\nGround Truth: {classes[true_class]}")    # 設定標題顯示預測與真實類別
    plt.axis('off')                                             # 關閉坐標軸顯示
    plt.show()                                                  # 顯示圖像

def run_multiple_test_samples(model, params, test_data, num_samples=20):            # 在測試集上隨機選擇20個樣本進行預測
    print(f"\n📷 Running inference on {num_samples} random test images...")        
    correct = 0                                                                     # 計算正確預測的數量
    error_stats = {}                                                                # 用來統計錯誤預測的類別對

    for _ in range(num_samples):                                                    # 隨機選擇樣本進行預測
        imgs, labels = test_data[np.random.randint(len(test_data))]                 # 隨機選擇一個批次的圖像和標籤
        image = imgs[0]                                                             # 將第一張圖像取出來進行預測
        label = int(labels[0])                                                      # 將標籤轉換為整數類型

        image_norm = image[None, ...]                                               # 將圖像數據增加一個維度以符合模型輸入要求
        logits = model.apply(params, image_norm)                                    # 將圖像數據傳入模型進行預測
        pred_class = int(jnp.argmax(logits, axis=-1)[0])                            # 獲取預測的類別索引

        visualize_prediction(image, pred_class, label)
        print(f"🔹 Predicted: {classes[pred_class]}")
        print(f"🔸 Ground Truth: {classes[label]}\n")

        if pred_class == label:
            correct += 1
        else:
            true_label = classes[label]
            pred_label = classes[pred_class]
            error_stats.setdefault((true_label, pred_label), 0)      
            error_stats[(true_label, pred_label)] += 1

    acc = correct / num_samples    
    print(f"✅ Correct Predictions: {correct}/{num_samples}")
    print(f"📈 Accuracy: {acc:.2f}")  

    if error_stats:
        print("\n❌ Misclassifications:")
        for (true_cls, pred_cls), count in error_stats.items():
            print(f"- {true_cls} → {pred_cls}: {count} time(s)")

def main():
    rng = jax.random.PRNGKey(0)

    model = MlpMixer(
        num_classes=10,
        num_blocks=4,
        patch_size=4,
        hidden_dim=64,
        tokens_mlp_dim=128,
        channels_mlp_dim=256,
    )

    batch_size = 128
    num_epochs = 50

    train_data = load_dataset(batch_size=batch_size, train=True)  
    test_data = load_dataset(batch_size=batch_size, train=False)

    steps_per_epoch = len(train_data)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(int(0.1 * total_steps), total_steps - 1)

    state = create_train_state(
        rng, model,
        learning_rate=0.001,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps
    )
    accs, losses, lrs = [], [], []

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_data):
            state, metrics = train_step(state, batch)
            current_step = epoch * steps_per_epoch + batch_idx
            if hasattr(state.tx, 'schedule'):
                lr = state.tx.schedule(current_step)
            else:
                lr = 0.001
            lrs.append(float(lr))
            if epoch == 0 and batch_idx < 10:
                print(f"Step {current_step}, LR: {float(lr):.6f}")
        accs.append(metrics['accuracy'])
        losses.append(metrics['loss'])
        print(f"Epoch {epoch+1} — Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, LR: {float(lr):.6f}")

    test_acc = evaluate(model, state.params, test_data)  
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    plot_metrics(accs, "Accuracy")
    plot_metrics(losses, "Loss")
    plot_metrics(lrs, "Learning Rate")

    print("\n🔍 Multiple image inference from test set:")
    run_multiple_test_samples(model, state.params, test_data, num_samples=20)

if __name__ == "__main__": 
    main()