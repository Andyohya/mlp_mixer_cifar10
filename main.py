import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time

# 模組載入
from model import MlpMixer
from dataset import load_dataset
from train import create_train_state, train_step
from eval import evaluate
from train import create_train_state, train_step, eval_step, EarlyStopping, split_dataset
from utils import plot_metrics

# CIFAR-10 類別名稱
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins} 分 {secs} 秒"

def preprocess(image):
    return jnp.array(image, dtype=jnp.float32) / 255.0

def visualize_prediction(image, pred_class, true_class):
    plt.imshow(image.astype(np.float32))
    plt.title(f"Prediction: {classes[pred_class]}\nGround Truth: {classes[true_class]}")
    plt.axis('off')
    plt.show()

def evaluate_loss_and_acc(model, params, data):
    total_loss = 0
    total_acc = 0
    total_count = 0
    for imgs, labels in data:
        logits = model.apply(params, imgs, train=False)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.sum(preds == labels)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))
        total_loss += float(loss) * imgs.shape[0]
        total_acc += float(acc)
        total_count += imgs.shape[0]
    return total_loss / total_count, total_acc / total_count

def plot_all_metrics(train_accs, train_losses, test_accs, test_losses, lrs):
    epochs = range(1, len(train_accs) + 1)
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    l1, = ax1.plot(epochs, train_accs, label='Train Accuracy', color='tab:blue', linestyle='-')
    l2, = ax1.plot(epochs, test_accs, label='Test Accuracy', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    l3, = ax2.plot(epochs, train_losses, label='Train Loss', color='tab:red', linestyle='-')
    l4, = ax2.plot(epochs, test_losses, label='Test Loss', color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Learning Rate', color='tab:green')
    l5, = ax3.plot(np.linspace(1, len(train_accs), len(lrs)), lrs, label='Learning Rate', color='tab:green', alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='tab:green')

    lines = [l1, l2, l3, l4, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.title('Training/Test Accuracy, Loss and Learning Rate')
    fig.tight_layout()
    plt.show()

def main():
    rng = jax.random.PRNGKey(0)
    start_time = time.time()

    model = MlpMixer(
        num_classes=10,
        num_blocks=4,
        patch_size=4,
        hidden_dim=64,
        tokens_mlp_dim=128,
        channels_mlp_dim=256,
        dropout_rate=0.4
    )

    batch_size = 128
    num_epochs = 100

    full_train_data = load_dataset(batch_size=batch_size, train=True)
    train_data, val_data = split_dataset(full_train_data, split_ratio=0.9)
    test_data = load_dataset(batch_size=batch_size, train=False)

    steps_per_epoch = len(train_data)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(int(0.1 * total_steps), total_steps - 1)
    weight_decay = 1e-4

    state, schedule = create_train_state(
        rng, model,
        learning_rate=0.003,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )

    train_accs, train_losses, val_accs, val_losses, lrs = [], [], [], [], []
    early_stopping = EarlyStopping(patience=5)

    print("訓練開始，計時開始")

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = 0, 0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_data):
            state, metrics = train_step(state, batch)
            current_step = epoch * steps_per_epoch + batch_idx
            lr = float(schedule(current_step)) if schedule else 0.001
            lrs.append(lr)

            epoch_train_loss += float(metrics['loss'])
            epoch_train_acc += float(metrics['accuracy'])
            batch_count += 1

            if epoch == 0 and batch_idx < 10:
                print(f"Step {current_step}, LR: {lr:.6f}")

        train_accs.append(epoch_train_acc / batch_count)
        train_losses.append(epoch_train_loss / batch_count)

        # ➕ 驗證集評估
        val_loss, val_acc = 0, 0
        for batch in val_data:
            metrics = eval_step(state.params, batch, state.apply_fn)
            val_loss += float(metrics['loss'])
            val_acc += float(metrics['accuracy'])
        val_loss /= len(val_data)
        val_acc /= len(val_data)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} — Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}")
        print(f"Epoch {epoch+1} 耗時: {format_duration(time.time() - epoch_start)}, Acc差: {(train_accs[-1] - val_acc):.4f}")

        if early_stopping.should_stop(val_loss):
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

    test_acc = evaluate(model, state.params, test_data)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print(f"總耗時: {format_duration(time.time() - start_time)}")

    plot_all_metrics(train_accs, train_losses, val_accs, val_losses, lrs)

if __name__ == "__main__":
    main()