
# /// script
# dependencies = [
#   "gymnasium[classic-control]",
#   "tensorflow",
# ]
# ///

import numpy as np
import tensorflow as tf
import gymnasium as gym

# --- 1) Environment ---
env = gym.make("CartPole-v1", render_mode="human")  # <- add this
n_obs = env.observation_space.shape[0]     # 4
n_act = env.action_space.n                 # 2 (left/right)

# --- 2) Tiny policy network: obs -> action probabilities ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_obs,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(n_act, activation="softmax"),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
gamma = 0.99  # discount

def sample_action(probs):
    return np.random.choice(n_act, p=probs)

# --- 3) Return (discounted reward) calculator ---
def compute_returns(rewards, gamma=0.99):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns = np.array(returns, dtype=np.float32)
    # normalize for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# --- 4) One update per episode (vanilla REINFORCE) ---
def train_one_episode(env):
    obs, _ = env.reset()
    done = False

    obs_buf, act_buf, rew_buf = [], [], []

    while not done:
        obs_buf.append(obs)
        probs = model(obs[None, :], training=False).numpy()[0]
        a = sample_action(probs)
        act_buf.append(a)

        obs, r, terminated, truncated, _ = env.step(a)
        rew_buf.append(r)
        done = terminated or truncated

    returns = compute_returns(rew_buf, gamma)

    with tf.GradientTape() as tape:
        logits = model(np.array(obs_buf, dtype=np.float32), training=True)
        # Pick log-prob of taken actions
        action_mask = tf.one_hot(act_buf, n_act)
        log_probs = tf.math.log(tf.reduce_sum(logits * action_mask, axis=1) + 1e-8)
        loss = -tf.reduce_mean(log_probs * returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return sum(rew_buf), loss.numpy()

# --- 5) Training loop ---
ep_rewards = []
for ep in range(300):
    R, L = train_one_episode(env)
    ep_rewards.append(R)
    if (ep + 1) % 10 == 0:
        avg = np.mean(ep_rewards[-10:])
        print(f"Episode {ep+1:3d} | return: {R:4.0f} | avg(10): {avg:5.1f} | loss: {L:.3f}")
    # simple success check: CartPole is 'solved' ~475+ avg returns in v1 (harder than v0)
    # but you'll usually see >200 within a couple hundred episodes.

# --- 6) Watch the trained agent (optional) ---
try:
    import time
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        probs = model(obs[None, :], training=False).numpy()[0]
        a = np.argmax(probs)  # greedy for viewing
        obs, r, terminated, truncated, _ = env.step(a)
        total += r
        env.render()
        time.sleep(0.01)
        done = terminated or truncated
    print("Render episode return:", total)
finally:
    env.close()
