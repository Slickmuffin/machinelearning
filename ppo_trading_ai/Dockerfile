FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310

WORKDIR /app

# Install requirements - same as proven DQN setup
RUN pip install --no-cache-dir torch torchvision numpy pandas scikit-learn google-cloud-storage

# Copy PPO scripts
COPY ppo_agent.py .
COPY ppo_trading_agent.py .
COPY ppo_env.py .

# Set environment variables - proven configuration
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run PPO training
CMD ["python", "ppo_trading_agent.py"]